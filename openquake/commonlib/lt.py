# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2020, GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.

from itertools import product
from collections import namedtuple
import numpy
import toml

from openquake.baselib import hdf5, general
from openquake.hazardlib import nrml


class InvalidLogicTree(Exception):
    pass


class Branch(tuple):
    def __new__(cls, bsid, brid, uncertainty, weight, childset=None):
        self = tuple.__new__(cls, (bsid, brid, uncertainty, weight))
        self.bsid = bsid
        self.brid = brid
        self.uncertainty = uncertainty
        self.weight = weight
        self.childset = childset
        return self

    def __repr__(self):
        return self.brid


BranchSet = namedtuple('BranchSet', 'branches attrs')
Realization = namedtuple('Realization', 'value weight lt_path ordinal')


def sample(weighted_objects, num_samples, seed):
    """
    Take random samples of a sequence of weighted objects

    :param weighted_objects:
        A finite sequence of objects with a `.weight` attribute.
        The weights must sum up to 1.
    :param num_samples:
        The number of samples to return
    :param seed:
        A random seed
    :return:
        A subsequence of the original sequence with `num_samples` elements
    """
    weights = []
    for obj in weighted_objects:
        w = obj.weight
        if isinstance(obj.weight, float):
            weights.append(w)
        else:
            weights.append(w['weight'])
    numpy.random.seed(seed)
    idxs = numpy.random.choice(len(weights), num_samples, p=weights)
    # NB: returning an array would break things
    return [weighted_objects[idx] for idx in idxs]


# manage the legacy node logicTreeBranchingLevel
def _bsnodes(fname, branchinglevel):
    if branchinglevel.tag.endswith('logicTreeBranchingLevel'):
        if len(branchinglevel) > 1:
            raise InvalidLogicTree(
                '%s: Branching level %s has multiple branchsets'
                % (fname, branchinglevel['branchingLevelID']))
        return branchinglevel.nodes
    elif branchinglevel.tag.endswith('logicTreeBranchSet'):
        return [branchinglevel]
    else:
        raise ValueError('Expected BranchingLevel/BranchSet, got %s' %
                         branchinglevel)


def count_rlzs(branch):
    if not branch.childset:
        return 1
    return sum(map(count_rlzs, branch.childset.branches))


def full_enum(branch):
    print(branch)
    if not branch.childset:
        yield branch
    else:
        for br in branch.childset.branches:
            yield from full_enum(br)


class LogicTree(object):
    """
    A simple logic tree object build over a list of branchsets,
    serializable to HDF5 and XML and with methods __iter__ and sample.
    """
    @classmethod
    def from_xml(cls, fname):
        branchsets = []
        for blnode in nrml.read(fname).logicTree:
            [bsnode] = _bsnodes(fname, blnode)
            attrs = bsnode.attrib.copy()
            attrs['bsid'] = attrs.pop('branchSetID')  # rename
            bset = BranchSet([], attrs)
            # example: bsnode.attrib = {
            # 'uncertaintyType': 'gmpeModel',
            # 'branchSetID': 'bs1',
            # 'applyToTectonicRegionType': 'Active Shallow Crust'}
            for brnode in bsnode:
                branch = Branch(
                    attrs['bsid'], brnode['branchID'],
                    ~brnode.uncertaintyModel, ~brnode.uncertaintyWeight)
                bset.branches.append(branch)
            branchsets.append(bset)
        return cls(branchsets)

    def __init__(self, branchsets):
        # example: branchsets [a1 a2] [b1 b2 b3] [c1 c2 c3]
        # with applyToBranches=a2 and applyToBranches=b1 b2
        self.branchset = {bs.attrs['bsid']: bs for bs in branchsets}
        for i, childset in enumerate(branchsets[1:]):
            atb = childset.attrs.get('applyToBranches')  # a2, then b1 b2
            for branch in branchsets[i].branches:  # parent branches
                if not atb or branch.brid in atb:
                    branch.childset = childset

    def rootbranches(self):
        return self.branchset[next(iter(self.branchset))].branches

    def reduce(self, bsids):
        """
        :returns: a reduced LogicTree defined on the given branchset IDs
        """
        return self.__class__([self.branchset[bsid] for bsid in bsids])

    def sample(self, n, seed):
        """
        :param n: number of samples
        :param seed: random seed
        :returns: n Realization objects with weight 1/n
        """
        brlists = [sample(self.branchset[bsid].branches, n, seed + i)
                   for i, bsid in enumerate(self.branchset)]
        weight = 1. / n
        for i in range(n):
            lt_path = []
            value = []
            for brlist in brlists:  # there is branch list for each bsid
                branch = brlist[i]
                lt_path.append(branch.brid)
                value.append(branch.uncertainty)
            yield Realization(tuple(value), weight, tuple(lt_path), i)

    def gen_rlzs(self, num_samples=0, seed=42):
        """
        Yield :class:`openquake.commonlib.lt.Realization` instances
        """
        if num_samples:
            # random sampling of the logic tree
            yield from self.sample(num_samples, seed)
        else:
            # full enumeration
            groups = [bset.branches for bset in self.branchset.values()]
            for i, branches in enumerate(product(*groups)):
                weight = 1
                lt_path = []
                value = []
                for branch in branches:
                    lt_path.append(branch.brid)
                    weight *= branch.weight
                    value.append(branch.uncertainty)
                yield Realization(tuple(value), weight, tuple(lt_path), i)

    def to_xml(self, fileobj):
        raise NotImplementedError

    def __toh5__(self):
        bsetdict = {}
        for bset in self.branchset.values():
            attrs = bset.attrs.copy()
            bsetdict[attrs.pop('bsid')] = attrs
        dt = [('bsid', hdf5.vstr), ('brid', hdf5.vstr),
              ('uncertainty', hdf5.vstr), ('weight', float)]
        branches = [b for bset in self.branchset.values()
                    for b in bset.branches]
        dic = {'branches': numpy.array(branches, dt),
               'branchsets': toml.dumps(bsetdict)}
        return dic, {}

    def __fromh5__(self, dic, attrs):
        self.branchset = {}
        dic1 = general.group_array(dic['branches'][()], 'bsid')
        attrs = toml.loads(dic['branchsets'][()])  # bsid -> attrs
        for bsid, branches in dic1.items():
            self.branchset[bsid] = BranchSet([], attrs[bsid])
            for branch in branches:
                self.branchset[bsid].branches.append(Branch(*branch))

    def __repr__(self):
        return '<%s%s>' % (self.__class__.__name__, list(self.branchset))


if __name__ == '__main__':
    import sys
    lt = LogicTree.from_xml(sys.argv[1])
    with hdf5.File('/tmp/x.h5', 'w') as f:
        f['lt'] = lt
    with hdf5.File('/tmp/x.h5', 'r') as f:
        lt = f['lt']
    for rlz in lt.gen_rlzs():
        print(rlz)
