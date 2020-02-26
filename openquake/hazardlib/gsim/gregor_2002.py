# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2012-2020 GEM Foundation
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
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

"""
Module exports :class:`GregorEtAl2002SInter`.
"""
import numpy as np
import copy

from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, SA


class GregorEtAl2002SInter(GMPE):
    """
    Implements GMPE developed by N.J. Gregor, W.J. Silva, I.G. Wong, and R.R. Young and published as "Ground-Motion Attenuation Relationships for Cascadia Subduction Zone Megathrust Earthquakes" (Bulletin of the Seismological Society of America Volume 92,
    No. 5, pages 1923-1932, 2002).
    This class implements the equations for 'Subduction Interface' (that's why
    the class name ends with 'SInter').
    """
    #: Supported tectonic region type is subduction interface
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTERFACE

    #: Supported intensity measure types are spectral acceleration,
    #: and peak ground acceleration, see table 2, page 67.
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        PGA,
        SA
    ])

    #: Supported intensity measure component is the average horizontal
    #: component
    #: attr:`~openquake.hazardlib.const.IMC.AVERAGE_HORIZONTAL`, see
    #: paragraph: 'Analysis of peak horizontal accelerations', p. 59.
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.AVERAGE_HORIZONTAL

    #: Supported standard deviation types is total, table 2, page 67.
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL
    ])

    #: Required site parameters is Vs30, used to distinguish between rock
    #: and soil sites, see paragraph 'Strong Motion Data Base', page 59.
    REQUIRES_SITES_PARAMETERS = set(('vs30', ))

    #: Required rupture parameters are magnitude and focal depth, see
    #: equations 1 and 2, pages 59 and 66, respectively.
    REQUIRES_RUPTURE_PARAMETERS = set(('mag', ))

    #: Required distance measure is Rrup, see equations 1 and 2, page 59 and
    #: 66, respectively.
    REQUIRES_DISTANCES = set(('rrup', ))

    #: Vs30 value representing typical rock conditions in California.
    ROCK_VS30 = 760

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        assert all(stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
                   for stddev_type in stddev_types)

        mean = np.zeros_like(sites.vs30)
        stddevs = [np.zeros_like(sites.vs30) for _ in stddev_types]

        idx_rock = sites.vs30 >= self.ROCK_VS30
        idx_soil = sites.vs30 < self.ROCK_VS30

        if idx_rock.any():
            C = self.COEFFS_ROCK[imt]
            self._compute_mean(C, rup.mag, 
                               dists.rrup, mean, idx_rock)
            self._compute_std(C, stddevs, idx_rock)


        if idx_soil.any():
            C = self.COEFFS_SOIL[imt]
            self._compute_mean(C, rup.mag,
                               dists.rrup, mean, idx_soil)
            self._compute_std(C, stddevs, idx_soil)

        return mean, stddevs

    def _compute_mean(self, C, mag, rrup, mean, idx):
        """
        Compute mean for subduction interface events, as explained in table 2,
        page 67.
        """
        mean[idx] = ( C['C1'] + mag * C['C2'] + (np.log(rrup[idx] + np.exp(C['C5']))) * (C['C3'] + mag * C['C4']) + C['C6'] * (mag - 10) ** 3 )

    def _compute_std(self, C, stddevs, idx):
        """
        Collect standard deviation of calculation.
        """

        for stddev in stddevs:
            stddev[idx] += C['Sig'] 

    #: Coefficient table containing soil coefficients,
    #: taken from table 3
    COEFFS_SOIL = CoeffsTable(sa_damping=5, table="""\
    IMT       C1         C2          C3        C4        C5    C6        Sig
    pga       23.8613    -2.2742     -4.8803   0.4399    4.7   0.0366    0.5436
    0.010     25.4516    -2.4206     -5.1071   0.4605    4.8   0.0372    0.5422
    0.020     25.4339    -2.4185     -5.1044   0.4602    4.8   0.0370    0.5422
    0.025     25.4200    -2.4168     -5.1026   0.4600    4.8   0.0369    0.5464
    0.032     25.3849    -2.4127     -5.0977   0.4594    4.8   0.0366    0.5422
    0.040     22.7042    -2.1004     -4.9006   0.4353    4.8   0.0164    0.5241
    0.050     23.2948    -2.1619     -4.8855   0.4332    4.8   0.0263    0.5319
    0.056     23.2165    -2.1528     -4.8744   0.4319    4.8   0.0255    0.5413
    0.0625    24.7067    -2.2814     -5.0947   0.4509    4.9   0.0245    0.5480
    0.070     24.9425    -2.3045     -5.0672   0.4476    4.9   0.0295    0.5413
    0.083     26.5395    -2.4402     -5.3025   0.4677    5.0   0.0276    0.5835
    0.100     29.9693    -2.7254     -5.8054   0.5098    5.2   0.0226    0.5926
    0.125     35.6660    -3.1853     -6.6251   0.5769    5.5   0.0123    0.6665
    0.143     50.7368    -4.5292     -8.7213   0.7649    5.9   0.0108    0.6532
    0.167     55.6402    -4.9662     -9.5555   0.8435    6.0   -0.0070   0.6393
    0.200     75.8218    -6.8396     -12.0687  1.0753    6.3   0.0096    0.6618
    0.250     100.3357   -9.0324     -15.3511  1.3731    6.6   -0.0043   0.6371
    0.330     71.7967    -6.4990     -11.6056  1.0415    6.2   0.0102    0.6431
    0.400     67.3720    -6.1755     -11.1567  1.0167    6.1   0.0035    0.6699
    0.500     56.0088    -5.1176     -9.5083   0.8632    5.9   0.0164    0.6139
    0.770     26.3013    -2.4482     -5.3818   0.4957    4.8   0.0259    0.7256
    1.000     17.2330    -1.5506     -4.3287   0.3930    4.2   0.0133    0.6606
    1.670     11.9971    -1.1180     -2.9451   0.2639    3.7   0.0538    0.6837
    2.000     17.9124    -1.7505     -3.8150   0.3574    4.1   0.0583    0.6276
    2.500     16.1666    -1.5091     -3.7101   0.3344    4.1   0.0473    0.6676
    5.000     7.4856     -0.8360     -2.0627   0.1779    -0.2  0.0821    0.8207
        """)

    #: Coefficient table containing rock coefficients,
    #: taken from table 2
    COEFFS_ROCK = CoeffsTable(sa_damping=5, table="""\
    IMT   C1        C2         C3       C4       C5     C6        Sig
    pga   21.0686   -1.7712    -5.0631   0.4153   4.2    0.0017   0.7240
    0.010 20.9932   -1.7658    -5.0404   0.4132   4.2    0.0226   0.7195
    0.020 21.072    -1.772     -5.0529   0.4142   4.2    0.0025   0.7195
    0.025 21.152    -1.779     -5.0663   0.4154   4.2    0.0023   0.7235
    0.032 21.366    -1.797     -5.1036   0.4187   4.2    0.0017   0.7221
    0.040 17.525    -1.339     -4.8602   0.3868   4.2    -0.0318  0.6969
    0.050 19.347    -1.519     -4.9731   0.3960   4.2    -0.0155  0.7086
    0.056 20.774    -1.625     -5.1875   0.4118   4.3    -0.0155  0.7215
    0.063 21.331    -1.672     -5.2561   0.4173   4.3    -0.0146  0.7302
    0.071 24.221    -1.924     -5.6250   0.4478   4.4    -0.0071  0.7326
    0.083 24.950    -1.979     -5.6696   0.4493   4.4    -0.0018  0.7815
    0.100 30.005    -2.349     -6.3862   0.5009   4.7    -0.0019  0.7954
    0.125 39.719    -3.090     -7.8541   0.6161   5.1    -0.0064  0.8605
    0.143 43.414    -3.385     -8.3122   0.6513   5.2    -0.0001  0.8544
    0.167 39.579    -2.957     -7.9723   0.6139   5.2    -0.0264  0.8478
    0.200 39.345    -3.087     -7.6002   0.5972   5.1    0.0060   0.8679
    0.250 37.690    -2.960     -7.3790   0.5842   5.1    -0.0023  0.8444
    0.333 34.787    -2.899     -6.7855   0.5616   4.9    0.0256   0.8776
    0.400 33.393    -2.776     -6.9595   0.5863   4.9    -0.0039  0.8801
    0.500 29.159    -2.424     -6.2114   0.5216   4.7    0.0161   0.8039
    0.769 15.279    -1.220     -4.3240   0.3618   3.9    -0.0011  0.8295
    1.000 6.528     -0.406     -3.1991   0.2589   3.2    -0.0225  0.7567
    1.667 7.467     -0.676     -2.6465   0.2193   2.8    0.0416   0.6943
    2.000 8.657     -0.851     -2.7398   0.2339   2.8    0.0370   0.6305
    2.500 6.637     -0.651     -2.3124   0.1879   2.8    0.0364   0.6657
    5.000 8.013     -0.943     -2.4087   0.2154   2.3    0.0647   0.7730
        """)




