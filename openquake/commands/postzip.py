#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2017-2020 GEM Foundation
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

import json
import logging
import requests
from openquake.baselib import sap, config
from openquake.calculators.extract import WebAPIError


@sap.Script
def postzip(zipfile):
    """Post a zipfile to the WebUI"""
    sess = requests.Session()
    if config.webapi.username:
        login_url = '%s/accounts/ajax_login/' % config.webapi.server
        logging.info('POST %s', login_url)
        resp = sess.post(
            login_url, data=dict(username=config.webapi.username,
                                 password=config.webapi.password))
        if resp.status_code != 200:
            raise WebAPIError(resp.text)
    resp = sess.post("%s/v1/calc/run" % config.webapi.server, {},
                     files=dict(archive=open(zipfile, 'rb')))
    print(json.loads(resp.text))


postzip.arg('zipfile', 'archive with the files of the computation')

if __name__ == '__main__':
    postzip.callfunc()
