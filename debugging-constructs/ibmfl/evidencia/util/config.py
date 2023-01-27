"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging	
import copy
import json

"""
Module providing utility functions helpful for loging config information to etb
"""

#Set up logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)-6s %(name)s :: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')




def config_to_json_str(config):
	result = copy.deepcopy(config)
	change_cls_ref_to_str(result)
	return json.dumps(result)


def change_cls_ref_to_str(d):
    for k, v in d.items():
        if isinstance(v, dict):
            change_cls_ref_to_str(v)
        if k == 'cls_ref':
            d[k] = str(d[k].__name__)