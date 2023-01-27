"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""

import logging	
import numpy as np
import hashlib
import jsonpickle

"""
Module providing hashing functions helpful for loging config information to etb
"""

#Set up logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)-6s %(name)s :: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')




def hash_str(input):
	h = hashlib.sha512()
	h.update(input.encode('utf-8'))
	result = h.hexdigest()
	return result


def hash_np_array(np_array):
	string = np.array_str(np_array)
	result = hash_str(string)
	return result

def hash_model_update(model_update):
	the_model = jsonpickle.encode(model_update)
	result = hashlib.sha512(the_model.encode('utf-8')).hexdigest()
	return result