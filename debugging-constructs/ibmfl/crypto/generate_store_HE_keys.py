"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import pyhelayers as pyhe
from ibmfl.crypto.crypto_exceptions import KeyManagerException
import logging

logger = logging.getLogger(__name__)
    
def generate_store_HE_keys(path_for_public_key=None, 
                           path_for_secret_key=None, 
                           HE_params=None):
    """
    Generate and store Homomorphic Encryption (HE) keys
    :param path_for_public_key: Path at which the public key file \
                                of HELayers will be stored
    :type path_for_public_key: `str`
    :param path_for_secret_key: Path at which the secret key file \
                                of HELayers will be stored
    :type path_for_secret_key: `str`
    :param HE_params: Parameters for the HE scheme, this is a dictionary \
                      containing the keys `security_level`, 
                      `multiplication_depth`, `integer_part_precision`, \
                      and `fractional_part_precision`
    :type HE_params: `dict`
    :return: None
    """
    try:
        if path_for_public_key is None or \
            path_for_secret_key is None:
                raise KeyManagerException('Path to store keys is not provided.')

        if HE_params is None:
            security_level = 128
            integer_part_precision =  10
            fractional_part_precision = 20
            multiplication_depth = 1
            num_slots = 2048
        elif not isinstance(HE_params, dict):
            raise KeyManagerException('HE parameters need to be provided '
                                  'as a dictionary.')
        else:
            num_slots = HE_params.get('num_slots')
            multiplication_depth = HE_params.get('multiplication_depth')
            fractional_part_precision = HE_params.get('fractional_part_precision')
            integer_part_precision = HE_params.get('integer_part_precision')
            security_level = HE_params.get('security_level')
            if num_slots is None or \
               multiplication_depth is None or \
               fractional_part_precision is None or \
               integer_part_precision is None or \
               security_level is None:
                   raise KeyManagerException('Missing parameter for HE key generation.')
    
    
        requirement = pyhe.HeConfigRequirement(num_slots=num_slots,
                                               multiplication_depth=multiplication_depth,
                                               fractional_part_precision=fractional_part_precision,
                                               integer_part_precision=integer_part_precision,
                                               security_level=security_level)
        context = pyhe.DefaultContext()
        context.init(requirement)
        context.save_secret_key_to_file(path_for_secret_key)
        context.save_to_file(path_for_public_key)  
        # note: the context file does not include keys, similar to public parameters
        
        print('HE public key context file is located at {}'.format(path_for_public_key))
        print('HE secret key file is located at {}'.format(path_for_secret_key))
    except RuntimeError as err:
        print('HELayer Error:', err)
    except KeyManagerException as ex:
        print('Error:', ex) 
