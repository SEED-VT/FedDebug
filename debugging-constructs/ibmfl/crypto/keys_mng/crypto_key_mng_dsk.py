"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
from ibmfl.crypto.keys_mng.crypto_key_mng_int import KeyManager
from ibmfl.crypto.crypto_exceptions import KeyManagerException

class LocalDiskKeyManager(KeyManager):

    def __init__(self, config):
        """ Initialize Key from local key file"""
        if config and 'files' not in config:
            raise KeyManagerException('key file path is not provided.')
        self.keys = config

    def initialize_keys(self, **kwargs):
        """ Initialize the keys directly from the config file"""
        return self.keys
