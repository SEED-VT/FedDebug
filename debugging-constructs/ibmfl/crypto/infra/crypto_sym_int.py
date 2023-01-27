"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import abc
from ibmfl.crypto.crypto_exceptions import *

class CryptoSym(abc.ABC):
    """
    This class defines an interface for symmetric encryption functions. 
    """

    @abc.abstractmethod
    def __init__(self, key: bytes = None, **kwargs):
        self.key = key
        self.cipher = None
        return

    @abc.abstractmethod
    def generate_key(self):
        raise NotImplementedError

    def get_key(self) -> bytes:
        if self.key is None:
            raise KeyDistributionInputException("self.key is None")
        return self.key

    def encrypt(self, plain_data: bytes) -> bytes:
        if self.cipher is None:
            raise KeyDistributionInputException("self.cipher is None")
        return self.cipher.encrypt(plain_data)        

    def decrypt(self, cipher_data: bytes) -> bytes:
        if self.cipher is None:
            raise KeyDistributionInputException("self.cipher is None")
        return self.cipher.decrypt(cipher_data)
