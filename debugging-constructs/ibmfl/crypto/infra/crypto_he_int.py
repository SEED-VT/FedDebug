"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import abc

class CryptoHe(abc.ABC):
    """
    This class defines an interface for HE keys generation functions. 
    """

    @abc.abstractmethod
    def __init__(self, config = None):
        return

    @abc.abstractmethod
    def generate_keys(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_public_key(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_private_key(self):
        raise NotImplementedError
