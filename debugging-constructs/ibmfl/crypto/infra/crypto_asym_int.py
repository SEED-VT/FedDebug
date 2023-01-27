"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import abc

class CryptoAsym(abc.ABC):
    """
    This class defines an interface for asymmetric encryption functions. 
    """

    @abc.abstractmethod
    def __init__(self, key_file: str = None, password: bytes = None, **kwargs):
        return

    @abc.abstractstaticmethod
    def generate_key():
        raise NotImplementedError

    @abc.abstractmethod
    def get_public_key(self, type: str):
        raise NotImplementedError

    @abc.abstractmethod
    def write_key_file(self, file_path: str, password: bytes):
        raise NotImplementedError

    @abc.abstractmethod
    def encrypt(self, plain_data: bytes) -> bytes:
        raise NotImplementedError

    @abc.abstractmethod
    def decrypt(self, cipher_data: bytes) -> bytes:
        raise NotImplementedError

    @abc.abstractstaticmethod
    def encrypt_wkey(public_key, plain_data: bytes) -> bytes:
        raise NotImplementedError

    @abc.abstractmethod
    def get_signature(self, data: bytes) -> bytes:
        raise NotImplementedError

    @abc.abstractstaticmethod
    def verify_signature(public_key, signature: bytes, data: bytes) -> bool:
        raise NotImplementedError
