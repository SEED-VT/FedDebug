"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import utils
from cryptography.exceptions import InvalidSignature
from ibmfl.crypto.infra.crypto_asym_int import CryptoAsym
from ibmfl.crypto.crypto_exceptions import *

class CryptoAsymRsa(CryptoAsym):
    """
    This class implements the interface for asymmetric encryption functions using RSA. 
    """

    KEY_SIZE = 4096
    PUBLIC_EXPONENT = 65537
    CRYPTO_HASH = hashes.SHA256()
    SIGNATURE_BLOCK_SIZE = 1024*1024

    def __init__(self, key_file: str = None, password: bytes = None, **kwargs):
        super(CryptoAsymRsa, self).__init__(key_file, password)
        if key_file is None:
            self.private_key = CryptoAsymRsa.generate_key()
        else:
            self.private_key = self._read_key_file(key_file, password)

    def generate_key():
        private_key = rsa.generate_private_key(
            public_exponent=CryptoAsymRsa.PUBLIC_EXPONENT,
            key_size=CryptoAsymRsa.KEY_SIZE,
        )
        return private_key

    def get_public_key(self, type: str = "obj"):
        if self.private_key is None:
            raise KeyDistributionInputException("self.private_key is None")
        if type == "obj":
            ret = self.private_key.public_key()
        elif type == "pem":
            ret = self.private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        else:
            raise KeyDistributionInputException("Invalid type=" + repr(type))
        return ret

    def _read_key_file(self, file_path: str, password):
        with open(file_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=password,
            )
        return private_key

    def write_key_file(self, file_path: str, password: bytes):
        if self.private_key is None:
            raise KeyDistributionInputException("self.private_key is None")
        encryption_algorithm = serialization.NoEncryption()
        if password is not None:
            encryption_algorithm = serialization.BestAvailableEncryption(password)
        pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption_algorithm
        )
        with open(file_path, "wb") as key_file:
            key_file.write(pem)
        return

    def encrypt(self, plain_data: bytes) -> bytes:
        if self.private_key is None:
            raise KeyDistributionInputException("self.private_key is None")
        cipher_data = self.private_key.public_key().encrypt(
            plain_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=CryptoAsymRsa.CRYPTO_HASH),
                algorithm=CryptoAsymRsa.CRYPTO_HASH,
                label=None
            )
        )
        return cipher_data

    def decrypt(self, cipher_data: bytes) -> bytes:
        if self.private_key is None:
            raise KeyDistributionInputException("self.private_key is None")
        plain_data = self.private_key.decrypt(
            cipher_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=CryptoAsymRsa.CRYPTO_HASH),
                algorithm=CryptoAsymRsa.CRYPTO_HASH,
                label=None
            )
        )
        return plain_data

    def encrypt_wkey(public_key, plain_data: bytes) -> bytes:
        if isinstance(public_key, rsa.RSAPublicKey):
            pb_key = public_key
        elif isinstance(public_key, bytes):
            pb_key = load_pem_public_key(public_key)
        else:
            raise KeyDistributionInputException("Invalid type of public_key=" + repr(type(public_key)))
        cipher_data = pb_key.encrypt(
            plain_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=CryptoAsymRsa.CRYPTO_HASH),
                algorithm=CryptoAsymRsa.CRYPTO_HASH,
                label=None
            )
        )
        return cipher_data

    def get_signature(self, data: bytes) -> bytes:
        if self.private_key is None:
            raise KeyDistributionInputException("self.private_key is None")
        hasher = hashes.Hash(CryptoAsymRsa.CRYPTO_HASH)
        n_blocks = int(len(data) / CryptoAsymRsa.SIGNATURE_BLOCK_SIZE)
        for i in range(n_blocks):
            hasher.update(data[i*CryptoAsymRsa.SIGNATURE_BLOCK_SIZE:(i+1)*CryptoAsymRsa.SIGNATURE_BLOCK_SIZE])
        if len(data) % CryptoAsymRsa.SIGNATURE_BLOCK_SIZE > 0:
            hasher.update(data[n_blocks*CryptoAsymRsa.SIGNATURE_BLOCK_SIZE:len(data)])
        digest = hasher.finalize()
        signature = self.private_key.sign(
            digest,
            padding.PSS(
                mgf=padding.MGF1(CryptoAsymRsa.CRYPTO_HASH),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            utils.Prehashed(CryptoAsymRsa.CRYPTO_HASH)
        )
        return signature

    def verify_signature(public_key, signature: bytes, data: bytes) -> bool:
        if isinstance(public_key, rsa.RSAPublicKey):
            pb_key = public_key
        elif isinstance(public_key, bytes):
            pb_key = load_pem_public_key(public_key)
        else:
            raise KeyDistributionInputException("Invalid type of public_key=" + repr(type(public_key)))
        hasher = hashes.Hash(CryptoAsymRsa.CRYPTO_HASH)
        n_blocks = int(len(data) / CryptoAsymRsa.SIGNATURE_BLOCK_SIZE)
        for i in range(n_blocks):
            hasher.update(data[i*CryptoAsymRsa.SIGNATURE_BLOCK_SIZE:(i+1)*CryptoAsymRsa.SIGNATURE_BLOCK_SIZE])
        if len(data) % CryptoAsymRsa.SIGNATURE_BLOCK_SIZE > 0:
            hasher.update(data[n_blocks*CryptoAsymRsa.SIGNATURE_BLOCK_SIZE:len(data)])
        digest = hasher.finalize()
        try:
            pb_key.verify(
                signature,
                digest,
                padding.PSS(
                    mgf=padding.MGF1(CryptoAsymRsa.CRYPTO_HASH),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                utils.Prehashed(CryptoAsymRsa.CRYPTO_HASH)
            )
        except InvalidSignature:
            return False
        return True
