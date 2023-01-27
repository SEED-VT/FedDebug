"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from ibmfl.crypto.infra.crypto_cert_int import CryptoCert
from ibmfl.crypto.crypto_exceptions import *

class CryptoCertRsa(CryptoCert):
    """
    This class implements the interface for certificate verification functions using x509 and RSA. 
    """

    def __init__(self, ca_cert_file_path: str, my_cert_file_path: str, **kwargs):
        super(CryptoCertRsa, self).__init__(ca_cert_file_path, my_cert_file_path)
        if ca_cert_file_path is None:
            raise KeyDistributionInputException("ca_cert_file_path is None")
        else:
            with open(ca_cert_file_path, "rb") as ca_cert_file:
                self.ca_cert = x509.load_pem_x509_certificate(ca_cert_file.read())
            self.ca_public_key = self.ca_cert.public_key()
            if not isinstance(self.ca_public_key, rsa.RSAPublicKey):
                raise KeyDistributionInputException("Invalid type of public key in certificate=" + repr(type(self.ca_public_key)))
            with open(my_cert_file_path, "rb") as my_cert_file:
                self.my_cert = x509.load_pem_x509_certificate(my_cert_file.read())
            vermy, _ = self.verify_cert_signature(self.my_cert)
            if not vermy:
                raise KeyDistributionVerificationException("Invalid certificate self.my_cert=" + repr(self.my_cert))

    def verify_cert_signature(self, certificate):
        if isinstance(certificate, bytes):
            check_cert = x509.load_pem_x509_certificate(certificate)
        elif isinstance(certificate, str):
            with open(certificate, "rb") as check_cert_file:
                check_cert = x509.load_pem_x509_certificate(check_cert_file.read())
        elif not isinstance(certificate, x509.Certificate):
            raise KeyDistributionInputException("Invalid type of certificate=" + repr(type(certificate)))
        else:
            check_cert = certificate
        try:
            self.ca_public_key.verify(
                check_cert.signature,
                check_cert.tbs_certificate_bytes,
                padding.PKCS1v15(),
                check_cert.signature_hash_algorithm,
            )
        except InvalidSignature:
            return (False, None)
        return (True, check_cert.public_key())

    def get_my_cert(self, ret_type: str):
        if ret_type == "pem":
            ret = self.my_cert.public_bytes(serialization.Encoding.PEM)
        elif ret_type == "obj":
            ret = self.my_cert
        else:
            raise KeyDistributionInputException("Invalid value of ret_type=" + repr(ret_type))
        return ret
