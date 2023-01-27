"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import abc
from ibmfl.crypto.crypto_exceptions import *

class CryptoCert:
    """
    This class defines an interface for certificate verification functions. 
    """

    @abc.abstractmethod
    def __init__(self, ca_cert_file_path: str, my_cert_file_path: str, **kwargs):
        return

    @abc.abstractmethod
    def verify_cert_signature(self, certificate):
        raise NotImplementedError

    def verify_certs(self, certificates):
        ret = {}
        for id, cert in certificates.items():
            ver, pbkey = self.verify_cert_signature(cert)
            if not ver:
                raise KeyDistributionVerificationException("Invalid certificate=" + repr(cert))
            ret[id] = pbkey
        return ret

    @abc.abstractmethod
    def get_my_cert(self, ret_type: str):
        raise NotImplementedError
