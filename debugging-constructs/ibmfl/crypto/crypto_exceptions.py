"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""

from ibmfl.exceptions import FLException


class CryptoException(FLException):
    pass


class KeyManagerException(FLException):
    pass

class KeyDistributionException(FLException):
    pass

class KeyDistributionInputException(KeyDistributionException):
    pass

class KeyDistributionVerificationException(KeyDistributionException):
    pass

class KeyDistributionCommunicationException(KeyDistributionException):
    pass
