"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
 An enumeration class for the crypto type field which describe what
 kind of data is being sent inside the Message
"""
from enum import Enum


class CryptoEnum(Enum):
    """
    Crypto types used for secure aggregation
    """
    CRYPTO_PAILLIER = 'Paillier'
    CRYPTO_THRESHOLD_PAILLIER = 'ThresholdPaillier'
    CRYPTO_MIFE = 'MIFE'
    CRYPTO_MCFE = 'MCFE'
    CRYPTO_DECENTRALIZED_MIFE = 'DMIFE'
    CRYPTO_FHE = 'FHE'

    KEY_PUBLIC_PARAMETER = 'pp'
    KEY_PRIVATE = 'sk'
    KEY_DECTYPT = 'dk'
    WEIGHTS_PLAINTEXT = 'plaintext-weights'
    WEIGHTS_CIPHERTEXT = 'ciphertext-weights'
