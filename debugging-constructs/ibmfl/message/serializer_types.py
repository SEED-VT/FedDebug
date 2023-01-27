"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
from enum import Enum


class SerializerTypes(Enum):
    """
    Types of supported Serializers
    """
    PICKLE = 1
    JSON_PICKLE = 2
