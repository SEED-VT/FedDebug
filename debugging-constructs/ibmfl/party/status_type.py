"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
 An enumeration class for the message type field which describe party status
"""
from enum import Enum


class StatusType(Enum):
    """
    Status types for Party
    """
    IDLE = 1
    TRAINING = 2
    EVALUATING = 3
    STOPPING = 4
