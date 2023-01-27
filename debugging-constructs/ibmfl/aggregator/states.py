"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
 An enumeration class for the message type field which describe Aggregator status
"""
from enum import Enum


class States(Enum):
    """
    States for Aggregator
    """
    INITIALIZING = 0
    CLI_WAIT = 1
    START = 2
    TRAIN = 3
    EVAL = 4
    SYNC = 5
    SAVE = 6
    STOP = 7
    ERROR = 8
    SND_REQ = 9
    PROC_RSP = 10
    QUORUM_WAIT = 11
    REGISTER_WAIT = 12
