"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
 An enumeration class for the message type field which describe what
 kind of data is being sent inside the Message
"""
from enum import Enum

__author__ = "Supriyo Chakraborty, Shalisha Witherspoon, Dean Steuer"


class MessageType(Enum):
    """
    Message types for communication between party and aggregator
    """
    MODEL_UPDATE = 1
    MODEL_HYPERPARAMETERS = 2
    MODEL_PARAMETERS = 3
    REQUEST_MODEL_HYPERPARAMETERS = 4
    REQUEST_MODEL_UPDATE = 5
    REGISTER = 6
    TRAIN = 7
    SAVE_MODEL = 8
    PREDICT_MODEL = 9
    EVAL_MODEL = 10
    ACK = 11
    SYNC_MODEL = 12
    STOP = 14
    PREPROCESS = 15
    REQUEST_CERT = 16
    GENERATE_KEYS = 17
    DISTRIBUTE_KEYS = 18
    SET_KEYS = 19
    ERROR_AUTH = 400
