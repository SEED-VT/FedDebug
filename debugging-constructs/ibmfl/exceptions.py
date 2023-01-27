"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
This module will host all the exceptions which are raised by fl components
"""


class FLException(Exception):
    pass


class DuplicateRouteException(FLException):
    pass


class InvalidConfigurationException(FLException):
    pass


class InvalidServerConfigurationException(InvalidConfigurationException):
    pass


class NotFoundException(FLException):
    pass


class LocalTrainingException(FLException):
    pass


class TCPMessageOutOfOrder(FLException):
    pass


class GlobalTrainingException(FLException):
    pass


class ModelException(FLException):
    pass


class ModelInitializationException(FLException):
    pass


class ModelUpdateException(FLException):
    pass


class HyperparamsException(FLException):
    pass


class WarmStartException(FLException):
    pass


class FusionException(FLException):
    pass


class QuorumException(FLException):
    pass


class PreprocessException(FLException):
    pass
