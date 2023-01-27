"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
    Abstract base class for providing evidence for accountability.
"""

from abc import ABC, abstractmethod

class AbstractEvidenceRecorder(ABC):
    """
    Class that supports providing evidence of FL actions.
    Concrete implementations should act in a black-box fashion
    with only the methods below exposed to the caller
    """

    def __init__(self, info):
        """
        Initializes an `AbstractEvidenceRecorder` object with info.

        :param info: info required for this recorder.
        :type info: `dict`
        """
        self.info = info


    @abstractmethod
    def add_claim(self, predicate: str, custom_string: str):
        """
        Adds a new claim as evidence.
        Throws: An exception on failure
        :custom_string: a caller provided string, non-empty
    """

    """
    We may need to:
    1) enhance the above method parameters etc
    2) provide for a module "registration" mechanism
    3) consider logging-like usage
    """
