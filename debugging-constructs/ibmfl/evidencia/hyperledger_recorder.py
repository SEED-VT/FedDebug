"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
    Concrete class for providing evidence for accountability.
"""


import logging
from ibmfl.evidencia.evidence_recorder import AbstractEvidenceRecorder
from evidentia.etb_connection import EtbConnection

#Set up logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)-6s %(name)s :: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

LOGGER = logging.getLogger(__package__)


class HyperledgerEvidenceRecorder(AbstractEvidenceRecorder):
    """
    Concrete implementation for mock purposes
    """

    def __init__(self, info):
        """
        Initializes an `AbstractEvidenceRecorder` object with info.

        :param info: info required for this recorder.
        :type info: `dict`
        """
        self.info = info
        self.etb = EtbConnection(info)


    def add_claim(self, predicate: str, custom_string: str):
        """
        Adds a new claim as evidence.
        Throws: An exception on failure
        :custom_string: a caller provided string, non-empty
        """

        if not custom_string:
            raise ValueError('Claim must be substantiated with non-empty value.')
        LOGGER.info(predicate)
        LOGGER.info(custom_string)
        self.etb.addClaim(predicate, custom_string)
