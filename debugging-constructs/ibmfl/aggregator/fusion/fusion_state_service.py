"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""

import json
import time
import logging

from ibmfl.exceptions import InvalidConfigurationException

logger = logging.getLogger(__name__)

"""
 An enumeration class for the message type field which describe fusion status
"""
from enum import Enum


class States(Enum):
    """
    States for Fusion handler
    """
    IDLE = 0
    RCV_MODEL = 1 
    SND_MODEL = 2
    AGGREGATING = 3
    SAVE_STATE = 4



class FLFusionStateManager(object):

    def __init__(self):
        """Keeps track of registered handlers, should be extented to handle event
        based handlers.
        """
        #self.events = []
        self._handlers = set()

    def register(self, handler):
        """Registers handlers/observers into the manager/scheduler

        :param handler: A routine which is invoked when scheduled
        :type handler: `method`
        """
        if callable(handler):
            self._handlers.add(handler)
        else:
            raise InvalidConfigurationException(
                "Fusion StatFusion State Handler not callable. Handler should be a function")

    def save_state(self, state ):
        """Invoke all the handlers registers for save event

        :param state: Fusion state
        :type state: States
        """
        logger.info("Fusion state " + str(state))
        for handle in self._handlers:
            try:
                handle(state)

            except Exception as ex:
                logger.error(
                    "Error occured while executing handler in fusion state manager" + ex)
                logger.error('Could not execute handler ' + handle)

