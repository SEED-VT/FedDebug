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


class FLMetricsManager(object):

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
                "Metric Handler not callable. Handler should be a function")

    def save_metrics(self, metrics):
        """Invoke all the handlers registers for save event and pass metrics

        :param metrics: Metrics dictionary
        :type metrics: `dict`
        """
        for handle in self._handlers:
            try:
                handle(metrics)

            except Exception as ex:
                logger.error(
                    "Error occured while executing handler in metric manager")
                logger.exception(ex)
                logger.error('Could not execute handler ' + handle)


class FileCheckpointHandler(object):
    """Simple filehandler which takes the metrics and saves it in a file.
    """

    def __init__(self, **kwargs):
        """Initialize checkpoint handler
        """
        logger.info("FileCheckPointHandler initialized")

    def handle(self, metrics):
        """Creates a json file containing metrics with timestamp.

        :param metrics: Metrics dictionary
        :type metrics: `dict`
        """
        filename = 'checkpoint_{}.json'.format(time.time())
        with open(filename, 'w') as fp:
            json.dump(metrics, fp)
