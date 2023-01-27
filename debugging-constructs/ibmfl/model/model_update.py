"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import abc
import numpy as np
import logging
from pickle import loads, dumps
from ibmfl.exceptions import ModelUpdateException

logger = logging.getLogger(__name__)


class ModelUpdate():
    """
    Class to hold model update dictionary. Dictionary can only be accessed
    using inbuild methods like `get` and `add`
    """

    def __init__(self, **kwargs):
        """
        Initialize the dictionary and add updates.
        :param kwargs: Dictionary of model-update specific arguments.
        :type kwargs: `dict`
        """
        self.__updates = {}
        for key, value in kwargs.items():
            self.add(key, value)

    def add(self, key, value):
        """
        Add update `value` for `key`

        :param key: Identifier which represents the update
        :type key: `str`
        :param value: Content of the update
        :type value: any ds/object which can be pickled
        """
        try:
            self.__updates[key] = dumps(value)
        except Exception as ex:
            logger.exception(ex)

            logger.exception("Error occured while adding a update.\
                                Make sure model update data structure is picklable")

            raise ModelUpdateException("Error updating model update")

    def get(self, key):
        """
        Get update value for given key from model update dictionary

        :param key: Identifier which represents the update
        :type key: `str`
        :return: content of the update
        :rtype: any ds/object after its unpickled
        """
        if key not in self.__updates:
            logger.error("Key "+key+" not found in model updates")
            raise ModelUpdateException(
                "Invalid key was requested from model update")

        ret_val = loads(self.__updates[key])
        return ret_val

    def exist_key(self, key):
        """
        Check if a specified key is used in model update dictionary

        :param key: Identifier which represents the update
        :type key: `str`
        :return: check result of existing the key
        :rtype: True or False
        """
        return True if key in self.__updates else False
