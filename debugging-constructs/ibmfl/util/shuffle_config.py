"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Module providing utility functions helpful for initializing shuffling operations
"""

import logging.config
from ibmfl.exceptions import InvalidConfigurationException

logger = logging.getLogger(__name__)


def get_seed_filename(info_dict):
    """
    Retrieve shuffling seed filename from argument
    :param info_dict: dictionary of configuration
    :type info_dict: `dict`
    :return: seed filename
    :rtype: `str`
    """

    try:
        seed_file = info_dict.get('info', {}).get('permute_secret')
    except Exception as ex:
        logger.exception(ex)
        raise InvalidConfigurationException(
            'Error occurred while loading permute_secret info.')

    if not isinstance(seed_file, str):
        raise InvalidConfigurationException('No valid seed file provided.')
    return seed_file


def get_seed(seed_file):
    """
    Retrieve shuffling permute secret seed from file
    :param seed_file: filename of permute secret file
    :type seed_file: `str`
    :return: permute secret
    :rtype: `int`
    """

    try:
        with open(seed_file, 'r') as infile:
            permute_secret = int(infile.read(), 0)
    except ValueError as ve:
        logger.exception(ve)
        raise InvalidConfigurationException('Error invalid seed value.')
    except Exception as ex:
        logger.exception(ex)
        raise InvalidConfigurationException('Error seed file not found.')
    return permute_secret