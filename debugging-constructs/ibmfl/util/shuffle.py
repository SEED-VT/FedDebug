"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import random
import numpy as np
from ibmfl.exceptions import ModelException
import logging

logger = logging.getLogger(__name__)

# Get deterministic permutation based on the seed
# To avoid race condition, do not set the global seed
def getperm(array, seed):
    """
    Compute permutation of model weights with the provided seed
    :param array: model weights
    :type  array: `numpy.ndarray`
    :param seed: key for permuting model weights
    :type seed: `int`
    :return: list of permuted model weight indexes
    :rtype: `list`
    """

    perm = list(range(len(array)))
    random.Random(seed).shuffle(perm)
    return perm


def shuffle(origarray, seed):
    """
    Shuffle model weights using the seed
    :param origarray: model weights
    :type  origarray: `numpy.ndarray`
    :param seed: key for shuffling model weights
    :type seed: `int`
    :return: shuffled model weights
    :rtype: `numpy.ndarray`
    """

    perm = getperm(origarray, seed)
    shuffledarray = [None] * len(origarray)
    shuffledarray[:] = [origarray[j] for j in perm]
    return np.array(shuffledarray)


def unshuffle(shuffledarray, seed):
    """
    Unshuffle model weights using the seed
    :param shuffledarray: shuffled model weights
    :type  shuffledarray: `numpy.ndarray`
    :param seed: key for unshuffling model weights
    :type seed: `int`
    :return: model weights
    :rtype: `numpy.ndarray`
    """

    perm = getperm(shuffledarray, seed)
    origarray = [None] * len(shuffledarray)
    for i, j in enumerate(perm):
        origarray[j] = shuffledarray[i]
    return np.array(origarray)


def checkallinstance(array, elementtype):
    """
    Test whether all elements of the array are of the specified elementtype
    :param array: model weights
    :type  array: `numpy.ndarray`
    :param elementtype: name of type
    :type elementtype: `type`
    :return: result of test
    :rtype: `boolean`
    """

    for i in array:
        if not isinstance(i, elementtype):
            return False
    return True


def shuffleweight(weight, seed):
    """
    Shuffle model weights using the seed
    :param weight: model weights
    :type weight: `numpy.ndarray`
    :param seed: key for shuffling model weights
    :type seed: `int`
    :return: shuffled model weights
    :rtype: `numpy.ndarray`
    """

    # if all elements in the array are floating number weights, shuffle the array
    # if the type is nested array, recursively shuffle all subarrays and concatenate
    if checkallinstance(weight, np.float32) or checkallinstance(weight, float):
        return shuffle(weight, seed)
    elif checkallinstance(weight, np.ndarray) or checkallinstance(weight, list):
        shuffledarray = []
        for subweight in weight:
            shuffledarray.append(shuffleweight(subweight, seed))
        return np.array(shuffledarray)
    else:
        logger.error("inconsistent weight types in shuffle")
        raise ModelException("Different types of weights in one array")


def unshuffleweight(weight, seed):
    """
    Unshuffle model weights using the seed
    :param weight: shuffled model weights
    :type weight: `numpy.ndarray`
    :param seed: key for unshuffling model weights
    :type seed: `int`
    :return: model weights
    :rtype: `numpy.ndarray`
    """

    # if all elements in the array are floating number weights, unshuffle the array
    # if the type is nested array, recursively unshuffle all subarrays and concatenate
    if checkallinstance(weight, np.float32) or checkallinstance(weight, float): 
        return unshuffle(weight, seed)
    elif checkallinstance(weight, np.ndarray) or checkallinstance(weight, list):
        unshuffledarray = []
        for subweight in weight:
            unshuffledarray.append(unshuffleweight(subweight, seed))
        return np.array(unshuffledarray)
    else:
        logger.error("inconsistent weight types in unshuffle")
        raise ModelException("Different types of weights in one array")