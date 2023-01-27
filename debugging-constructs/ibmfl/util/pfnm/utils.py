"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
from __future__ import print_function
import numpy as np


def change_global_dtypes(w_global, w_client):
    """
    Converts w_global to the same dtype as w_client. PFNM procedure changes the dtype
    causing discrepancies between data dtype and model dtype.
    :param w_global: global weights
    :param w_client: client_weights
    :return: w_global: global weights
    """
    result = []
    for i in range(len(w_client)):
        dtype = w_client[i].dtype
        result.append(w_global[i].astype(dtype))

    return result


def transpose_weights(weights, transpose_weights):
    """
    Owing to different parameter shapes returned by different frameworks, for ex,
    Keras vs. PyTorch, this function fixes a standard order to the weights matrix
    :param weights: client weights
    :type: `list`
    :param transpose_weights: indicator variables determining if client weights should
    be transposed or not
    :type transpose_weights: `list`
    :return: weights: weights with dimension order standardized
    :rtype: `list`
    """

    weights_standardized = []

    for w_client, wT_client in zip(weights, transpose_weights):
        if not wT_client:
            weights_standardized.append(w_client)
            continue

        w_client_standardized = []

        for w in w_client:
            if len(w.shape) == 2:
                w_client_standardized.append(w.T)
            else:
                w_client_standardized.append(w)

        weights_standardized.append(np.array(w_client_standardized))

    return weights_standardized


def prepare_class_freqs(cls_counts, n_classes):
    """
    Converts the dict formatted class counts into a list with the
    array index being the class label and the corresponding value
    being the number of examples of this class

    :param cls_counts:
    :type cls_counts: `list of dictionary`
    :param n_classes: Number of output classes for the task
    :type n_classes: `int`
    :return: lst_cls_counts: A list of individual network batch frequencies
    :rtype: `list` of `list`
    """

    if None in cls_counts:
        return None

    lst_cls_counts = []

    for party_cls_counts in cls_counts:
        temp = [0] * n_classes
        for label, count in party_cls_counts.items():
            temp[int(label)] = int(count)

        lst_cls_counts.append(np.array(temp))

    return lst_cls_counts


def compute_net_dimensions(weights):
    """
    Computes the network dimensions from the its weights
    :param weights: networks weights
    :type weights: `numpy.ndarray`
    :return: network dimensions
    :rtype: `list`
    """
    new_dims = []
    for i in range(1, len(weights), 2):
        new_dims.append(weights[i].shape[0])
    return new_dims
