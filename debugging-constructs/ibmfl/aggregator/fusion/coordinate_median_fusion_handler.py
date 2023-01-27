"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Module to where fusion algorithms are implemented.
"""
import logging
import numpy as np

from ibmfl.aggregator.fusion.iter_avg_fusion_handler import \
    IterAvgFusionHandler

logger = logging.getLogger(__name__)


class CoordinateMedianFusionHandler(IterAvgFusionHandler):

    """
    Class for weight based Coordinate-Median aggregation.

    In this class, the averaging aggregation is performed using Coordinate-Median policy model weights.
    Implements the algorithm in Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates: https://arxiv.org/pdf/1803.01498.pdf
    """

    def __init__(self, hyperparams, protocol_handler,
                 fl_model=None, data_handler=None, **kwargs):
        super().__init__(hyperparams,
                         protocol_handler,
                         data_handler,
                         fl_model, **kwargs)

        self.name = "COORDMEDIAN"

    def fusion_collected_responses(self, lst_model_updates, key='weights'):
        """
        Receives a list of model updates, where a model update is of the type
        `ModelUpdate`, using the values (indicating by the key)
        included in each model_update, it finds the update by combining the
        ModelUpdates together at each layer and
        determining the median of each layer

        :param lst_model_updates: List of model updates of type `ModelUpdate` to be averaged.
        :type lst_model_updates:  `list`
        :param key: A key indicating what values the method will aggregate over.
        :type key: `str`
        :return: results after aggregation
        :type: `list`
        """
        parameters = []
        results = []

        for update in lst_model_updates:
            parameters.append(update.get(key))
        for layer in zip(*parameters):
            tensors = []
            for i, tensor in enumerate(layer):
                tensors.append(layer[i])

            temp = np.array(tensors)
            results.append(np.array(np.median(temp, axis=0)))

        return results
