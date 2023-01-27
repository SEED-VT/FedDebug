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

from ibmfl.aggregator.fusion.fedavg_fusion_handler import FedAvgFusionHandler

logger = logging.getLogger(__name__)


class RLWeightedAvgFusionHandler(FedAvgFusionHandler):
    """
    Class for weight based Federated Averaging aggregation.

    In this class, the weighted averaging aggregation is performed over the RL
    policy model weights with averaging weights depends on rewards.
    """

    def __init__(self, hyperparams, protocol_handler,
                 fl_model=None,
                 data_handler=None,
                 **kwargs):
        super().__init__(hyperparams,
                         protocol_handler,
                         data_handler,
                         fl_model,
                         **kwargs)

        self.name = "RLWeightedAvg"

    def fusion_collected_responses(self, lst_model_updates):
        """
        Receives a list of model updates, where a model update is of the type
        `ModelUpdate`, using the weights and rewards included in each
        model_update, it finds the weighted average of the model weights
        per layer with averaging weights depends on rewards.

        :param lst_model_updates: List of model updates of type `ModelUpdate` \
        to be averaged.
        :type lst_model_updates: `list`
        :return: results after aggregation
        :rtype: `dict`
        """

        weights = dict()
        # Key list gives layers of the neural network
        weights_key_list = list(lst_model_updates[0].get('weights').keys())

        # Iterate through the layers of neutral network
        for key in weights_key_list:
            w = []
            n_k = []
            for update in lst_model_updates:
                w.append(np.array(update.get('weights').get(key)))
                n_k.append(update.get('train_result').get(
                    'episode_reward_mean'))
            n_norm = n_k / (np.sum(n_k) + self._eps)
            avg_weight = np.sum(
                [w[i] * n_norm[i] for i in range(len(n_k))], axis=0)
            weights[key] = avg_weight

        return weights
