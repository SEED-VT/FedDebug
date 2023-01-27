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
import math

from ibmfl.aggregator.fusion.fusion_handler import \
    FusionUtil
from ibmfl.aggregator.fusion.iter_avg_fusion_handler import \
    IterAvgFusionHandler
from ibmfl.exceptions import GlobalTrainingException, HyperparamsException

logger = logging.getLogger(__name__)


class KrumFusionHandler(IterAvgFusionHandler):
    """
    Class for Krum Fusion.

    Implements the Krum algorithm presented
    here: https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent
    """

    def __init__(self, hyperparams,
                 protocol_handler,
                 data_handler=None,
                 fl_model=None,
                 **kwargs):
        """
        Initializes an KrumAvgFusionHandler object with provided fl_model,
        data_handler and hyperparams.

        :param hyperparams: Hyperparameters used for training.
        :type hyperparams: `dict`
        :param protocol_handler: Protocol handler used for handling learning \
        algorithm's request for communication.
        :type protocol_handler: `ProtoHandler`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param fl_model: model to be trained
        :type fl_model: `model.FLModel`
        :param kwargs: Additional arguments to initialize a fusion handler.
        :type kwargs: `Dict`
        """

        super().__init__(hyperparams,
                         protocol_handler,
                         data_handler,
                         fl_model,
                         **kwargs)
        self.name = "Krum"
        self._eps = 1e-6

        if hyperparams and hyperparams.get('global') is not None \
            and hyperparams['global'].get('byzantine_threshold') is not None:
                self.byzantine_threshold = hyperparams['global']['byzantine_threshold']
        else:
            raise HyperparamsException("KRUM Fusion: Byzantine threshold is not specified")

        if hyperparams and hyperparams.get('global') is not None \
            and hyperparams['global'].get('num_parties') is not None:
                num_parties = hyperparams['global']['num_parties']
        else:
            raise HyperparamsException("KRUM Fusion: Number of parties is not specified")

        target_quorum = math.ceil(self.perc_quorum * num_parties)
        if target_quorum <= 2 * self.byzantine_threshold + 2:
            logging.error('KRUM Fusion: Byzantine resilience assumes t > 2*f + 2.\n'
                          'Note: Target Quorum (t) is computed using <num_parties> '
                          'and <perc_quorum> fields in the config file.\n'
                          'Please pick the parameters appropriately\n'
                          'Current parameters:\n'
                          'Target Quorum (t): {}\n'
                          'Byzantine threshold (f): {}\n'.format(target_quorum, self.byzantine_threshold))
            raise HyperparamsException

    def fusion_collected_responses(self, lst_model_updates, key='weights'):
        """
        Receives a list of model updates, where a model update is of the type
        `ModelUpdate`, using the weights included in each model_update, it
        finds the best model for the next round.

        :param lst_model_updates: List of model updates of type `ModelUpdate`
        :type lst_model_updates:  `lst`
        :param key: The key we wish to access from the model update
        :return: Result after fusion
        :rtype: `list`
        """
        weights = None

        num_updates = len(lst_model_updates)
        # Check the following condition again to accommodate dropouts and rejoins
        if num_updates <= 2 * self.byzantine_threshold + 2:
            logging.error('KRUM Fusion: Byzantine resilience assumes t > 2*f + 2.\n'
                          'Please pick the parameters appropriately\n'
                          'Current parameters:\n'
                          'Number of parties that replied (t): {}\n'
                          'Byzantine threshold (f): {}\n'.format(len(lst_model_updates), self.byzantine_threshold))
            raise GlobalTrainingException
        else:
            distance = self.get_distance(lst_model_updates, key)
            # score is computed using n-f-2 closest vectors
            th = num_updates - self.byzantine_threshold - 2
            scores = self.get_scores(distance, th)
            selected_idx = np.argmin(scores)

            weights = lst_model_updates[selected_idx].get(key)
            return weights

    def get_distance(self, lst_model_updates, key):
        """
        Generates a matrix of distances between each of the model updates
        to all of the other model updates 

        :param lst_model_updates: List of model updates participating in fusion round
        :type lst_model_updates: `list`
        :param key: Key to pull from model update (default to 'weights')
        :return: distance
        :rtype: `np.array`
        """
        num_updates = len(lst_model_updates)
        distance = np.zeros((num_updates, num_updates), dtype=float)

        lst_model_updates_flattened = []
        for update in lst_model_updates:
            lst_model_updates_flattened.append(
                FusionUtil.flatten_model_update(update.get(key)))

        for i in range(num_updates):
            curr_vector = lst_model_updates_flattened[i]
            for j in range(num_updates):
                if j is not i:
                    distance[i, j] = np.square(np.linalg.norm(
                        curr_vector - lst_model_updates_flattened[j]))
                    # Default is L-2 norm
        return distance

    @staticmethod
    def get_scores(distance, th):
        """
        Sorts the distances in an ordered list and returns the list for use to 
        the fusion_collected_responses function

        :param distance: List of distance vector
        :type distance: `list`
        :param th: Threshold
        :return: list of summation of distances
        :rtype: `list`
        """
        distance.sort(axis=1)
        # the +1 is added to account for the zero entry (distance from itself)
        return np.sum(distance[:, 0:th+1], axis=1)
