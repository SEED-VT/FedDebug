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

from ibmfl.aggregator.fusion.iter_avg_fusion_handler import \
    IterAvgFusionHandler
from ibmfl.aggregator.fusion.fusion_handler import \
    FusionUtil
from ibmfl.exceptions import HyperparamsException

logger = logging.getLogger(__name__)


class ComparativeEliminationFusionHandler(IterAvgFusionHandler):
    """
    Class for Comparative Elimination (CE) Fusion Algorithm.
    This class implements the CE fusion algorithm presented
    here: https://arxiv.org/abs/2108.11769 
    The CE fusion algorithm is a Byzantine-robust fusion algorithm.
    At high level, the CE fusion algorithm sorts the parties
    according to their l2-distance from the current global model 
    (computed in the previous round), and selects the parties 
    with smallest n - f diststances, where n is the number of parties 
    and f is the estimated number of Byzantine parties.
    """

    def __init__(self, hyperparams,
                 protocol_handler,
                 data_handler=None,
                 fl_model=None,
                 **kwargs):
        """
        Initializes a ComparitiveEliminationFusionHandler object \
        with provided information, such as hyperparams, protocol_handler, \
        data_handler, and fl_model.

        :param hyperparams: Hyperparameters used for training
        :type hyperparams: `dict`
        :param protocol_handler: Protocol handler used for handling learning \
        algorithm's request for communication.
        :type protocol_handler: `ProtoHandler`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param fl_model: model to be trained
        :type fl_model: `model.FLModel`
        :param kwargs: Additional arguments to initialize a fusion handler
        :type kwargs: `Dict`
        :return: None
        """

        super().__init__(hyperparams,
                         protocol_handler,
                         data_handler,
                         fl_model,
                         **kwargs)
        self.name = "ComparativeElimination"
        # Initialize model weights; check that weights ae of type list
        # This is needed as aggregator computes distances from current model
        if hyperparams.get('initial_weights') is not None:
            if not self.current_model_weights:
                logger.info('Initializing the model using initial weights '
                            'provided in config file')
                self.current_model_weights = hyperparams.get('initial_weights')
            else:
                logger.info('Both initial model and initial weights detected. '
                            'Using the initial model.')
        if self.current_model_weights is None:
            raise HyperparamsException('Either initial model or initial weights '
                                        'must be provided for aggregator!')
        elif not isinstance(self.current_model_weights,list): 
            raise HyperparamsException('Model weights must be of type list')

        # Check constraints on Byzantine threshold parameter
        if hyperparams and hyperparams.get('global') is not None \
            and hyperparams['global'].get('byzantine_threshold') is not None:
                self.byzantine_threshold = hyperparams['global']['byzantine_threshold']
        else:
            raise HyperparamsException("CE Fusion: Byzantine threshold is not specified")

        if hyperparams and hyperparams.get('global') is not None \
            and hyperparams['global'].get('num_parties') is not None:
                num_parties = hyperparams['global']['num_parties']
        else:
            raise HyperparamsException("CE Fusion: Number of parties is not specified")

        target_quorum = math.ceil(self.perc_quorum * num_parties)
        if self.byzantine_threshold < 0 \
            or target_quorum <= 2 * self.byzantine_threshold:
            logging.error('Comparative Elimination Fusion: '
                          'Byzantine resilience assumes t > 2*f + 2.\n'
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
        `ModelUpdate`, and using the weights included in each model_update, it
        finds a robust aggregate model for the next round.

        :param lst_model_updates: List of model updates of type `ModelUpdate`
        :type lst_model_updates:  `lst`
        :param key: A key indicating what values the method will aggregate over
        :return: Result after aggregation
        :rtype: `list`
        """
        num_updates = len(lst_model_updates)
        if self.byzantine_threshold < 0 or \
            self.byzantine_threshold >= num_updates/2:
            logging.error('Comparitive Elimination Fusion: '  
                'Byzantine parties f must be non-negative and \
                 less than half the number of parties. '
                'Please pick the parameters appropriately\n'
                'n: {}\n'
                'f: {}\n'.format(len(lst_model_updates), self.byzantine_threshold))
            raise HyperparamsException
        else:
            # Compute distances between current model and party updates
            # Details: w_i^t: party ypdate, w^{t-1}: current mode
            # Compute: ||w_i^t - w^{t-1}||_2^2
            distances = self.get_distances_from_current_model(lst_model_updates, key)
            # Select n - f parties with smallest distances
            num_updates_selected = num_updates - self.byzantine_threshold
            sorted_indices = np.argsort(distances, axis=0)
            selected_indices = sorted_indices[0:num_updates_selected]
            selected_updates = []
            for index in selected_indices:
                try:
                    update = np.array(lst_model_updates[index].get(key),dtype=float)
                except Exception as ex:
                    update = self.transform_update_to_np_array(
                        lst_model_updates[index].get(key))
                
                selected_updates.append(update)    
            updated_weights = np.mean(np.array(selected_updates), axis=0)
            
        return updated_weights.tolist()

    def get_distances_from_current_model(self, lst_model_updates, key='weights'):
        """
        Computes distances between the current model and each model update

        :param lst_model_updates: List of model updates participating in fusion round
        :type lst_model_updates: `list`
        :param key: Key to pull from model update (default to 'weights')
        :return: distance
        :rtype: `np.array`
        """
        current_model = self.current_model_weights.copy()
        current_model_flattened = FusionUtil.flatten_model_update(
                                                                current_model)
        
        lst_model_updates_flattened = []
        for update in lst_model_updates:
            lst_model_updates_flattened.append(
                FusionUtil.flatten_model_update(update.get(key)))

        num_updates = len(lst_model_updates)
        distances = np.zeros(num_updates)
        for i in range(num_updates):
            distances[i] = np.square(np.linalg.norm(
                current_model_flattened - lst_model_updates_flattened[i])
                )
            
        return distances
