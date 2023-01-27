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

from ibmfl.model.model_update import ModelUpdate
from ibmfl.aggregator.fusion.fusion_handler import FusionHandler
from ibmfl.exceptions import GlobalTrainingException

from ibmfl.util.pfnm import utils as pfnm_utils
from ibmfl.util.pfnm import core as pfnm_core
from ibmfl.util.pfnm import matching as pfnm_matching


logger = logging.getLogger(__name__)


class PFNMFusionHandler(FusionHandler):
    """
    Class for weight based PFNM aggregation.
    The method is described here: https://arxiv.org/abs/1905.12022

    This method supports only Fully-Connected Networks.
    Batch Normalization layer is not supported.
    """

    def __init__(self, hyperparams, protocol_handler,
                 fl_model, data_handler=None, **kwargs):
        """
        Initializes an PFNMFusionHandler object with provided fl_model,
        data_handler and hyperparams.

        :param hyperparams: Hyperparameters used for PFNM training. \
                The four hyperparameters used are: \
                sigma: `float` (default 1.0) Determines how far the local model \
                neurons are allowed from the global model. A bigger value \
                results in more matching and hence a smaller global model.\
                sigma0: `float` (default 1.0) Defines the standard-deviation \
                of the global network neurons. Acts as a regularizer. \
                gamma: `float` (default 1.0) Indian Buffet Process parameter \
                controlling the expected number of features present in each \
                observation. \
                iters: `int` (default 3) How many iterations of randomly \
                initialized matching-unmatching procedure is to be performed \
                before finalizing the solution
        :type hyperparams: `dict`
        :param protocol_handler: Protocol handler used for handling learning \
        algorithm's request for communication.
        :type protocol_handler: `ProtoHandler`
        :param fl_model: model to be trained
        :type fl_model: `model.FLModel`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param kwargs: Additional arguments to initialize a fusion handler.
        :type kwargs: `Dict`
        """
        super().__init__(hyperparams,
                         protocol_handler,
                         data_handler,
                         fl_model,
                         **kwargs)
        self.name = "PFNM"
        self.rounds = hyperparams['global']['rounds']
        self.termination_accuracy = hyperparams['global'].get('termination_accuracy')
        self.sigma = hyperparams['global'].get('sigma', 1.0)
        self.sigma0 = hyperparams['global'].get('sigma0', 1.0)
        self.gamma = hyperparams['global'].get('gamma', 1.0)
        self.iters = hyperparams['global'].get('iters', 3)

        self._local_weights = dict()
        self._assignment = None
        self._party_list = None
        self.curr_round = 0
        self.global_accuracy = -1

        if fl_model is None:
            self.model_update = None
        else:
            self.model_update = fl_model.get_model_update()

    def start_global_training(self):
        """
        Starts global federated learning training process.
        """
        self.curr_round = 0
        while not self.reach_termination_criteria(self.curr_round):

            lst_parties = self.get_available_parties()
            lst_payload = []

            # Create custom payload for each local model
            # and get their model updates
            for party_id in lst_parties:
                payload = dict()
                payload['hyperparams'] = self.hyperparams
                payload['model_update'] = ModelUpdate(weights=self._local_weights[party_id]) \
                        if party_id in self._local_weights else self.model_update
                lst_payload.append(payload)

            lst_replies, lst_parties = self.query_parties(lst_payload, lst_parties, return_party_list=True)

            # If a different set of parties respond as compared to the last round, 
            # reset assignment variable
            if self.reset_assignment_variable(responding_parties=lst_parties):
                self._assignment = None

            # Order `lst_replies` according to last
            lst_replies = self.order_party_replies(lst_replies, lst_parties)

            # Collect all model updates for fusion:
            self.fusion_collected_responses(lst_replies, lst_parties, self._assignment)

            # Update model if we are maintaining one
            if self.fl_model is not None:
                new_dims = pfnm_utils.compute_net_dimensions(
                    self.model_update.get('weights'))
                self.fl_model.expand_model_by_layer_name(
                    new_dimension=new_dims, layer_name="dense")
                self.fl_model.update_model(self.model_update)

            self.curr_round += 1
            self.save_current_state()

    def order_party_replies(self, lst_replies, lst_parties):
        """
        Since the next round of PFNM depends on the order of response due to the 
        usage of assignment variable from the previous round, it is important to 
        order the party model updates in the same order as before.
        This function accepts the model updates and the list of parties in the current
        round, and returns an ordered list of model updates that's consistent across
        communication rounds

        :param lst_replies: List of model updates (replies) from parties
        :type lst_replies: `list`
        :param lst_parties: List of party IDs corresponding to the model updates
        :type lst_parties: `list`
        :return: `lst_replies` ordered consistently across communication rounds
        :rtype: `list`
        """

        if self.reset_assignment_variable(lst_parties):
            return lst_replies
        
        lst_replies_ordered = []
        for party_id in self._party_list:
            
            if party_id not in lst_parties:
                continue

            idx = lst_parties.index(party_id)
            lst_replies_ordered.append(lst_replies[idx])
        
        return lst_replies

    def reach_termination_criteria(self, curr_round):
        """
        Returns True when termination criteria has been reached, otherwise
        returns False.
        Termination criteria is reached when the number of rounds run reaches
        the one provided as global rounds hyperparameter.
        If a `DataHandler` has been provided and a targeted accuracy has been
        given in the list of hyperparameters, early termination is verified.

        :param curr_round: Number of global rounds that already run
        :type curr_round: `int`
        :return: boolean
        :rtype: `boolean`
        """
       
        if curr_round >= self.rounds:
            logger.info('Reached maximum global rounds. Finish training :) ')
            return True

        return self.terminate_with_metrics(curr_round)


    def get_global_model(self):
        """
        Returns last model_update

        :return: model_update
        :rtype: `ModelUpdate`
        """
        return self.model_update
    
    def reset_assignment_variable(self, responding_parties):
        """
        Assignment variable stores the mapping from local neurons to global neurons in PFNM.
        While using perc_quorum, if fewer parties reply with updates, the assignment variable
        becomes stale, and cannot be used. 
        This function returns true if fewer than registered parties have replied with updated.

        :param: Number of parties responding for fusion
        :return: boolean
        :rtype: `boolean`
        """

        if self._party_list is None:
            return True
        
        last_round_parties = set(self._party_list)
        current_round_parties = set(responding_parties)

        if len(last_round_parties.difference(current_round_parties)) == 0:
            # there are no discrepancies between current round and last round parties. 
            # do not reset assignment
            return False
        
        return True

    def fusion_collected_responses(self, lst_model_updates, lst_parties, assignment_old=None):
        """
        Receives a list of model updates, where a model update is of the type
        `ModelUpdate`, using the weighs included in each model_update, it
        finds the mean.
        The averaged weights is stored in self.model_update
        as type `ModelUpdate`.

        :param lst_model_updates: List of model updates of type `ModelUpdate` \
        to be averaged.
        :type lst_model_updates: `list`
        :param assignment_old: Array of previous PFNM neuron assignments
        :type assignment_old: `np.ndarray`
        :return: None
        """

        weights = []
        cls_counts = []
        _transpose_weights = []
        n_models = len(lst_model_updates)
        n_classes = None

        for update in lst_model_updates:
            weights.append(np.array(update.get('weights')))
            cls_counts.append(update.get('class_counts'))
            _transpose_weights.append(update.get('transpose_weight'))

            n_classes = weights[-1][-1].shape[-1] if n_classes is None else n_classes

        if len(weights) == 0:
            raise GlobalTrainingException('No weights available for PFNM')

        cls_counts = pfnm_utils.prepare_class_freqs(cls_counts, n_classes)
        weights = pfnm_utils.transpose_weights(weights, _transpose_weights)

        global_weights, assignment_old = pfnm_matching.match_network(batch_weights=weights,
                                                                     batch_frequencies=cls_counts,
                                                                     sigma_layers=self.sigma, sigma0_layers=self.sigma0,
                                                                     gamma_layers=self.gamma, iters=self.iters,
                                                                     assignments_old=assignment_old)

        # make global weights dtype same as local weights
        global_weights = pfnm_utils.change_global_dtypes(
            global_weights, weights[0])

        localnet_weights = [pfnm_core.build_init(
            global_weights, assignment_old, j) for j in range(n_models)]

        localnet_weights = pfnm_utils.transpose_weights(
            localnet_weights, _transpose_weights)

        # If weights were transposed then global weights need to be transposed as well
        _transpose_w_global = True in _transpose_weights
        global_weights = pfnm_utils.transpose_weights(
            [global_weights], [_transpose_w_global])
        global_weights = global_weights[0]

        self.model_update = ModelUpdate(
            weights=np.array(global_weights).tolist())

        self._local_weights = {lst_parties[i]: localnet_weights[i] for i in range(len(lst_parties))}
        self._party_list = lst_parties
        self._assignment = assignment_old

    def get_current_metrics(self):
        """Returns metrics pertaining to current state of fusion handler

        :return: metrics
        :rtype: `dict`
        """
        fh_metrics = {}
        fh_metrics['rounds'] = self.rounds
        fh_metrics['curr_round'] = self.curr_round
        fh_metrics['acc'] = self.global_accuracy

        #fh_metrics['model_update'] = self.model_update
        return fh_metrics
