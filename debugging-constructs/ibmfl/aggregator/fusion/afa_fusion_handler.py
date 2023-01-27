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
from scipy.stats import beta
import copy

from ibmfl.aggregator.fusion.iter_avg_fusion_handler import \
    IterAvgFusionHandler
from ibmfl.exceptions import GlobalTrainingException, HyperparamsException, ModelUpdateException, FusionException
from ibmfl.model.model_update import ModelUpdate
from ibmfl.evidencia.util.hashing import hash_model_update
from ibmfl.aggregator.fusion.fusion_handler import FusionUtil

logger = logging.getLogger(__name__)

class PartyAttributes:
    def __init__(self, id, updates, alpha, beta):
        self.id = id
        self.updates = updates
        self.alpha = alpha
        self.beta = beta
        self.similarity = None

    def get_pr(self):
        return self.alpha / (self.alpha + self.beta)


class AFAFusionHandler(IterAvgFusionHandler):
    """
    Class for Adaptive Federated Averaging Fusion.

    Implements the Adaptive Federated Averaging algorithm presented
    here: https://arxiv.org/abs/1909.05125. This aggregation scheme
    provides robust aggregation against malicious clients
    participating in the training process and blocks their contribution
    in the training once they have been identified as malicious.
    """

    def __init__(self, hyperparams,
                 protocol_handler,
                 data_handler=None,
                 fl_model=None,
                 **kwargs):
        """
        Initializes an AFAFusionHandler object with provided fl_model,
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
        self.name = "AFA"
        self._eps = 1e-6
        self._parties = {}
        self._blocked_parties = []

        if hyperparams and hyperparams.get('global') is not None \
            and hyperparams['global'].get('slack0') is not None:
                self.slack0 = hyperparams['global']['slack0']
                if self.slack0 <= 0.0:
                    logging.error('AFA Fusion: Robust aggregation assumes positive value for slack0.\n'
                                  'Note: slack0 controls the tradeoff between false positive rate and false negative '
                                  'rate.\n'
                                  'Small slack0 may increase false positives and large slack0 may increase '
                                  'false negatives.\n'
                                  'Please pick the parameters appropriately.\n'
                                  'Current parameters:\n'
                                  'slack0: {}\n'.format(self.slack0))
                    raise HyperparamsException
        else:
            self.slack0 = 2.0

        if hyperparams and hyperparams.get('global') is not None \
            and hyperparams['global'].get('slack_delta') is not None:
                self.slack_delta = hyperparams['global']['slack_delta']
                if self.slack_delta <= 0.0:
                    logging.error('AFA Fusion: Robust aggregation assumes positive value for slack_delta.\n'
                                  'Note: slack_delta controls the tradeoff between false positive rate and false '
                                  'negative rate.\n'
                                  'Small slack_delta may increase false positives and large slack_delta may increase '
                                  'false negatives.\n'
                                  'Please pick the parameters appropriately.\n'
                                  'Current parameters:\n'
                                  'slack_delta: {}\n'.format(self.slack_delta))
                    raise HyperparamsException
        else:
            self.slack_delta = 0.5

        if hyperparams and hyperparams.get('global') is not None \
            and hyperparams['global'].get('alpha0') is not None:
                self.alpha0 = hyperparams['global']['alpha0']
                if self.alpha0 <= 0.0:
                    logging.error('AFA Fusion: alpha0 is the shape parameter "alpha" of Beta distribution '
                                  'and must be positive.\n'
                                  'Note: The value of alpha0 determines the variance of the Beta distribution during '
                                  'the training process and for big values of alpha0, AFA will require more iterations '
                                  'to block the bad clients.\n'
                                  'Please pick the parameters appropriately.\n'
                                  'Current parameters:\n'
                                  'alpha0: {}\n'.format(self.alpha0))
                    raise HyperparamsException
        else:
            self.alpha0 = 3.0

        if hyperparams and hyperparams.get('global') is not None \
            and hyperparams['global'].get('beta0') is not None:
                self.beta0 = hyperparams['global']['beta0']
                if self.beta0 <= 0.0:
                    logging.error('AFA Fusion: beta0 is the shape parameter "beta" of Beta distribution '
                                  'and must be positive.\n'
                                  'Note: The value of beta0 determines the variance of the Beta distribution during '
                                  'the training process and for big values of beta0, AFA will require more iterations '
                                  'to block the bad clients.\n'
                                  'Please pick the parameters appropriately.\n'
                                  'Current parameters:\n'
                                  'beta0: {}\n'.format(self.beta0))
                    raise HyperparamsException
        else:
            self.beta0 = 3.0

        if hyperparams and hyperparams.get('global') is not None \
            and hyperparams['global'].get('block_threshold') is not None:
                self.block_threshold = hyperparams['global']['block_threshold']
                if self.block_threshold <= 0.0 or self.block_threshold >= 1.0:
                    logging.error('AFA Fusion: block_threshold must be in the interval (0.0, 1.0).\n'
                                  'Note: The value of block_threshold controls the tradeoff between false positive '
                                  'rate and false negative rate. Small block_threshold may increase false positives '
                                  'and large block_threshold may increase false negatives.\n'
                                  'Please pick the parameters appropriately.\n'
                                  'Current parameters:\n'
                                  'block_threshold: {}\n'.format(self.block_threshold))
                    raise HyperparamsException
        else:
            self.block_threshold = 0.95


    def start_global_training(self):
        """
        Starts an iterative global federated learning training process.
        """
        self.curr_round = 0
        while not self.reach_termination_criteria(self.curr_round):
            # construct ModelUpdate
            if self.current_model_weights:
                model_update = ModelUpdate(weights=self.current_model_weights)
            else:
                model_update = None


            if model_update:
                # log to Evidentia
                if self.evidencia:
                    self.evidencia.add_claim("sent_global_model",
                                            "{}, '\"{}\"'".format(self.curr_round + 1,
                                            hash_model_update(model_update)))

            payload = {'hyperparams': {'local': self.params_local},
                       'model_update': model_update
                       }
            logger.info('Model update' + str(model_update))

            # query all available parties
            lst_parties = self.get_available_parties()
            lst_replies, lst_parties = self.query_parties(payload, lst_parties, return_party_list=True)

            # log to Evidentia
            if self.evidencia:
                updates_hashes = []
                for update in lst_replies:
                    updates_hashes.append(hash_model_update(update))
                    self.evidencia.add_claim("received_model_update_hashes",
                                            "{}, '{}'".format(self.curr_round + 1,
                                            str(updates_hashes).replace('\'', '"')))

            self.update_weights(lst_replies, lst_parties)

            # Update model if we are maintaining one
            if self.fl_model is not None:
                self.fl_model.update_model(
                    ModelUpdate(weights=self.current_model_weights))

            self.curr_round += 1
            self.save_current_state()

    def update_weights(self, lst_model_updates, lst_parties):
        """
        Update the global model's weights with the list of collected
        model_updates from parties.
        In this method, it calls the self.fusion_collected_response to average
        the local model weights collected from parties and update the current
        global model weights by the results from self.fusion_collected_response.

        :param lst_model_updates: list of model updates of type `ModelUpdate` to be averaged.
        :type lst_model_updates: `list`
        :param lst_parties: List of parties to receive the query.
        :type lst_parties: `list`
        :return: None
        """
        self.current_model_weights = self.fusion_collected_responses(
            lst_model_updates, lst_parties)

    def get_weighted_average(self, list_partyids, dict_parties):
        """
        Performs weighted average of the provided model updates from different parties
        according to the probability of sending good model updates and number of training samples of
        each party.

        :param list_partyids: list of sudo IDs of the parties.
        :type list_partyids: `list`
        :param dict_parties: Dictionary containing parties' sudo IDs as keys and instances of class PartyAttributes as
                             values, containing all relevant information of a party.
        :type dict_parties: `dict`
        :return: model weights after performing weighted average
        :rtype: `list`
        """

        wt = []
        lst_wt = []
        lst_scale = []
        try:
            for pty in list_partyids:
                lst_wt.append(dict_parties[pty].updates.get("weights"))
                nk = dict_parties[pty].updates.get('train_counts')
                pk = dict_parties[pty].get_pr()
                lst_scale.append(pk * nk)
            norm_lst_scale = np.array(lst_scale)/ (np.sum(lst_scale) + self._eps)
            for layer in zip(*lst_wt):
                layer_weights = []
                assert len(layer) == len(norm_lst_scale), "mismatch of number of parties in layer and norm_lst_" \
                                                          "scale"
                for i, layer_party in enumerate(layer):
                    layer_weights.append(np.array(layer_party) * norm_lst_scale[i])
                avg_layer_weights = np.sum(layer_weights, axis=0)
                wt.append(avg_layer_weights)
        except ModelUpdateException as ex:
            logger.exception(ex)
            raise FusionException("Model updates are not appropriate for this fusion method.  Check local training.")

        return wt

    def filter_bad_parties(self, list_partyids, dict_parties):
        """
        Performs iterative filtering on model updates provided from different parties
        according to the model updates provided, probability of sending good model updates
        and number of training samples of each party.

        :param list_partyids: list of sudo IDs of the parties.
        :type list_partyids: `list`
        :param dict_parties: Dictionary containing parties' sudo IDs as keys and instances of class PartyAttributes as
                             values, containing all relevant information of a party.
        :type dict_parties: `dict`
        :return: Tuple containing lists of the sudo IDs of good parties and bad parties
        :rtype: `tuple`
        """

        good_parties = copy.deepcopy(list_partyids)
        bad_parties = []
        flag_filtering = True
        slack = self.slack0

        while flag_filtering:
            temp_bad_parties = []
            flag_filtering = False
            # compute weighted average of good parties weights
            new_weights = self.get_weighted_average(good_parties, dict_parties)
            # compute cosine similarity for all good parties
            flattened_new_weights = FusionUtil.flatten_model_update(new_weights)
            lst_party_sim = []
            for party in good_parties:
                model_weights = dict_parties[party].updates.get("weights")
                flattened_model_update = FusionUtil.flatten_model_update(model_weights)
                assert len(flattened_model_update.shape) == 1, "flattened_model_update must be a one dimensional " \
                                                               "vector for measuring similarity"
                assert flattened_new_weights.shape == flattened_model_update.shape, "shape mismatch between " \
                                                                "flattened_new_weights and flattened_model_update"
                dict_parties[party].similarity = np.sum(flattened_new_weights * flattened_model_update) / \
                                        (np.linalg.norm(flattened_new_weights) * np.linalg.norm(flattened_model_update))
                lst_party_sim.append(dict_parties[party].similarity)

            mean_sim = np.mean(lst_party_sim)
            median_sim = np.median(lst_party_sim)
            std_sim = np.std(lst_party_sim)
            if mean_sim < median_sim:
                for party in good_parties:
                    if dict_parties[party].similarity < median_sim - slack * std_sim:
                        temp_bad_parties.append(party)
                        bad_parties.append(party)
                        flag_filtering = True
            else:
                for party in good_parties:
                    if dict_parties[party].similarity > median_sim + slack * std_sim:
                        temp_bad_parties.append(party)
                        bad_parties.append(party)
                        flag_filtering = True

            if len(temp_bad_parties) > 0:
                for party in temp_bad_parties:
                    good_parties.remove(party)
            # increase slack to reduce false positives
            slack += self.slack_delta
        return good_parties, bad_parties

    def fusion_collected_responses(self, lst_model_updates, lst_parties,  key='weights'):
        """
        Receives a list of model updates, where a model update is of the type
        `ModelUpdate`, using the weights included in each model_update, it
        iteratively filters out bad model updates sent from clients and
        apply aggregation to the remaining model updates.

        :param lst_model_updates: List of model updates of type `ModelUpdate`
        :type lst_model_updates:  `list`
        :param lst_parties: List of parties to receive the query.
        :type lst_parties: `list`
        :param key: The key we wish to access from the model update
        :return: Result after fusion
        :rtype: `list`
        """

        for ct, party in enumerate(lst_parties):
            if party not in self._parties:
                self._parties[party] = PartyAttributes(party, lst_model_updates[ct], self.alpha0, self.beta0)
            else:
                self._parties[party].updates = lst_model_updates[ct]

        all_parties = [x for x in lst_parties if x not in set(self._blocked_parties)]
        lst_good_parties, lst_bad_parties = self.filter_bad_parties(all_parties, self._parties)

        # update alpha and beta for all parties
        for party in all_parties:
            if party in lst_good_parties:
                self._parties[party].alpha += 1
            if party in lst_bad_parties:
                self._parties[party].beta += 1
            score = beta.cdf(0.5, self._parties[party].alpha, self._parties[party].beta)
            if score >= self.block_threshold:
                self._blocked_parties.append(party)
                logger.info("Blocked party %s" % party)

        # computer average using good parties and update model weights
        final_weights = self.get_weighted_average(lst_good_parties, self._parties)
        return final_weights
