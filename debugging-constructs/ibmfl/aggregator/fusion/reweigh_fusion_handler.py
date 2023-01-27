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

from ibmfl.model.model_update import ModelUpdate
from ibmfl.aggregator.fusion.iter_avg_fusion_handler import IterAvgFusionHandler

logger = logging.getLogger(__name__)


class ReweighFusionHandler(IterAvgFusionHandler):
    """
    Class for iterative averaging based fusion algorithms.
    An iterative fusion algorithm here referred to a fusion algorithm that
    sends out queries at each global round to registered parties for
    information, and use the collected information from parties to update
    the global model.
    The type of queries sent out at each round is the same. For example,
    at each round, the aggregator send out a query to request local model's
    weights after parties local training ends.
    The iterative algorithms can be terminated at any global rounds.

    In this class, the aggregator requests local model's weights from all
    parties at each round, and the averaging aggregation is performed over
    collected model weights. The global model's weights then are updated by
    the mean of all collected local models' weights.
    """

    def start_global_training(self):
        """
        Starts an iterative global federated learning training process.
        """
        self.curr_round = 0

        payload = {'is_handshake': True}
        lst_replies = self.query_all_parties(payload)

        global_counts = self.global_reweighing(lst_replies)

        payload = {'global_weights': True, 'global_counts': global_counts}
        lst_replies = self.query_all_parties(payload)

        while not self.reach_termination_criteria(self.curr_round):
            # construct ModelUpdate
            if self.current_model_weights:
                model_update = ModelUpdate(weights=self.current_model_weights)
            else:
                model_update = None

            payload = {'hyperparams': {'local': self.params_local},
                       'model_update': model_update
                       }
            logger.info('Model update' + str(model_update))

            # query all available parties
            lst_replies = self.query_all_parties(payload)

            self.update_weights(lst_replies)

            # Update model if we are maintaining one
            if self.fl_model is not None:
                self.fl_model.update_model(
                    ModelUpdate(weights=self.current_model_weights))

            self.curr_round += 1
            self.save_current_state()

    @staticmethod
    def global_reweighing(lst_replies):
        """


        :param lst_replies: party response with local DP counts for weight calculation
        :type lst_replies: `dict`
        :return: global counts for weight calculation
        :rtype: `dict`
        """
        unpriv_neg = 0
        unpriv_pos = 0
        priv_neg = 0
        priv_pos = 0

        for i in lst_replies:
            unpriv_neg += i['unp_neg']
            unpriv_pos += i['unp_pos']
            priv_neg += i['p_neg']
            priv_pos += i['p_pos']

        total_samples = int(priv_neg) + int(priv_pos) + int(unpriv_neg) + int(unpriv_pos)
        global_counts = {}
        global_counts['priv'] = (int(priv_neg) + int(priv_pos)) / total_samples / len(lst_replies)
        global_counts['unpriv'] = (int(unpriv_neg) + int(unpriv_pos)) / total_samples / len(lst_replies)
        global_counts['pos'] = (int(priv_pos) + int(unpriv_pos)) / total_samples / len(lst_replies)
        global_counts['neg'] = (int(unpriv_neg) + int(priv_neg)) / total_samples / len(lst_replies)
        global_counts['unpriv_neg'] = int(unpriv_neg) / total_samples / len(lst_replies)
        global_counts['unpriv_pos'] = int(unpriv_pos) / total_samples / len(lst_replies)
        global_counts['priv_neg'] = int(priv_neg) / total_samples / len(lst_replies)
        global_counts['priv_pos'] = int(priv_pos) / total_samples / len(lst_replies)

        return global_counts

