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

from ibmfl.aggregator.fusion.gradient_fusion_handler import \
    GradientFusionHandler
from ibmfl.model.model_update import ModelUpdate
from ibmfl.exceptions import NotFoundException, HyperparamsException, GlobalTrainingException
from sklearn import metrics

logger = logging.getLogger(__name__)


class ZenoGradientFusionHandler(GradientFusionHandler):
    """
    Class for Zeno Fusion.
    Implements the Zeno algorithm presented here: http://proceedings.mlr.press/v97/xie19b.html
    """

    def __init__(self, hyperparams,
                 protocol_handler,
                 data_handler=None,
                 fl_model=None,
                 **kwargs):
        """
        Initializes an ZenoGradientFusionHandler object with provided information,
        such as protocol handler, fl_model, data_handler and hyperparams.

        :param hyperparams: Hyperparameters used for training.
        :type hyperparams: `dict`
        :param protocol_handler: Protocol handler used for handling learning \
        algorithm's request for communication.
        :type protocol_handler: `ProtoHandler`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param fl_model: model to be trained
        :type fl_model: `model.FLModel`
        :return: None
        """
        if not fl_model:
            raise NotFoundException('A model must be provided for Zeno')

        if not data_handler:
            raise NotFoundException('Data handler must be provided for Zeno')

        super().__init__(hyperparams,
                         protocol_handler,
                         data_handler,
                         fl_model,
                         **kwargs)

        self.name = "Zeno-SGD"
        self.lr = self.params_global.get('lr') or 0.1
        self.rho = self.params_global.get('zeno_rho') or 5e-4
        self.b = self.params_global.get('zeno_b') or 0
        self.zeno_batch = self.params_global.get('zeno_batch') or 4

        (self.x_train, self.y_train), (_, _) = self.data_handler.get_data()

    def fusion_collected_responses(self, lst_model_updates, key='gradients'):
        """
        Receives a list of model updates and computes the score for each party
        as defined in Zeno.

        :param lst_model_updates: List of model updates of type `ModelUpdate` \
        to be averaged.
        :type lst_model_updates: `list`
        :param key: A key indicating what values the method will aggregate over.
        :type key: `str`
        :return: results after aggregation
        :rtype: `list`
        """
        v = []

        if not self.b < len(lst_model_updates):
            raise HyperparamsException('zeno\'s parameter of b should be '
                                       'less than the number of parties')

        try:
            for update in lst_model_updates:
                v.append(np.array(update.get(key)))

            original_weights = np.array(self.current_model_weights.copy())

            indices = np.random.choice(self.x_train.shape[0],
                                       self.zeno_batch,
                                       replace=False)
            train_batch = self.x_train[indices], self.y_train[indices]

            orig_loss_value = self.fl_model.get_loss(train_batch)

            # compute the score for each party
            scores = []
            for grads in v:
                weights = np.array(self.current_model_weights) - \
                    self.lr * np.array(grads)
                norm_squared = np.sum([np.linalg.norm(t) ** 2 for t in grads])
                # compute the change in loss-function value
                self.fl_model.update_model(
                    ModelUpdate(weights=weights.tolist()))
                
                curr_loss = self.fl_model.get_loss(train_batch)
                scores.append(orig_loss_value - curr_loss - self.rho * norm_squared)

            self.fl_model.update_model(
                ModelUpdate(weights=original_weights.tolist()))

            # compute the mean update while ignoring
            # the updates from 'b' parties with lowest scores
            sorted_scores = sorted(
                enumerate(scores), key=lambda x: x[1])[-self.b:]
            results = np.mean(np.array([v[i]
                                        for i, _ in sorted_scores]), axis=0)
        except NotImplementedError as nie:
            logger.info("Error occurred while training! "
                        "Model is not compatible with Zeno fusion Handler")
            raise GlobalTrainingException(
                "Incompatible model and fusion types.")

        except Exception as ex:
            logger.exception(ex)
            raise GlobalTrainingException("Error occurred during training.")

        return results.tolist()
        
    def update_weights(self, lst_model_updates):
        """
        Update the global model's weights with the list of collected
        model_updates from parties.
        In this method, it calls self.fusion_collected_responses to average
        the gradients collected from parties, and performs a one-step
        gradient descent with learning rate as self.lr.

        :param lst_model_updates: List of model updates of type `ModelUpdate` \
        to be averaged.
        :type lst_model_updates: `lst`
        :return: None
        """
        agg_gradient = self.fusion_collected_responses(lst_model_updates,
                                                       key='gradients')
        new_weights = np.array(self.current_model_weights) - \
            self.lr * np.array(agg_gradient)
        self.current_model_weights = new_weights.tolist()
