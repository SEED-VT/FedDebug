"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Module where fusion algorithms are implemented.
"""
import logging
import numpy as np

from ibmfl.aggregator.fusion.iter_avg_fusion_handler import \
    IterAvgFusionHandler
from ibmfl.exceptions import ModelUpdateException, FusionException

logger = logging.getLogger(__name__)


class FedAvgFusionHandler(IterAvgFusionHandler):
    """
    Class for weight based Federated Averaging aggregation.

    Implements the FedAvg algorithm presented
    here: https://arxiv.org/pdf/1602.05629.pdf
    """

    def __init__(self, hyperparams,
                 protocol_handler,
                 data_handler=None,
                 fl_model=None,
                 **kwargs):
        """
        Initializes an FedAvgFusionHandler object with provided fl_model,
        data_handler and hyperparams.

        :param hyperparams: Hyperparameters used for training.
        :type hyperparams: `dict`
        :param protocol_handler: Protocol handler used for handling learning \
        algorithm's request for communication.
        :type protocol_handler: `ProtoHandler`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param fl_model: Model to be trained
        :type fl_model: `model.FLModel`
        :param kwargs: Additional arguments to initialize a fusion handler.
        :type kwargs: `dict`
        """

        super().__init__(hyperparams,
                         protocol_handler,
                         data_handler,
                         fl_model,
                         **kwargs)
        self.name = "FedAvg"
        self._eps = 1e-6

    def fusion_collected_responses(self, lst_model_updates):
        """
        Receives a list of model updates, where a model update is of the type
        `ModelUpdate`, using the weights included in each model_update, it
        finds the mean. It returns the weighted averaged model weights,
        where these weights to average model weights
        depends on parties' sample size and indicating by key `train_counts`.

        :param lst_model_updates: List of model updates of type `ModelUpdate` \
        to be averaged.
        :type lst_model_updates: `list`
        :return: weighted averaged model weights
        :rtype: `list`
        """

        w = []
        n_k = []

        try:
            for update in lst_model_updates:
                w.append(np.array(update.get('weights')))
                n_k.append(update.get('train_counts'))
        except ModelUpdateException as ex:
            logger.exception(ex)
            raise FusionException("Model updates are not appropriate for this fusion method.  Check local training.")
            

        n_norm = n_k / (np.sum(n_k) + self._eps)
        weights = np.sum([w[i] * n_norm[i] for i in range(len(n_k))], axis=0)

        return weights.tolist()
