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
import copy

from ibmfl.aggregator.fusion.fedplus_fusion_handler import FedplusFusionHandler
from ibmfl.exceptions import ModelUpdateException, FusionException

logger = logging.getLogger(__name__)


class GeometricMedianFedplusFusionHandler(FedplusFusionHandler):
    """
    Class for Geometric-Median Fedplus aggregation.

    Implements the Geometric Median Fedplus algorithm presented here : https://arxiv.org/pdf/2009.06303.pdf
    """

    def __init__(self, hyperparams, protocol_handler,
                 data_handler=None, fl_model=None, **kwargs):
        super().__init__(hyperparams,
                         protocol_handler,
                         data_handler,
                         fl_model, **kwargs)
        self.name = "GEODMEDIAN"

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
        w = []
        n_k = []
        try:
            for update in lst_model_updates:
                w.append(np.array(update.get('weights')))
                n_k.append(update.get('train_counts'))
        except ModelUpdateException as ex:
            logger.exception(ex)
            raise FusionException("Model updates are not appropriate for this fusion method.  Check local training.")

        x_hat = self.aggregate(w, n_k)
        if self.round == 1:
            return x_hat

        x_hat_new = [[] for x in range(np.shape(x_hat)[0])]
        iter_ = 0
        error = 1
        while error >= 1e-4 and iter_ <= 10000:
            v_n = []
            for p in range(len(w)):
                v = [[] for x in range(np.shape(x_hat)[0])]
                diff = []
                p_model = w[p][1]
                for i in range(np.shape(x_hat)[0]):
                    diff.append(x_hat[i].flatten() - p_model[i].flatten())
                coeff = max(0, 1 - self.rho / np.linalg.norm(np.concatenate(diff)))
                for i in range(np.shape(x_hat)[0]):
                    v[i] = coeff * (p_model[i] - x_hat[i])
                v_n.append(np.array(v))
            v_n_agg = self.aggregate(v_n, n_k)
            for i in range(np.shape(x_hat)[0]):
                x_hat_new[i] = x_hat[i] - v_n_agg[i]
            errors = []
            for i in range(np.shape(x_hat)[0]):
                errors.append(x_hat_new[i].flatten() - x_hat[i].flatten())
            error = np.linalg.norm(np.concatenate(errors))
            x_hat = copy.deepcopy(x_hat_new)
            iter_ += 1

        self.round += 1
        return x_hat_new
