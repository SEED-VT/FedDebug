"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Module to where fedplus fusion algorithms are implemented.
"""
import logging
import numpy as np

from ibmfl.aggregator.fusion.iter_avg_fusion_handler import IterAvgFusionHandler

logger = logging.getLogger(__name__)


class FedplusFusionHandler(IterAvgFusionHandler):
    """
    Class for fedplus fusion algorithms which provides a
    Unified Approach to Robust Personalized Federated Learning.

    Fedplus algorithms presented here : https://arxiv.org/pdf/2009.06303.pdf
    """

    def __init__(self, hyperparams, protocol_handler,
                 data_handler=None, fl_model=None, **kwargs):
        super().__init__(hyperparams,
                         protocol_handler,
                         data_handler,
                         fl_model, **kwargs)

        self.params_global = hyperparams.get('global') or {}
        self.rho = self.params_global.get('rho') or 0
        self.round = 1
        self._eps = 1e-6

    def aggregate(self, w, n_k):
        """
        :param w: weights array
        :param n_k: sample counts array
        :return: aggregated weights array using weighted average
        """
        n_norm = n_k / (np.sum(n_k) + self._eps)
        weights = np.sum([w[i] * n_norm[i] for i in range(len(n_k))], axis=0)
        return weights.tolist()
