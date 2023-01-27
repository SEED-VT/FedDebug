"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging

import numpy as np

from ibmfl.model.model_update import ModelUpdate
from ibmfl.party.training.fedplus_local_training_handler import \
    FedPlusLocalTrainingHandler

logger = logging.getLogger(__name__)


class GeometricMedianFedPlusLocalTrainingHandler(FedPlusLocalTrainingHandler):

    def __init__(self, fl_model, data_handler, hyperparams=None, **kwargs):
        """
        Initialize GeometricMedianFedPlus LocalTrainingHandler \
        with fl_model, data_handler

        :param fl_model: model to be trained
        :type fl_model: `model.FLModel`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param hyperparams: Hyperparameters used for training.
        :type hyperparams: `dict`
        :param kwargs: Additional arguments to initialize a local training \
        handler, e.g., a crypto library object to help with encryption and \
        decryption.
        :type kwargs: `dict`
        :return None
        """
        super().__init__(fl_model, data_handler, hyperparams, **kwargs)

    def soft_update_model(self, model_update, key='weights'):
        """
        Soft update to local model using geometric median fedplus algo

        :param model_update:ModelUpdate
        :type model_update: `ModelUpdate`
        :param key: model weights
        :type key:str
        :return:None
        """
        EPS = 1E-6
        local_weights = self.fl_model.get_model_update().get(key)
        global_weights = model_update.get(key)
        diff = []
        for i in range(np.shape(global_weights)[0]):
            diff.append(global_weights[i].flatten() - local_weights[i].flatten())
        lambda_ = min(1, self.rho / (np.linalg.norm(np.concatenate(diff)) + EPS))
        self.mixed_model = [[] for x in range(np.shape(global_weights)[0])]
        for i in range(np.shape(global_weights)[0]):
            self.mixed_model[i] = lambda_ * global_weights[i] + (1 - lambda_) * local_weights[i]
