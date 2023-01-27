"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import numpy as np
import logging

from diffprivlib.models import GaussianNB

from ibmfl.aggregator.fusion.fusion_handler import FusionHandler
from ibmfl.model.model_update import ModelUpdate
from ibmfl.model.naive_bayes_fl_model import NaiveBayesFLModel

logger = logging.getLogger(__name__)


class NaiveBayesFusionHandler(FusionHandler):
    """
    Class for Gaussian Naive Bayes federated learning with differential
    privacy.

    Implements GaussianNB from diffprivlib, with party updates combined with
    the fusion handler.
    """

    def __init__(self, hyperparams, proto_handler, data_handler,
                 fl_model=None, **kwargs):
        """
        Initializes a NaiveBayesFusionHandler object with provided fl_model,
        data_handler, proto_handler and hyperparams.

        :param hyperparams: Hyperparameters used for training.
        :type hyperparams: `dict`
        :param proto_handler: Proto_handler that will be used to send message
        :type proto_handler: `ProtoHandler`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param fl_model: (optional) model to be trained
        :type fl_model: `model.FLModel`
        :param kwargs: Additional arguments to initialize a fusion handler.
        :type kwargs: `dict`
        """
        if fl_model is None:
            fl_model = NaiveBayesFLModel("naive-bayes", None, GaussianNB())

        super().__init__(hyperparams, proto_handler, data_handler, fl_model,
                         **kwargs)
        self.name = "NaiveBayesFusion"
        self.model_update = \
            fl_model.get_model_update() if fl_model else None

    def fusion_collected_responses(self, lst_model_updates, **kwargs):
        """
        Receives a list of model updates, where a model update is of the type
        `ModelUpdate`. Using the count, means and variances of each
        model_update, combines them into a single model update.

        :param lst_model_updates: list of model updates of type `ModelUpdate` \
        to be combined.
        :type lst_model_updates: `list`
        :return: Model update with combined counts, means and variances.
        :rtype: `ModelUpdate`
        """
        collected_theta = None
        collected_var = None
        collected_class_count = None

        # Begin with theta and class_count, as they're needed for var
        for model_update in lst_model_updates:
            if model_update.get("class_count") is None:
                continue

            if collected_class_count is None:
                collected_theta = np.zeros_like(model_update.get("theta"),
                                                dtype=float)
                collected_var = np.zeros_like(model_update.get("var"),
                                              dtype=float)
                collected_class_count = np.zeros_like(
                    model_update.get("class_count"))

            collected_theta += \
                np.array(model_update.get("theta")) * \
                np.array(model_update.get("class_count"))[:, np.newaxis]
            collected_class_count += model_update.get("class_count")

        if (collected_class_count == 0).any():
            collected_class_count[collected_class_count == 0] = np.infty

        collected_theta /= collected_class_count[:, np.newaxis]

        for model_update in lst_model_updates:
            collected_var += \
                (model_update.get("var") +
                 (model_update.get("theta") - collected_theta) ** 2) \
                * np.array(model_update.get("class_count"))[:, np.newaxis]

        collected_var /= collected_class_count[:, np.newaxis]

        if (collected_class_count == np.infty).any():
            collected_class_count[collected_class_count == np.infty] = 0

        return ModelUpdate(theta=collected_theta,
                           var=collected_var,
                           class_count=collected_class_count)

    def start_global_training(self):
        """
        Starts global federated learning training process.
        """
        payload = {'hyperparams': self.hyperparams,
                   'model_update': self.model_update
                   }

        # query all available parties
        lst_replies = self.query_all_parties(payload)

        # Collect all model updates for fusion:
        self.model_update = self.fusion_collected_responses(lst_replies)

        # Update model if we are maintaining one
        if self.fl_model is not None:
            self.fl_model.update_model(self.model_update)

    def get_global_model(self):
        """
        Returns last model_update

        :return: model_update
        :rtype: `ModelUpdate`
        """
        try:
            return ModelUpdate(theta=self.fl_model.model.theta_,
                               var=self.fl_model.model.var_,
                               class_count=self.fl_model.model.class_count_)
        except AttributeError:
            return ModelUpdate(theta=self.fl_model.model.theta_,
                               var=self.fl_model.model.sigma_,
                               class_count=self.fl_model.model.class_count_)

