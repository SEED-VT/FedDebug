"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging
import pandas as pd
import numpy as np

from ibmfl.party.training.local_training_handler import LocalTrainingHandler
from ibmfl.util.fairness_metrics.metrics import fairness_report, uei

logger = logging.getLogger(__name__)


class PRLocalTrainingHandler(LocalTrainingHandler):

    def __init__(self, fl_model, data_handler, hyperparams=None, **kwargs):
        """
        Initialize LocalTrainingHandler with fl_model, data_handler

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

    def eval_model(self, payload=None):
        """
        Evaluate the local model based on the local test data.

        :param payload: data payload received from Aggregator
        :type payload: `dict`
        :return: Dictionary of evaluation results
        :rtype: `dict`
        """

        train_dataset, test_dataset = self.data_handler.get_data()
        cols = self.data_handler.get_col_names()
        sensitive_attribute = self.data_handler.get_sa()

        try:
            x_train = train_dataset[0]
            y_train = train_dataset[1]
            x_test = test_dataset[0]
            y_test = test_dataset[1]
        except Exception as ex:
            logger.error("Expecting the test dataset to be of type tuple. "
                         "However, test dataset is of type "
                         + str(type(test_dataset)))
            logger.exception(ex)

        evaluations = self.fl_model.evaluate_model(x_test, y_test)
        y_pred = self.fl_model.predict_pr(test_dataset)
        evaluations['Fairness Report'] = fairness_report(
            x_test, y_test, y_pred, sensitive_attribute, cols)

        y_train_pred = self.fl_model.predict(x_train)
        evaluations['Underestimation Index'] = uei(y_train, y_train_pred)

        logger.info(evaluations)
        return evaluations
