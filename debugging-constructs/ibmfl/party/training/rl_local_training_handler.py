"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Module to where local training handler are implemented.
"""
import inspect
import logging

from ibmfl.data.env_spec import EnvHandler
from ibmfl.party.training.local_training_handler import \
    LocalTrainingHandler

logger = logging.getLogger(__name__)


class RLLocalTrainingHandler(LocalTrainingHandler):
    """
    Local training handler for RL
    """

    def __init__(self, fl_model, data_handler, hyperparams=None, **kwargs):
        """
        Initialize LocalTrainingHandler with fl_model, data_handler

        :param fl_model: model to be trained
        :type fl_model: `model.FLModel`
        :param data_handler: data handler that will be used to \
        obtain data and environment reference
        :type data_handler: `DataHandler`
        :param hyperparams: Hyperparameters used for training.
        :type hyperparams: `dict`
        :param kwargs: Additional arguments to initialize an RL local training \
        handler.
        :type kwargs: `Dict`
        :return: None
        """
        super().__init__(fl_model, data_handler, hyperparams, **kwargs)

        train_data, test_data = data_handler.get_data() or (None, None)
        env_class_ref = data_handler.get_env_class_ref()

        if not inspect.isclass(env_class_ref):
            raise ValueError('Environment reference should be a class reference '
                             'and not an instance')

        if not issubclass(env_class_ref, EnvHandler):
            raise ValueError(
                'Environment reference should be of type EnvHandler')

        fl_model.create_rl_trainer(
            env_class_ref, train_data=train_data, test_data=test_data)

    def train(self, fit_params=None):
        """
        Train locally using fl_model. At the end of training, a
        model_update with the new model information is generated and
        send through the connection.

        :param fit_params: (optional) Query instruction from aggregator
        :type fit_params: `dict`
        :return: ModelUpdate
        :rtype: `ModelUpdate`
        """
        train_data = None

        self.update_model(fit_params.get('model_update'))

        self.get_train_metrics_pre()

        logger.info('Local training started...')

        self.fl_model.fit_model(train_data, fit_params, local_params=self.hyperparams)

        self.get_train_metrics_post()

        update = self.fl_model.get_model_update()
        logger.info('Local training done, generating model update...')

        return update

    def eval_model(self, payload=None):
        """
        Evaluate the local model based on the local test data.

        :param payload: data payload received from Aggregator
        :type payload: `dict`
        :return: Dictionary of evaluation results
        :rtype: `dict`
        """

        x_test = None
        y_test = None
        evaluations = self.fl_model.evaluate((x_test, y_test))
        logger.info(evaluations)
        return evaluations
