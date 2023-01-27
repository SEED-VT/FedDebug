"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging

import numpy as np

from ibmfl.exceptions import LocalTrainingException
from ibmfl.model.model_update import ModelUpdate
from ibmfl.party.training.local_training_handler import \
    LocalTrainingHandler

logger = logging.getLogger(__name__)


class FedPlusLocalTrainingHandler(LocalTrainingHandler):

    def __init__(self, fl_model, data_handler, hyperparams=None, **kwargs):
        """
        Initialize Fedplus LocalTrainingHandler with fl_model, data_handler

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
        self.fl_model = fl_model
        self.data_handler = data_handler
        info = kwargs.get('info')
        self.alpha = 0
        self.rho = 0
        if info is not None:
            self.alpha = info.setdefault("alpha", 0)
            self.rho = info.setdefault("rho", 0)
        self.mixed_model = None

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
        train_data, (_) = self.data_handler.get_data()
        _train_count = self.data_handler.get_train_counts()
        val_data = self.data_handler.get_val_data()

        try:
            lr = fit_params['hyperparams']['local']['optimizer']['lr']
        except KeyError:
            raise ValueError('lr value is not set in hyperparams.local.optimizer.lr config')
        except TypeError:
            logger.info("This is a PyTorch model. Trying to read configs again")
            try:
                lr = fit_params['hyperparams']['local']['training']['lr']
            except KeyError:
                raise ValueError('lr value is not set in hyperparams.local.training.lr config')

        try:
            num_epochs = fit_params['hyperparams']['local']['training']['epochs']
        except KeyError:
            raise ValueError('epochs value is not set in hyperparams.local.training.epochs config')

        self.update_model(fit_params['model_update'])

        self.get_train_metrics_pre()

        logger.info('Local training started...')
        if self.mixed_model is not None:
            logger.info('Solving minimization locally epoch by epoch for ' + str(num_epochs) + ' epochs')
            fit_params['hyperparams']['local']['training']['epochs'] = 1
            theta = 1 / (1 + self.alpha * lr)
            for _ in range(num_epochs):
                self.fl_model.fit_model(train_data, fit_params, val_data, local_params=self.hyperparams)
                local_model = [[] for x in range(np.shape(self.mixed_model)[0])]
                for i in range(np.shape(self.mixed_model)[0]):
                    local_model[i] = (1 - theta) * self.mixed_model[i] + theta * \
                                     self.fl_model.get_model_update().get('weights')[i]
                self.fl_model.update_model(model_update=ModelUpdate(weights=local_model))
        else:
            self.fl_model.fit_model(train_data, fit_params, val_data, local_params=self.hyperparams)

        update = self.fl_model.get_model_update()
        update.add('train_counts', _train_count)

        logger.info('Local training done, generating model update...')

        self.get_train_metrics_post()

        return update

    def update_model(self, model_update):
        """
        Update local model with model updates received from FusionHandler

        :param model_update: ModelUpdate
        :type model_update: `ModelUpdate`
        """
        try:
            if model_update is not None:
                self.soft_update_model(model_update)
                logger.info('Local model updated.')
            else:
                logger.info('No model update was provided.')
        except Exception as ex:
            raise LocalTrainingException('No query information is provided. ' + str(ex))

    def sync_model_impl(self, payload=None):
        """
        Update the local model with global ModelUpdate received \
        from the Aggregator.

        :param payload: data payload received from Aggregator
        :type payload: `dict`
        :return: Status of sync model request
        :rtype: `boolean`
        """
        status = False
        model_update = payload['model_update']
        self.soft_update_model(model_update)
        return status

    def soft_update_model(self, model_update, key='weights'):
        """
        Soft update to local model using fedplus algo

        :param model_update:ModelUpdate
        :type model_update: `ModelUpdate`
        :param key: model weights
        :type key:str
        :return:None
        """
        local_weights = self.fl_model.get_model_update().get(key)
        global_weights = model_update.get(key)
        soft_update_weights = np.add((1 - self.alpha) * np.array(local_weights),
                                     self.alpha * np.array(global_weights))
        self.fl_model.update_model(model_update=ModelUpdate(weights=soft_update_weights))
