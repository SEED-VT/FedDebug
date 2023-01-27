"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging
from ibmfl.party.training.local_training_handler import \
    LocalTrainingHandler

logger = logging.getLogger(__name__)


class FedAvgLocalTrainingHandler(LocalTrainingHandler):

    def train(self,  fit_params=None):
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

        self.update_model(fit_params['model_update'])

        logger.info('Local training started...')

        self.fl_model.fit_model(train_data, fit_params, local_params=self.hyperparams)

        update = self.fl_model.get_model_update()
        update.add('train_counts', _train_count)

        logger.info('Local training done, generating model update...')

        return update
