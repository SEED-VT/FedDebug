"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging
import numpy as np

from ibmfl.party.training.local_training_handler import \
    LocalTrainingHandler

logger = logging.getLogger(__name__)


class PFNMLocalTrainingHandler(LocalTrainingHandler):

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
        y = train_data[1]

        y_unq, y_cnts = np.unique(y, return_counts=True)
        _class_counts = {
            int(cls_label): int(cls_counts) for (cls_label, cls_counts) in
            zip(y_unq, y_cnts)
        }

        self.update_model(fit_params['model_update'])

        logger.info('Local training started...')

        self.fl_model.fit_model(train_data, fit_params, local_params=self.hyperparams)

        update = self.fl_model.get_model_update()
        update.add('class_counts', _class_counts)

        # Different frameworks have different return dimension.
        # PyTorch returns (features_out, features_in) dimensional weights for linear layers
        # Keras returns (features_in, features_out) dimensional weights for linear layers
        # Thus, this flag indicates if a transpose of the weight matrix is required or not

        update.add('transpose_weight', self.fl_model.model_type == 'PyTorch')
        logger.info('Local training done, generating model update...')

        return update
