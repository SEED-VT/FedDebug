"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging

from ibmfl.exceptions import LocalTrainingException
from ibmfl.party.training.local_training_handler import \
    LocalTrainingHandler
from ibmfl.model.model_update import ModelUpdate

logger = logging.getLogger(__name__)


class GradientLocalTrainingHandler(LocalTrainingHandler):
    """
    Class for stochastic gradient descent based fusion algorithms.
    In this class, the aggregation is performed over gradients.
    """

    def train(self,  fit_params):
        """
        Retrieve the current gradient from party. With the provided
        model weights inside the `fit_param`, a model_weights including
        the current gradient information is generated and
        send through the connection.

        :param fit_params: Query instruction containing a set of model weights \
         from aggregator
        :type fit_params: `dict`
        :return: ModelUpdate
        :rtype: `ModelUpdate`
        """
        current_weight = fit_params.get('model_update') or None
        if current_weight is None:
            raise LocalTrainingException('New set of model weights must be '
                                         'provided to retrieve '
                                         'gradient information.')
        self.update_model(current_weight)

        logger.info('Local training started...')

        train_data, (_) = self.data_handler.get_data()
        gradient = self.fl_model.get_gradient(train_data)

        update = ModelUpdate(gradients=gradient)

        logger.info('Local training done, generating model update...')

        return update
