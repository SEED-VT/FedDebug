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


class ReweighLocalTrainingHandler(LocalTrainingHandler):

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
        self.sample_weight = []

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

        update = {}
        if 'is_handshake' in fit_params:
            update = self.data_handler.get_hist()

        elif 'global_weights' in fit_params:
            train_data, (_) = self.data_handler.get_data()
            sensitive_attribute = self.data_handler.get_sa()
            columns = self.data_handler.get_col_names()

            train_data = pd.DataFrame(data=train_data[0])
            class_values = train_data[1].tolist()
            train_data.columns = columns
            train_data['class'] = class_values

            global_counts = fit_params['global_counts']

            priv = global_counts['priv']
            unpriv = global_counts['unpriv']
            pos = global_counts['pos']
            neg = global_counts['neg']
            unpriv_neg = global_counts['unpriv_neg']
            unpriv_pos = global_counts['unpriv_pos']
            priv_neg = global_counts['priv_neg']
            priv_pos = global_counts['priv_pos']

            weight = []
            for index, row in train_data.iterrows():
                if row[sensitive_attribute] == 0 and row['class'] == 0:
                    weight.append(unpriv * neg / unpriv_neg)
                elif row[sensitive_attribute] == 0 and row['class'] == 1:
                    weight.append(unpriv * pos / unpriv_pos)
                elif row[sensitive_attribute] == 1 and row['class'] == 0:
                    weight.append(priv * neg / priv_neg)
                elif row[sensitive_attribute] == 1 and row['class'] == 1:
                    weight.append(priv * pos / priv_pos)

            self.sample_weight = np.array(weight)

        else:
            train_data, (_) = self.data_handler.get_data()
            if self.sample_weight == []:
                self.sample_weight = self.data_handler.get_weight()

            fit_params['sample_weight'] = self.sample_weight

            self.update_model(fit_params.get('model_update'))

            logger.info('Local training started...')

            self.fl_model.fit_model(train_data, fit_params, local_params=self.hyperparams)

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

        y_pred = self.fl_model.predict(x_test)
        evaluations['Fairness Report'] = fairness_report(
            x_test, y_test, y_pred, sensitive_attribute, cols)

        y_train_pred = self.fl_model.predict(x_train)
        evaluations['Underestimation Index'] = uei(y_train, y_train_pred)

        logger.info(evaluations)
        return evaluations
