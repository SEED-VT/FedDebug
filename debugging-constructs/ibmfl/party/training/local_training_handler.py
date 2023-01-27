"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging

from ibmfl.exceptions import LocalTrainingException, \
    ModelUpdateException
import numpy as np

logger = logging.getLogger(__name__)

class LocalTrainingHandler():

    def __init__(self, fl_model, data_handler, hyperparams=None, evidencia=None, **kwargs):
        """
        Initialize LocalTrainingHandler with fl_model, data_handler

        :param fl_model: model to be trained
        :type fl_model: `model.FLModel`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param hyperparams: Hyperparameters used for training.
        :type hyperparams: `dict`
        :param evidencia: evidencia to use
        :type evidencia: `evidencia.EvidenceRecorder`
        :param kwargs: Additional arguments to initialize a local training \
        handler, e.g., a crypto library object to help with encryption and \
        decryption.
        :type kwargs: `dict`
        :return None
        """
        self.fl_model = fl_model
        self.data_handler = data_handler
        self.hyperparams = hyperparams
        self.evidencia = evidencia

        self.metrics_recorder = None
        self.n_completed_trains = 0
        self.n_completed_evals = 0

        if self.evidencia:
            from ibmfl.evidencia.util.hashing import hash_np_array, \
                hash_model_update


    def set_metrics_recorder_obj(self, metrics_recorder):
        """
        Set metrics instance variable to the input parameter. We do this because the \
        party_protocol_handler tells the local_training_handler which metrics object to use; the \
        local_training_handler can be constructed somewhere else, so we don't want to force the \
        metrics object to necessarily exist at that time.

        :param metrics_recorder: Metrics-recording object (probably empty at time this is called)
        :type metrics_recorder: `MetricsRecorder`
        """
        self.metrics_recorder = metrics_recorder

    def update_model(self, model_update):
        """
        Update local model with model updates received from FusionHandler

        :param model_update: ModelUpdate
        :type model_update: `ModelUpdate`
        """
        try:
            if model_update is not None:
                self.fl_model.update_model(model_update)
                logger.info('Local model updated.')
            else:
                logger.info('No model update was provided.')
        except Exception as ex:
            raise LocalTrainingException('No query information is provided. '+ str(ex))

    def get_train_metrics_pre(self):
        """
        Call the post-train metrics hook. This hook runs immediately before the training starts at
        each party during the routine corresponding to the TRAIN command.

        :param: None
        :return: None
        """
        if self.metrics_recorder:
            try:
                # TODO: find sound way of determining if we really want to do a pre_train (i.e.
                # "synced model") eval
                if self.metrics_recorder.compute_pre_train_eval and self.get_n_completed_trains() > 0:
                    ret = self.data_handler.get_data()
                    if ret is not None:
                        (_), test_dataset = ret
                    else:
                        test_dataset = None
                    pre_eval_results = self.fl_model.evaluate(test_dataset)
                else:
                    pre_eval_results = None
                # collect metrics specific to the model class, that the user may customize
                additional_metrics = self.fl_model.get_custom_metrics_pre()
                self.metrics_recorder.pre_train_hook(pre_eval_results, additional_metrics)
            except Exception as e:
                logger.exception(str(e))
                raise LocalTrainingException(
                    'Error occurred while running pre-train hooks')

    def get_train_metrics_post(self):
        """
        Call the post-train metrics hook. This hook runs immediately after the training finishes at
        each party during the routine corresponding to the TRAIN command.

        :param: None
        :return: None
        """
        if self.metrics_recorder:
            try:
                train_result = self.fl_model.get_train_result()
                # TODO: find sound way of determining if we really want to do a post_train (i.e.
                # "locally-trained") eval
                if self.metrics_recorder.compute_post_train_eval:
                    ret = self.data_handler.get_data()
                    if ret is not None:
                        (_), test_dataset = ret
                    else:
                        test_dataset = None
                    post_eval_results = self.fl_model.evaluate(test_dataset)
                else:
                    post_eval_results = None
                additional_metrics = self.fl_model.get_custom_metrics_post()
                self.metrics_recorder.post_train_hook(train_result, post_eval_results, additional_metrics)
            except Exception as e:
                logger.exception(str(e))
                raise LocalTrainingException(
                    'Error occurred while running post-train hooks')

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


        if self.evidencia:
            self.evidencia.add_claim("training_data_hash", "'{}'".format(hash_np_array(train_data[0])))
            self.evidencia.add_claim("training_data_labels_hash", "'{}'".format(hash_np_array(train_data[1])))
            self.evidencia.add_claim("training_data_size", str(train_data[0].shape[0]))
            self.evidencia.add_claim("training_data_labels_number",
                                                  str(np.unique(train_data[1], axis=0).shape[0]))
            # TODO labels are hardcoded
            labels_text = ['Verkehr & Mobilität', 'Städtebau & Stadtraum', 'Sonstiges',
                            'Grün & Erholung', 'Soziales & Kultur',
                            'Wohnen & Arbeiten',
                            'Sport & Freizeit', 'Klima & Umweltschutz']
            self.evidencia.add_claim("labels_list", "'{}'".format(str(labels_text).replace('\'', '"')))
            # also log number of training instances per label
            (labels, counts) = np.unique(np.argmax(train_data[1], axis=1), return_counts=True)

            for idx, _ in np.ndenumerate(labels):
                self.evidencia.add_claim("training_data_count_per_label",
                                                          '{}, {}'.format(labels[idx], counts[idx]))

        self.update_model(fit_params.get('model_update'))


        if self.evidencia:
            self.evidencia.add_claim("received_model_update", "'\"{}\"'".format(
            hash_model_update(self.fl_model.get_model_update())))

        self.get_train_metrics_pre()

        logger.info('Local training started...')

        self.fl_model.fit_model(train_data, fit_params, local_params=self.hyperparams)

        update = self.fl_model.get_model_update()
        logger.info('Local training done, generating model update...')

        if self.evidencia:
            self.evidencia.add_claim("sent_model_update", "'\"{}\"'".format(hash_model_update(update)))

        self.get_train_metrics_post()

        return update

    def save_model(self, payload=None):
        """
        Save the local model.

        :param payload: data payload received from Aggregator
        :type payload: `dict`
        :return: Status of save model request
        :rtype: `boolean`
        """
        status = False
        try:
            self.fl_model.save_model()
            status = True
        except Exception as ex:
            logger.error("Error occurred while saving local model")
            logger.exception(ex)

        return status

    def get_update_metrics_pre(self):
        """
        Call the pre-update metrics hook. This hook runs before the model update from the SYNC
        command, but after the SYNC command instruction has been received.

        :param: None
        :return: None
        """
        if self.metrics_recorder:
            try:
                self.metrics_recorder.pre_update_hook()
            except Exception as e:
                logger.exception(str(e))
                raise LocalTrainingException(
                    'Error occurred while running pre-update hooks')

    def get_update_metrics_post(self):
        """
        Call the post-update metrics hook. This hook runs after the model update from the SYNC
        command, but still during the routine corresponding to that SYNC.

        :param: None
        :return: None
        """
        if self.metrics_recorder:
            try:
                self.metrics_recorder.post_update_hook()
            except Exception as e:
                logger.exception(str(e))
                raise LocalTrainingException(
                    'Error occurred while running post-update hooks')

    def sync_model_impl(self, payload=None):
        """
        Update the local model with global ModelUpdate received
        from the Aggregator. This function is meant to be 
        overridden in base classes as opposed to sync_model, which
        contains boilerplate for exception handling and metrics.

        :param payload: data payload received from Aggregator
        :type payload: `dict`
        :return: Status of sync model request
        :rtype: `boolean`
        """
        status = False
        model_update = payload['model_update']
        status = self.fl_model.update_model(model_update)
        return status
    
    def sync_model(self, payload=None):
        """
        Update the local model with global ModelUpdate received
        from the Aggregator.

        :param payload: data payload received from Aggregator
        :type payload: `dict`
        :return: Status of sync model request
        :rtype: `boolean`
        """
        status = False
        if payload is None or 'model_update' not in payload:
            raise ModelUpdateException(
                "Invalid Model update request aggregator")

        self.get_update_metrics_pre()

        try:
            status = self.sync_model_impl(payload)
        except Exception as ex:
            logger.error("Exception occurred while sync model")
            logger.exception(ex)

        self.get_update_metrics_post()

        return status

    def eval_model(self, payload=None):
        """
        Evaluate the local model based on the local test data.

        :param payload: data payload received from Aggregator
        :type payload: `dict`
        :return: Dictionary of evaluation results
        :rtype: `dict`
        """

        (_), test_dataset = self.data_handler.get_data()
        evaluations = dict()
        try:
            evaluations = self.fl_model.evaluate(test_dataset)
            logger.info(evaluations)

        except Exception as ex:
            logger.error("Expecting the test dataset to be of type tuple. "
                         "However, test dataset is of type "
                         + str(type(test_dataset)))
            logger.exception(ex)

        return evaluations

    def get_n_completed_trains(self):
        """
        Return the number of completed executions of the TRAIN command at the party side

        :param: None
        :return: Number indicating how many TRAINs have been completed
        :rtype: `int`
        """
        return self.n_completed_trains

    def get_n_completed_evals(self):
        """
        Return the number of completed executions of the EVAL command at the party side

        :param: None
        :return: Number indicating how many EVALs have been completed
        :rtype: `int`
        """
        return self.n_completed_evals
