"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import abc
import time
import logging
import threading

from multiprocessing.pool import ThreadPool
from ibmfl.message.message import Message, ResponseMessage
from ibmfl.message.message_type import MessageType
from ibmfl.exceptions import LocalTrainingException
from ibmfl.exceptions import PreprocessException
from ibmfl.party.status_type import StatusType

logger = logging.getLogger(__name__)


class PartyProtocolHandler(abc.ABC):
    """
    Base class for all PartyProtocolHandlers
    """

    def __init__(self, fl_model, connection, data_handler,
                 local_training_handler, metrics_recorder=None,
                 hyperparams=None, agg_info=None, synch=False, is_private=True,
                 local_preprocess_handler=None,
                 **kwargs):
        """
        Initiate PartyProtocolHandler with provided fl_model, connection,
        data_handler and local_training_handler, hyperparams, aggregator
        information and synchronous flag are optional.

        :param fl_model: model to be trained
        :type fl_model: `model.FLModel`
        :param connection: connection that will be used to send messages
        :type connection: `Connection`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param local_training_handler: local training handler class that \
        handles the access of the local model, for example, local training \
        of the model, prediction based on the model, etc.
        :type local_training_handler: `LocalTrainingHandler`
        :param hyperparams: Hyperparameters used for training.
        :type hyperparams: `Hyperparameters`
        :param agg_info: communication information related to aggregator
        :type agg_info: `dict`
        :param synch: get model update synchronously
        :type synch: `boolean`

        :return: None
        """

        self.fl_model = fl_model
        self.data_handler = data_handler
        self.connection = connection
        self.hyperparams = hyperparams
        self.agg_info = agg_info
        self.lock = threading.Lock()
        self.status = StatusType.IDLE
        self.synch = synch
        self.pool = ThreadPool(processes=1)
        self.local_training_handler = local_training_handler
        self.metrics_recorder = metrics_recorder
        if ( self.local_training_handler is not None ):
            self.local_training_handler.set_metrics_recorder_obj(self.metrics_recorder)
        self.local_preprocess_handler = local_preprocess_handler
        self.is_private = is_private

    def get_handle(self, message_type):
        """
        Get handler for given message type.

        :param message_type: request message type
        :type message_type: `int`
        :return: a handler which was assigned for given message type
        """
        if message_type == MessageType.SAVE_MODEL.value:
            return self.local_training_handler.save_model
        elif message_type == MessageType.SYNC_MODEL.value:
            return self.local_training_handler.sync_model
        elif message_type == MessageType.EVAL_MODEL.value:
            return self.local_training_handler.eval_model
        elif message_type == MessageType.TRAIN.value:
            return self.local_training_handler.train
        elif message_type == MessageType.PREPROCESS.value:
            return self.local_preprocess_handler.local_preprocess
        elif message_type == MessageType.REQUEST_CERT.value:
            return self.local_training_handler.request_cert
        elif message_type == MessageType.GENERATE_KEYS.value:
            return self.local_training_handler.generate_keys
        elif message_type == MessageType.DISTRIBUTE_KEYS.value:
            return self.local_training_handler.distribute_keys
        elif message_type == MessageType.SET_KEYS.value:
            return self.local_training_handler.set_keys
        else:
            raise LocalTrainingException("Unsupported message type!")

    def handle_request(self, msg):
        """
        Handle all incoming requests and route it to respective methods
        in local training handler.

        :param msg: Message object form connection
        :type msg: `Message`
        :return: Response message sent back to requester
        :rtype: ResponseMessage

        """
        logger.info("Received request from aggregator")
        message_type = msg.message_type
        logger.info("Received request in with message_type:  " +
                    str(message_type))

        data = msg.get_data()

        response_msg = ResponseMessage(req_msg=msg)
        response_data = {"status": "success"}
        status = True
        logger.info("Received request in PH " + str(message_type))

        try:
            if message_type is MessageType.TRAIN.value:
                if self.metrics_recorder:
                    self.metrics_recorder.add_entry()
                    self.metrics_recorder.set_round_no(self.local_training_handler.n_completed_trains)
            elif message_type is MessageType.STOP.value:
                self.status = StatusType.STOPPING
                # May be no need to send response
                response_msg = ResponseMessage(message_type=MessageType.ACK.value,
                                               id_request=-1,
                                               data={'ACK': True})
                logger.info("received a STOP request")
                return response_msg
        
            self.wait_for_model_initialization()
            handler = self.get_handle(message_type)

            response = handler(data.get('payload'))
            if message_type is MessageType.TRAIN.value:
                self.local_training_handler.n_completed_trains += 1
                if not self.is_private:
                    metrics_handler = self.get_handle(MessageType.EVAL_MODEL.value)
                    metrics = metrics_handler(data.get('payload'))
                    response_data['metrics'] = metrics
            elif message_type is MessageType.EVAL_MODEL.value:
                self.local_training_handler.n_completed_evals += 1

        except Exception as ex:
            logger.exception(ex)
            raise LocalTrainingException(
                "Error occurred while handling request")

        response_data['payload'] = response
        if not status:
            response_data['status'] = "error"

        response_msg.set_data(response_data)
        return response_msg

    def execute_async(self, id_request, msg):
        """
        Handle run in a different thread to allow asynchronous requests.

        :param msg: Message object form connection
        :type msg: `Message`
        """
        try:
            # Acquire lock so that we do not run train twice
            self.lock.acquire()
            logger.info("Handling async request in a separate thread")
            response_msg = self.handle_request(msg)

        except Exception as ex:
            logger.info("Exception occurred while async handling of msg: "
                        + msg)
            logger.exception(ex)

            response_msg = ResponseMessage(msg)
            response_msg.set_data({
                "status": "error",
                "payload": None
            })
        logger.info("successfully finished async request")
        self.connection.send_message(self.agg_info, response_msg)

        if self.metrics_recorder:
            with open(self.metrics_recorder.get_output_file(), 'w') as metrics_file:
                metrics_output_type = self.metrics_recorder.get_output_type()
                if metrics_output_type == 'json':
                    metrics_output_data = self.metrics_recorder.to_json()
                else:
                    logger.info("Bad metrics output filetype. Defaulting to json.")
                    metrics_output_data = self.metrics_recorder.to_json()
                metrics_file.write('{}\n'.format(metrics_output_data))

        # Release lock
        self.lock.release()
        return

    def handle_async_request(self, msg):
        """
        Handle all incoming requests asynchronously and route it to respective
         methods in local training handler.

        :param msg: Message object form connection
        :type msg: `Message`
        :return: Response message sent back to requester
        :rtype: ResponseMessage
        """
        try:
            response_msg = ResponseMessage(message_type=MessageType.ACK.value,
                                           id_request=-1,
                                           data={'ACK': True})
            logger.info("received a async request")

            id_request = msg.get_header()['id_request']

            self.pool.apply_async(
                self.execute_async, args=(id_request, msg))
            logger.info("finished async request")
        except Exception as ex:
            logger.info(ex)

        return response_msg

    def handle_prepreocess_request(self, msg):
        logger.info("Received preprocess request from aggregator")
        message_type = msg.message_type
        logger.info("Received request in with message_type: {}".format(message_type))

        data = msg.get_data()

        response_msg = ResponseMessage(req_msg=msg)
        response_data = {"status": "success"}
        try:
            handler = self.get_handle(message_type)
            response = handler(data.get('payload'))
            response_data['payload'] = response
        except Exception as ex:
            logger.exception(ex)
            response_data['status'] = "error"
            response_data['payload'] = {'error message': 'Error occurred at local preprocess'}
            raise PreprocessException("Error occurred while handling preprocess request")
        response_msg.set_data(response_data)
        return response_msg

    def print_evaluate_local_model(self, hyperparams=None):
        """
        Print local evaluations in console

        :param hyperparams: hyperparams for evaluation on the local model
        :type hyperparams: `dict`
        :return: None
        """
        try:
            evaluations = self.local_training_handler.eval_model(hyperparams)
            # logger.info(evaluations)
        except Exception as ex:
            logger.info('Error occurred on evaluating model on local data. ' +
                        str(ex))


    def set_model(self, model):
        """Set model instance
        :param model: model to be trained
        :type model: `model.FLModel`
        :return: None
        """
        self.fl_model = model

    def set_training_handler(self, training_handler):
        """Set localtraining handler instance

        :param local_training_handler: local training handler class that \
        handles the access of the local model, for example, local training \
        of the model, prediction based on the model, etc.
        :type local_training_handler: `LocalTrainingHandler`        
        """
        self.local_training_handler = training_handler
        if ( self.local_training_handler is not None ):
            self.local_training_handler.set_metrics_recorder_obj(self.metrics_recorder)

    def wait_for_model_initialization(self):
        """Wait until model and localtraininghandler are initialized
        """
        logger.debug("Waiting for model initialization to finish")

        while not (self.fl_model and self.local_training_handler):
            time.sleep(10)


class PartyProtocolHandlerRabbitMQ(PartyProtocolHandler):
    """
    Extended class for PartyProtocolHandler for using with RabbitMQ connection
    """

    def __init__(
            self,
            fl_model,
            connection,
            data_handler,
            local_training_handler,
            hyperparams=None,
            agg_info=None,
            synch=True,
            is_private=True,
            **kwargs
    ):
        """
        Initiate PartyProtocolHandlerRabbitMQ with provided fl_model,
        connection and data_handler and hyperparams

        :param fl_model: model to be trained
        :type fl_model: `model.FLModel`
        :param connection: connection that will be used to send messages
        :type connection: `Connection`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param hyperparams: Hyperparameters used for training.
        :type hyperparams: `Hyperparameters`
        :param agg_info: communication information related to aggregator
        :type agg_info: `dict`
        :param synch: get model update synchronously
        :type synch: `boolean`
        """
        if synch:
            super(PartyProtocolHandlerRabbitMQ, self).__init__(
                fl_model,
                connection,
                data_handler,
                local_training_handler,
                hyperparams,
                agg_info,
                synch,
                is_private,
                **kwargs
            )

        else:
            raise Exception(
                'RabbitMQ connection currently only supports synchronous mode'
            )

    def handle_async_request(self, msg):
        """Handle all incoming requests asynchronously and route it to
        respective methods in local training handler

        :param msg: Message object form connection
        :type msg: `Message`
        :return: Response message sent back to requester
        :rtype: ResponseMessage
        """
        try:
            response_msg = ResponseMessage(
                message_type=MessageType.ACK.value,
                id_request=-1, data={'ACK': True}
            )
            logger.info("received a async request")

            id_request = msg.get_header()['id_request']

            self.execute_async(id_request, msg)

            logger.info("finished async request")

        except Exception as ex:
            logger.info(ex)

        return response_msg
