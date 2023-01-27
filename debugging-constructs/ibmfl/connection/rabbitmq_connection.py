"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging
import threading

import pycloudmessenger.ffl.abstractions as ffl
import pycloudmessenger.ffl.fflapi as fflapi
from pycloudmessenger.ffl.fflapi import Aggregator
from pycloudmessenger.ffl.fflapi import Participant
from pycloudmessenger.ffl.fflapi import User

from ibmfl.connection.connection import FLConnection
from ibmfl.connection.connection import FLSender
from ibmfl.connection.connection import FLReceiver
from ibmfl.message.message import Message
from ibmfl.message.message_type import MessageType
from ibmfl.message.serializer_types import SerializerTypes
from ibmfl.message.serializer_factory import SerializerFactory
from ibmfl.exceptions import InvalidServerConfigurationException


logger = logging.getLogger(__name__)


class RabbitMQConnection(FLConnection):

    def __init__(self, config):
        """
        Initialized the connection object and validates the config
        provided to this connection instance

        :param config: dictionary of configuration provided to connection
        :type config: `dict`
        """
        self.settings = self.process_config(config)
        self.sender = None
        self.receiver = None

        ffl.Factory.register(
            'cloud',
            fflapi.Context,
            fflapi.User,
            fflapi.Aggregator,
            fflapi.Participant
        )

        context = ffl.Factory.context(
            'cloud',
            self.settings.get('credentials'),
            dispatch_threshold = 0
        )

        role = self.settings.get('role')
        if role == 'aggregator':
            self.comms = [ffl.Factory.aggregator(
                context,
                task_name=self.settings.get('task_name')
            )]
        else:
            self.comms = [ffl.Factory.user(context)]

    def initialize(self, **kwargs):
        """
        Initialize receiver and sender
        """
        router = kwargs.get('router')
        self.initialize_sender()
        self.initialize_receiver(router=router)
        self.sender.receiver = self.receiver

    def initialize_receiver(self, router=None):
        """
        Initialize server using the settings and handler

        :param router: Router object describing the routes for each request \
            which are passed down to PH
        :type router: `Router`
        """
        self.receiver = RabbitMQReceiver(router, self.comms)
        self.receiver.initialize()
        self.sender.receiver = self.receiver

    def initialize_sender(self):
        """
        Initialize a sender object using the settings provided during
        connection creation
        """
        self.sender = RabbitMQSender(self.comms, self.settings)
        self.sender.initialize()

    def process_config(self, config):
        """
        Validates the configuration provided to RabbitMQ connection and returns
        a settings dictionary with all the information extracted from config

        :param config: configuration sent from application
        :type config: `dict`
        :return: settings
        :rtype: `dict`
        """
        if config:
            credentials = config.get('credentials')
            user = config.get('user')
            password = config.get('password')
            task_name = config.get('task_name')
            role = config.get('role')
            settings = {
                'credentials': credentials,
                'user': user,
                'password': password,
                'task_name': task_name,
                'role': role
            }
        else:
            raise InvalidServerConfigurationException(
                'No connection configuration found')

        return settings

    def get_connection_config(self):
        """Provide a connection information such that a node can communicate 
        to other nodes on how to communicate with it.

        """
        return {}

    def start(self):
        """
        Starts the receiver in a new thread
        """
        self.start_receiver()

    def start_receiver(self):
        """
        Starts the receiver in a new thread
        """
        server_process = threading.Thread(target=self.receiver.start)
        server_process.start()
        self.sender.receiver_thread = server_process

    def stop(self):
        """
        Stop and cleanup the connection
        """
        logger.info('Stopping Receiver and Sender')
        self.receiver.set_stopable()
        self.sender.receiver_thread.join()

        role = self.settings.get('role')
        if role == 'aggregator':
            if isinstance(self.comms[0], Aggregator):
                with self.comms[0]:
                    self.comms[0].stop_task()


stop_receiver = False


class RabbitMQReceiver(FLReceiver):

    def __init__(self, router, comms):
        """
        Initializes server instance with reference to Router

        :param comms: A list of RabbitMQ connection objects
        :type comms: `lst`
        :param router: Router object describing the routes for each request \
            which are passed down to PH
        :type router: `Router`
        """
        self.comms = comms
        self.router = router

    @staticmethod
    def set_stopable():
        global stop_receiver
        stop_receiver = True

    def initialize(self):
        pass

    def start(self):
        """
        Start a new thread for the party
        """
        serializer = SerializerFactory(SerializerTypes.JSON_PICKLE).build()

        global stop_receiver
        while not stop_receiver:
            if isinstance(self.comms[0], Aggregator):
                try:
                    with self.comms[0]:
                        response = self.comms[0].receive(1)

                    if ffl.Notification.is_participant_joined(
                            response.notification
                    ):
                        party = response.notification['participant']
                        handler, kwargs = self.router.get_handler(
                            request_path=str(MessageType.REGISTER.value)
                        )

                        handler(
                            Message(
                                MessageType.REGISTER.value,
                                sender_info=party
                            )
                        )

                        logger.info("Registration successful...")

                except Exception:
                    pass

            elif isinstance(self.comms[0], Participant):
                try:
                    with self.comms[0]:
                        response = self.comms[0].receive(1)

                    message = serializer.deserialize(
                        response.content['message'].encode()
                    )

                    request_path = str(message.message_type)
                    handler, kwargs = self.router.get_handler(
                        request_path=request_path
                    )
                    handler(message)

                except Exception:
                    pass

    def receive_message(self):
        """
        Receive function of the receiver
        """
        serializer = SerializerFactory(SerializerTypes.JSON_PICKLE).build()

        while True:
            if isinstance(self.comms[0], Aggregator):
                with self.comms[0]:
                    try:
                        response = self.comms[0].receive(1)
                        party = response.notification['participant']
                        message = serializer.deserialize(
                            response.content['message'].encode()
                        )

                        return party, message

                    except Exception:
                        pass

    def stop(self):
        """
        Stop and cleanup
        """
        pass


class RabbitMQSender(FLSender):

    def __init__(self, comms, settings):
        """
        Initializes the connection with settings object which is passed
        from Aggregator/party

        :param comms: A list of RabbitMQ connection objects
        :type comms: `lst`
        :param settings: dictionary with connection details
        :type settings: `dict`
        """
        self.comms = comms
        self.settings = settings
        self.receiver = None
        self.receiver_thread = None

    def initialize(self):
        """
        Does all the setups required for the sender.
        """
        logger.info('RabbitMQSender initialized')

    def send_message(self, destination, message):
        """
        Used for sending all the requests with a single message

        :param destination: information about the destination to which message \
        should be forwarded, not used in RabbitMQ connection
        :type destination: `string`
        :param message: message object constructed by aggregator/party
        :type message: `Message`
        :return: response object
        :rtype: `Message`
        """
        serializer = SerializerFactory(SerializerTypes.JSON_PICKLE).build()

        response = None
        if isinstance(self.comms[0], Aggregator):
            try:
                with self.comms[0]:
                    self.comms[0].send(
                        {
                            'message': serializer.serialize(message).decode()
                        }
                    )

            except Exception as exp:
                logger.exception(exp)

        elif isinstance(self.comms[0], User):
            try:
                with self.comms[0]:
                    self.comms[0].join_task(
                        task_name=self.settings.get('task_name')
                    )

            except Exception as exp:
                logger.exception(exp)

            context = ffl.Factory.context(
                'cloud',
                self.settings.get('credentials'),
                dispatch_threshold = 0
            )

            self.comms[0] = ffl.Factory.participant(
                context,
                task_name=self.settings.get('task_name')
            )

            response = Message(MessageType.REGISTER.value)
            response.set_data({'status': 'success'})

        elif isinstance(self.comms[0], Participant):
            with self.comms[0]:
                try:
                    self.comms[0].send(
                        {
                            'message': serializer.serialize(message).decode()
                        }
                    )

                except Exception as exp:
                    logger.exception(exp)

        return response

    def send_messages(self, destinations, messages):
        """
        Used for sending all the requests with a multiple messages from the
        aggregator to all parties

        :param destination: information about the destination to which message \
        should be forwarded, not used in RabbitMQ connection
        :type destination: `string`
        :param message: message object constructed by aggregator/party
        :type message: `Message`
        :return: response object
        :rtype: `Message`
        """
        serializer = SerializerFactory(SerializerTypes.JSON_PICKLE).build()

        if isinstance(self.comms[0], Aggregator):
            try:
                for i in range(len(destinations)):
                    with self.comms[0]:
                        self.comms[0].send(
                            {
                                'message': serializer.serialize(
                                    messages[i]
                                ).decode()
                            },
                            destinations[i]
                        )

            except Exception as exp:
                logger.exception(exp)

        else:
            raise NotImplementedError(
                "This function currently only supports Aggregator."
            )

    def cleanup(self):
        """
        Cleanup the Sender object and close the hanging connections if any
        """
        pass
