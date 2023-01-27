"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import abc
from enum import Enum


class FLConnection(abc.ABC):

    SENDER_STATUS = None
    RECEIVER_STATUS = None

    """
    Initialze sender and receiver

    :param kwargs: Dictionary of arguments required by each type of
        sender implementation
    """
    @abc.abstractmethod
    def initialize(self, **kwargs):
        pass

    """
    Validate config dictionary and initialize receiver

    :param kwargs: Dictionary of arguments required by each type of
        receiver implementation
    """
    @abc.abstractmethod
    def initialize_receiver(self, **kwargs):
        pass

    """
    Validate config dictionary and initialize sender

    :param kwargs: Dictionary of arguments required by each type of
        sender implementation
    """
    @abc.abstractmethod
    def initialize_sender(self, **kwargs):
        pass

    """
    Start connection
    """
    @abc.abstractmethod
    def start(self, **kwargs):
        pass

    """Stop and cleanup connection object
    """
    @abc.abstractmethod
    def stop(self, **kwargs):
        pass

    """Provide a connection information such that a node can communicate 
        to other nodes on how to communicate with it.
    """
    @abc.abstractmethod
    def get_connection_config(self):
        pass


class FLReceiver(abc.ABC):

    @abc.abstractmethod
    def initialize(self, **kwargs):
        pass

    """
    Start the server as per configuration after a clean intialize_server
    call
    :param kwargs: Dictionary of arguments required by each type of
        server implementation
    """
    @abc.abstractmethod
    def start(self, **kwargs):
        pass

    """
    Cleanup all the resources and stop serving
    :param kwargs: Dictionary of arguments required by each type of
        server implementation
    """
    @abc.abstractmethod
    def stop(self, **kwargs):
        pass


class FLSender(abc.ABC):

    @abc.abstractmethod
    def initialize(self, **kwargs):
        pass

    """
    Requests the end point as per the message forwarded to this module.
    Should have a logic to incorporate additional authentication details as
    per the configuration.

    :param kwargs: Dictionary of arguments required by each type of
        connection implementation
    """
    @abc.abstractmethod
    def send_message(self, **kwargs):
        pass

    """
    Cleanup the Sender object and close the hanging connections if any
    """
    @abc.abstractmethod
    def cleanup(self):
        pass


class ConnectionStatus(Enum):
    """Enum Class to describe the current status of the connection
    """
    INITIALIZED = 1
    STARTED = 2
    STOPPED = 3
