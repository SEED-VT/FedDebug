"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
from abc import ABC, abstractmethod


class Serializer(ABC):
    """
    Abstract class for Serializer
    """
    @abstractmethod
    def serialize(self, msg):
        """
        Serialize a message

        :param msg: message to serialize
        :type msg: `Message`
        :return: serialized byte stream
        :rtype: `b[]`
        """
        pass

    @abstractmethod
    def deserialize(self, serialized_byte_stream):
        """
        Deserialize a byte stream to a message

        :param serialized_byte_stream: byte stream
        :type serialized_byte_stream: `b[]`
        :return: deserialized message
        :rtype: `Message`
        """
        pass
