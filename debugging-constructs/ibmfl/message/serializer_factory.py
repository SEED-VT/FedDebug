"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Serialization factory provides a way to create a serializer
and deserializer to convert byte streams to a message and
vice versa
"""
from ibmfl.message.json_serializer import JSONSerializer
from ibmfl.message.pickle_serializer import PickleSerializer
from ibmfl.message.serializer_types import SerializerTypes


class SerializerFactory(object):
    """
    Class for a factory to serialize and deserialize
    """
    def __init__(self, serializer_type):
        """
        Creates an object of `SerializerFactory` class

        :param serializer_type: type of seriaze and deserialize
        :type serializer_type: `Enum`
        """
        self.serializer = None
        if serializer_type is SerializerTypes.PICKLE:
            self.serializer = PickleSerializer()
        elif serializer_type is SerializerTypes.JSON_PICKLE:
            self.serializer = JSONSerializer()

    def build(self):
        """
        Returns a serializer

        :param: None
        :return: serializer
        :rtype: `Serializer`
        """
        return self.serializer
