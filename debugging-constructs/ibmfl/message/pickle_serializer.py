"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Pickle based serialization
"""
import pickle
from ibmfl.message.message import Message
from ibmfl.message.serializer import Serializer


class PickleSerializer(Serializer):
    """
    Class for Pickle based serialization
    """

    def serialize(self, msg):
        """
        Serialize a message using pickle

        :param msg: message to serialize
        :type msg: `Message`
        :return: serialize byte stream
        :rtype: `b[]`
        """
        msg_header = msg.get_header()
        serialized_data = msg.get_data()  # need to serialize the data
        return pickle.dumps({'header': msg_header,
                             'data': serialized_data,
                             })

    def deserialize(self, serialized_byte_stream):
        """
        Deserialize a byte stream to a message

        :param serialized_byte_stream: byte stream
        :type serialized_byte_stream: `b[]`
        :return: deserialized message
        :rtype: `Message`
        """
        data_dict = pickle.loads(serialized_byte_stream)
        if 'MSG_LEN|' in data_dict:
            msg_length = int(data_dict.split('|')[1])
            return msg_length

        msg = Message(data=data_dict['data'])
        msg.set_header(data_dict['header'])

        return msg
