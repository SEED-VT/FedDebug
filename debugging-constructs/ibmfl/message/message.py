"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
The Message class is the essential information passing object,
it contains information about the data sent from a node,
the id_request, and the rank associated with the node in the network
"""

__author__ = "Supriyo Chakraborty, Shalisha Witherspoon, Dean Steuer, all ARC team"


class Message(object):
    """
    Class to create message for communication between party and aggregator
    """
    # Counter to keep track of the request
    request_id = 0

    def __init__(self, message_type=None, id_request=None, data=None, sender_info=None):
        """
        Initializes an `Message` object

        :param message_type: type of message
        :type message_type: `int`
        :param id_request: id of current request - will be used to track responses
        :type id_request: `int`
        :param data: actual data payload
        :type data: `b[]`
        """
        self.message_type = message_type
        self.sender_info = sender_info
        self.data = data

        if id_request is None:
            self.id_request = Message.get_request_id()
        else:
            self.id_request = id_request

        return

    def get_header(self):
        """
        Get header information for the message

        :param: None
        :return: information about rank, id_request, and message_type
        :rtype: `dict`
        """
        return {
            'id_request': self.id_request,
            'message_type': self.message_type,
            'sender_info': self.sender_info
        }

    def set_data(self, data):
        """set data into the message

        :param data:
        :type data: `dict`

        """
        # replace this with methods from the data class
        self.data = data

    def set_header(self, header):
        """update message information using contents in header

        :param header: dictionary with message information
        :type header: `dict`
        """
        self.id_request = header['id_request']
        self.message_type = header['message_type']
        self.sender_info = header['sender_info']

    def get_data(self):
        """
        Get actual data from the message

        :param: None
        :return: data
        :rtype: `bytes`
        """
        # replace this with methods from the data class
        return self.data

    def add_sender_info(self, info):
        """Information related to source who is sending/initiating this
        message

        :param info: Sender information
        :type info: `dict`
        """
        self.sender_info = info

    def get_sender_info(self):
        """Information related to source who sent this message

        :return: info. Sender information
        :rtype: `dict`
        """
        return self.sender_info

    @staticmethod
    def get_request_id():
        Message.request_id = Message.request_id + 1
        return Message.request_id

    def __getstate__(self):
        print('called')
        msg_dict = self.__dict__.copy()

        return msg_dict

    def __setstate__(self, dict):
        self.__dict__.update(dict)


class ResponseMessage(Message):

    def __init__(self, req_msg=None, message_type=None, id_request=None, data=None):
        if req_msg and isinstance(req_msg, Message):
            super().__init__(message_type=req_msg.message_type,
                             id_request=req_msg.id_request)
        else:
            super().__init__(message_type, id_request, data)
        return
