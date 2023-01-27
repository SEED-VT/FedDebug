"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Module to track each party at the aggregator side
"""

import datetime
from collections import OrderedDict


class PartyConnection(object):
    """
    Class for maintaining party related information
    for the `ProtoHandler`
    """

    def __init__(self, info):
        """
        Initializes an `PartyConnection` object

        """
        self.info = info
        self.creation_time = datetime.datetime.now()
        self.open_port = True
        self.connection = None
        self.open_connection = False
        self.party_last_response = None
        self.party_last_request = None
        self.state = None
        self.replies = OrderedDict()  # <id_request, reply>
        self.metrics = OrderedDict()
        self.ack_replies = []

    def add_new_reply(self, id_request, reply):
        """
        Adds the payload of the replied sent by the party in specific id_request to
        a dictionary where the key is the id_request and the reply the data.

        :param id_request: id_request for provided reply
        :type id_request: `int`
        :param reply: payload associated with the reply message
        :type reply: may change
        :return: None
        """
        # Prune if replies stores are more than 10
        if len(self.replies) >= 10:
            _ = self.replies.popitem(last=False)
        if 'ACK' not in reply and 'payload' in reply:
            if len(self.metrics) >=10:
                _ = self.metrics.popitem(last=False)
            self.replies[id_request] = reply.get('payload')
            self.metrics[id_request] = reply.get('metrics') or {}

        elif 'ACK' in reply:
            self.ack_replies.append(id_request)

    def has_party_replied(self, id_request):
        """
        Verifies if party has replied in provided id_request. If it does
        it returns True, otherwise returns false.

        :param id_request: training id_request
        :type id_request: `int`
        :return: if replied has been received
        :rtype: `boolean`
        """
        if id_request in self.replies:
            return True
        return False

    def get_party_response(self, id_request):
        """
        Returns party's response for given id_request

        :param id_request: training id_request
        :type id_request: `int`
        :return: reply from given id_request
        :rtype:
        """
        if id_request in self.replies:
            return self.replies[id_request]
        else:
            return None

    def del_party_response(self, id_request):
        """
        Removes party's response for given id_request +
        :param id_request: training id_request
        :type id_request: `int`
        :return: None
        """
        if id_request in self.replies:
            del self.replies[id_request]
        else:
         return


    def del_party_metrics(self, id_request):
        """
        Removes party's metrics for given id_request +
        :param id_request: training id_request
        :type id_request: `int`
        :return: None
        """
        if id_request in self.metrics:
            del self.metrics[id_request]
        else:
            return

    def get_party_metrics(self, id_request):
        """
        Returns party's metrics for given id_request

        :param id_request: training id_request
        :type id_request: `int`
        :return: metric reply from given id_request
        :rtype:
        """
        if id_request in self.metrics:
            return self.metrics[id_request]
        else:
            return None
