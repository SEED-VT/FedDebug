"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Module which will control overall Federated
Learning algorithm such as plain FL, homomorphic encryption,
Hybrid-one etc.
"""
import logging
import threading
import abc
import time
import uuid
import math
import random

from multiprocessing.pool import ThreadPool

from ibmfl.aggregator.party_connection import PartyConnection
from ibmfl.aggregator.states import States
from ibmfl.aggregator.fusion.fusion_state_service import States as FusionStates
from ibmfl.message.message import Message
from ibmfl.message.message_type import MessageType
from ibmfl.exceptions import GlobalTrainingException, QuorumException, FLException

logger = logging.getLogger(__name__)


lock = threading.RLock()

class ProtoHandler(abc.ABC):
    """
    Base class for ProtoHandler (global federated learning algorithm)
    """

    def __init__(self, connection, synch=False, max_timeout=None, **kwargs):
        """
        Initializes an `ProtoHandler` object

        :param connection: connection that will be used to send messages
        :type connection: `Connection`
        :param synch: get model update synchronously
        :type synch: `boolean`
        :param max_timeout: time in seconds to wait for parties
        :type max_timeout: `int`
        """
        # List of parties involved in Federated Learning
        self.parties_list = dict()
        self.active_party_id = None
        self.dropped_parties_list = []
        self.connection = connection
        self.synch = synch
        self.state = States.START
        self.max_timeout = max_timeout
        self.num_parties = 0
        logger.info("State: " + str(self.state))

    def check_if_duplicate_party(self, party_id):
        """
        Checks if this party has already registered and if yes remove it
        :param party_id: party id to identify the details
        :type party_id: `tuple`
        :param party: party details
        :type party: `PartyConnection`
        :return: None
        """
        with lock:
            for dup_party_id in self.parties_list:
                if party_id == dup_party_id:
                    self.parties_list.pop(dup_party_id)
                    self.num_parties = self.num_parties - 1
                    break

    def check_if_dropped_party(self, party_id=None, party=None):
        """
        Checks if this party is a dropped out party and remove it
        :param party_id: party id to identify the details
        :type party_id: `tuple`
        :param party: party details
        :type party: `PartyConnection`
        :return: None
        """
        if party_id:
            with lock:
                for drop_party in self.dropped_parties_list:

                    # droppped_parties_list is list of PartyConnection
                    if isinstance(drop_party.info, str):
                        drop_party_id = drop_party.info
                    elif {'id' in drop_party.info.keys()
                          and drop_party.info['id'] is not None}:
                        drop_party_id = str(drop_party.info['id'])
                    else:
                        logger.exception("Error occurred while adding a party")

                    if party_id == drop_party_id:
                        logger.info("Cleaning dropped parties for addition of "+str(party_id))
                        self.dropped_parties_list.remove(drop_party)
                        self.parties_list.pop(drop_party_id)
                        self.num_parties = self.num_parties - 1
        elif party:
            with lock:
                for drop_party in self.dropped_parties_list:
                    if str(party.info) == str(drop_party.info):
                        self.dropped_parties_list.remove(drop_party)
                        self.parties_list.pop(drop_party)
                        self.num_parties = self.num_parties - 1
        if party is None and party_id is None:
            logger.exception("Error occurred while adding a party")

    def add_party(self, party):
        """
        Add a data party to federated learning process

        :param party: party details that are needed by the protocol handler
        :type party: `PartyConnection`
        :return guid assigned to the party
        :rtype `uuid`
        """
        if isinstance(party.info, str):
            id = party.info
            self.check_if_dropped_party(party_id=id, party=None)
            self.check_if_duplicate_party(id)

        elif 'id' in party.info.keys() and party.info['id'] is not None:
            id = str(party.info['id'])
            self.check_if_dropped_party(party_id=id, party=None)
            self.check_if_duplicate_party(id)
        else:
            random.seed(str(sorted(party.info.items())))
            id = str(uuid.UUID(bytes=bytes(random.getrandbits(8)
                                           for _ in range(16)), version=4))
            self.check_if_dropped_party(party_id=None, party=party)
            self.check_if_duplicate_party(id)

        self.parties_list[id] = party
        self.num_parties = self.num_parties + 1
        logger.info("Adding party with id " + id)
        logger.info("Total number of registered parties:" +
                    str(self.num_parties))
        return id

    def send_different_message_concurrently(self, party_ids,
                                            messages):
        """
        Send a message to list of parties asynchronously

        :param party_ids: list of parties to query
        :type party_ids: `list`
        :param messages: List of Messages to be sent to party
        :type messages: `list`
        :return: Response message status
        :rtype: `boolean`
        """
        results = []
        self.state = States.SND_REQ
        logger.info("State: " + str(self.state))

        if len(party_ids) == 0:
            logger.info('No data party registered!')
            return
        if len(party_ids) != len(messages):
            logger.error('number of parties and messages are not equal.')
            return
        pool = ThreadPool(processes=len(party_ids))
        for id, message in zip(party_ids, messages):
            # Query each party
            results.append(pool.apply_async(
                self.send_message, (id, message)))
        pool.close()
        pool.join()
        results = [r.get() for r in results]
        logger.info('Total number of success responses :{}'.format(
            sum(results)))
        logger.info("Number of parties queried:" +
                    str(len(party_ids)))
        logger.info("Number of registered parties:" +
                    str(self.num_parties))

    def add_dropped_party(self, party):
        """
        Add a party to dropped party list

        :param party: party to query
        :type party: `PartyConnection`
        :return: None
        """
        # Populate dropped party list to make sure we exclude this party
        if party not in self.dropped_parties_list:
            with lock:
                self.dropped_parties_list.append(party)

    def send_message(self, party_id, message):
        """
        Send a message to party

        :param party_id: party to query
        :type party_id: `PartyConnection`
        :param message: Message to be sent to party
        :type message: `Message`
        :return: Response message status
        :rtype: `boolean`
        """
        try:
            res_status = False
            party = self.get_party_by_id(party_id)
            res_msg = self.connection.send_message(party.info, message)

            if res_msg:
                data = res_msg.get_data()

                if 'status' in data and data['status'] == 'error':
                    logger.error('Error occurred in party.')
                if self.synch:
                    party.add_new_reply(res_msg.get_header()[
                                        'id_request'], data)
                    res_status = True
                else:
                    # This check is here for test cases that uses connection stub
                    if 'ACK' not in data and 'model_update' in data:
                        party.add_new_reply(res_msg.get_header()['id_request'],
                                            data)
                    else:
                        party.add_new_reply(message.get_header()['id_request'],
                                            data)
                    res_status = True
            else:
                logger.error('Invalid response from party.')
        except Exception as fl_ex:
            self.add_dropped_party(party)
            logger.warning("Party dropped out: Error occurred while sending request to party: "
                             + str(party) + " - " + str(fl_ex))
            return False

        return res_status

    def query_parties_with_same_payload(self, party_ids, data,
                                        msg_type=MessageType.TRAIN):
        """
        Query a list of parties by constructing a message with MessageType
        default as `TRAIN` and packaging the data as payload.

        :param party_ids: List of parties selected for an epoch
        :type party_ids: `lst`
        :param data: query information which is sent to each party
        :type data: `dict`
        :param msg_type: The type of message the query should belong in. \
        The default type is TRAIN, which allows the router to direct \
        this type of queries to the `train` method defined \
        inside the `LocalTrainingHandler` class. \
        See `MessageType` class for other possible messages types.
        :type: msg_type: `MessageType`
        :return: message number
        :rtype: `int`
        """


        try:
            message = Message(msg_type.value, data={'payload': data})
        except Exception as ex:
            logger.exception(ex)
            raise FLException("Error occurred when constructing message.")

        id_request = message.get_header()['id_request']

        try:
            messages = [message for _ in range(len(party_ids))]
            self.send_different_message_concurrently(party_ids, messages)
        except Exception as fl_ex:
            logger.exception('Error occurred while sending {} request '
                             'to parties'.format(str(msg_type)) + str(fl_ex))
            raise FLException('Error occurred while sending {} request '
                              'to parties'.format(str(msg_type)))
        return id_request

    def query_parties_with_different_payloads(self, party_ids, data_queries,
                                              msg_type=MessageType.TRAIN):
        """
        Query a list of parties by constructing a message with MessageType
        as `TRAIN` and packaging the data as payload

        :param party_ids: List of parties selected for an epoch
        :type party_ids: `lst`
        :param data_queries: query information which is sent to each party
        :type data_queries: `lst`
        :param msg_type: The type of message the query should belong in. \
        The default type is TRAIN, which allows the router to direct \
        this type of queries to the `train` method defined \
        inside the `LocalTrainingHandler` class. \
        See `MessageType` class for other possible messages types.
        :type: msg_type: `MessageType`
        :return: list of message numbers
        :rtype: `list`
        """
        messages = []
        id_request_lst = []

        if len(party_ids) != len(data_queries):
            logger.error('Number of parties and messages are not equal.')
            raise GlobalTrainingException(
                'Number of parties and messages are not equal.')
        for data in data_queries:
            try:
                message = Message(msg_type.value, data={'payload': data})
            except Exception as ex:
                logger.exception(ex)
                raise FLException(
                    "Error occurred when constructing message.")
            id_request = message.get_header()['id_request']
            messages.append(message)
            id_request_lst.append(id_request)

        try:
            self.send_different_message_concurrently(party_ids, messages)
        except Exception as fl_ex:
            logger.exception('Error occurred while sending {} request '
                             'to parties'.format(str(msg_type)) + str(fl_ex))
            raise FLException('Error occurred while sending {} request '
                              'to parties'.format(str(msg_type)))
        return id_request_lst

    def query_parties(self, payload, lst_parties, perc_quorum=1.,
                      msg_type=MessageType.TRAIN,
                      collect_metrics=False, metrics_party={},
                      fusion_state=None, return_responding_parties=False
                      ):
        """
        Sending queries to the corresponding list of parties.
        The query contents is provided in `payload`.
        The corresponding recipients are provided in `lst_parties`.

        :param payload: Content of a query or contents of multiple queries
        :type payload: `dict` if a single query content will be sent
        to `lst_parties` or `list` if multiple queries will be sent to
        the corresponding parties specifying by `lst_parties`.
        :param lst_parties: List of parties to receive the query.
        Each entry of the list should be of type `PartyConnection`, and
        the length of the `lst_parties` should match the length of `payload`
        if multiple queries will be sent.
        :type lst_parties: `list`
        :param perc_quorum: A float to specify percentage of parties that
        are needed for quorum to reach.
        :type perc_quorum: `float`
        :param msg_type: The type of message the query should belong in.
        See `MessageType` class for other possible messages types.
        :type: msg_type: `MessageType`
        :param collect_metrics: A flag to indicate if metrics
        will be collected along with query replies.
        :type collect_metrics: `boolean`
        :param metrics_party: A dictionary contains metrics collected from
        parties if `collect_metrics` is True;
         otherwise, it will be set to None.
        :type metrics_party: `dict`
        :param return_responding_parties: A flag to indicate if the list of parties ids
        corresponding to the model updates should be returned or not
        :type return_responding_parties: `boolean`
        :return:lst_model_updates: a list of replies gathered from
        the queried parties, each entry of the list should be
        of type `ModelUpdate`.
        :rtype: `list`
        """
        if lst_parties is None:
            raise FLException('No recipient is provided for the query.')

        lst_model_updates = []
        lst_responding_parties = []
        if not collect_metrics:
            metrics_party = None
        if fusion_state is not None and MessageType.TRAIN == msg_type:
            fusion_state.save_state(FusionStates.SND_MODEL)

        try:
            quorum_offset = 0
            parties_to_query = []
            for party in lst_parties:
                if party in self.dropped_parties_list:
                    quorum_offset = quorum_offset + 1 
                else:
                    parties_to_query.append(party)
                    
            if isinstance(payload, dict):

                # send one payload to a list of parties
                id_request = self.query_parties_with_same_payload(
                    parties_to_query, payload, msg_type=msg_type)
                self.periodically_verify_quorum(parties_to_query,
                                                id_request=id_request,
                                                perc_quorum=perc_quorum,
                                                quorum_offset=quorum_offset)
                if fusion_state is not None and MessageType.TRAIN == msg_type:
                    fusion_state.save_state(FusionStates.RCV_MODEL)
                for p in parties_to_query:
                    party = self.get_party_by_id(p)
                    if party.get_party_response(id_request):
                        lst_model_updates.append(
                            party.get_party_response(id_request))
                        party.del_party_response(id_request)
                        lst_responding_parties.append(p)
                    if collect_metrics:
                        if party.get_party_metrics(id_request):
                            metrics_party[str(p)] = party.get_party_metrics(
                                id_request)
                            party.del_party_metrics(id_request)
            elif isinstance(payload, list):
                # send multiply payloads to the corresponding lst of parties
                lst_id_request = self.query_parties_with_different_payloads(
                    parties_to_query, payload, msg_type=msg_type)
                self.periodically_verify_quorum(
                    parties_to_query,
                    id_request_list=lst_id_request,
                    perc_quorum=perc_quorum,
                    quorum_offset=quorum_offset)
                if fusion_state is not None and MessageType.TRAIN == msg_type:
                    fusion_state.save_state(FusionStates.RCV_MODEL)
                for p in parties_to_query:
                    party = self.get_party_by_id(p)
                    if party.get_party_response(
                            lst_id_request[parties_to_query.index(p)]):
                        lst_model_updates.append(party.get_party_response(
                            lst_id_request[parties_to_query.index(p)]))
                        party.del_party_response(
                            lst_id_request[parties_to_query.index(p)])
                        lst_responding_parties.append(p)
                    if collect_metrics:
                        if party.get_party_metrics(
                                lst_id_request[parties_to_query.index(p)]):
                            metrics_party[str(p)] = party.get_party_metrics(
                                lst_id_request[parties_to_query.index(p)])
                            party.del_party_metrics(
                                lst_id_request[parties_to_query.index(p)])
            else:
                raise FLException('Message content is not in the correct '
                                  'format. Message content should be in the '
                                  'type of dictionary or '
                                  'list of dictionaries. Instead it is '
                                  + str(type(payload)))
        except QuorumException:
            raise
        except Exception as ex:
            logger.exception(str(ex))
            logger.info('Error occurred when sending queries to parties.')
            raise GlobalTrainingException(
                'Error occurred while sending requests to parties')
        if fusion_state is not None and MessageType.TRAIN == msg_type:
            fusion_state.save_state(FusionStates.AGGREGATING)
        
        if return_responding_parties:
            return lst_model_updates, lst_responding_parties
        return lst_model_updates

    def sync_model_parties(self, party_ids, data):
        """
        Send global model to a list of parties by constructinga message
        with MessageType as `SYNC` and packaging the model as payload

        :param party_ids: List of parties selected for an epoch
        :type party_ids: `lst`
        :param data: query information which is sent to each party
        :type data: `dict`
        :return: message number
        :rtype: `int`
        """
        message = Message(MessageType.SYNC_MODEL.value, data={'payload': data})
        id_request = message.get_header()['id_request']

        try:
            messages = [message for _ in range(len(party_ids))]

            self.send_different_message_concurrently(party_ids, messages)
        except Exception as fl_ex:
            logger.exception(
                'Error occurred while sending SYNC_MODEL request to parties'
                + str(fl_ex))
            raise FLException(
                'Error occurred while sending SYNC_MODEL request to parties')
        return id_request

    def save_model_parties(self, party_ids, data):
        """
        Send save model request to parties by constructing a message with
        MessageType as `SAVE_MODEL` and packaging the data as payload

        :param party_ids: List of parties selected for an epoch
        :type party_ids: `lst`
        :param data: query information which is sent to each party
        :type data: `dict`
        :return: message number
        :rtype: `int`
        """
        message = Message(MessageType.SAVE_MODEL.value, data={'payload': data})
        id_request = message.get_header()['id_request']

        try:
            messages = [message for _ in range(len(party_ids))]

            self.send_different_message_concurrently(party_ids, messages)
        except Exception as fl_ex:
            logger.exception(
                'Error occurred while sending SAVE request to parties'
                + str(fl_ex))
            raise FLException(
                'Error occurred while sending SAVE request to parties')
        return id_request

    def eval_model_parties(self, party_ids, data):
        """
        Send eval request to parties by constructing a message with
        MessageType as `EVAL_MODEL` and packaging the data as payload

        :param party_ids: List of parties selected for an epoch
        :type party_ids: `list`
        :param data: query information which is sent to each party
        :type data: `dict`
        :return: message number
        :rtype: `int`
        """
        message = Message(MessageType.EVAL_MODEL.value,
                          data={'payload': data})
        id_request = message.get_header()['id_request']

        try:
            messages = [message for _ in range(len(party_ids))]

            self.send_different_message_concurrently(party_ids, messages)
        except FLException as fl_ex:
            logger.exception(
                'Error occurred while sending request to parties' + str(fl_ex))
            # raise Exception
        return id_request

    def stop_parties(self):
        """
        STOP all available parties.

        :return: message number
        :rtype: `int`
        """
        party_ids = self.get_available_parties()

        message = Message(MessageType.STOP.value, None)
        id_request = message.get_header()['id_request']

        try:
            messages = [message for _ in range(len(party_ids))]

            self.send_different_message_concurrently(party_ids, messages)
        except Exception as fl_ex:
            logger.exception(
                'Error occurred while sending STOP request to parties'
                + str(fl_ex))
        return id_request

    def get_registered_parties(self):
        """
        Returns all registered parties.

        :return: List of registered parties
        :rtype: `list` of `PartyConnection`
        """
        # Get available parties
        available_parties = self.get_available_parties()

        # dropped_parties_list may contain parties that are not in lst_parties
        return self.dropped_parties_list + available_parties 

    def get_available_parties(self):
        """
        Returns all registered parties that are available

        :return: List of available parties
        :rtype: `list` of `PartyConnection`
        """
        lst_parties = list(self.parties_list.keys())
        if not lst_parties or len(lst_parties) == 0:
            raise GlobalTrainingException(
                'Insufficient number of parties to start training.')

        available_parties = [party_id for party_id in lst_parties
                             if self.get_party_by_id(party_id)
                             not in self.dropped_parties_list]

        return available_parties

    def get_party_by_id(self, id):
        """
        Get `Party_Connection` object using id.

        :param id: dictionary with information about source
        :type id: `uuid`
        :return: party
        :rtype: `PartyConnection`
        """
        if id in self.parties_list:
            return self.parties_list[id]
        else:
            logger.error('Unknown party id '+str(id))
            raise GlobalTrainingException(
                'Unknown party id')

    def get_party_by_info(self, info):
        """
        Get `Party_Connection` object using info received in message.

        :param info: Dictionary with information about source
        :type info: `dict`
        :return: party
        :rtype: `PartyConnection`
        """
        for party in self.parties_list.values():
            if info == party.info:
                return party

    def get_parties_ids_by_cond(self, cond):
        res = []
        for id, party in self.parties_list.items():
            if cond.apply(party):
                res.append(id)
        return res

    def get_parties_by_cond(self, cond):
        res = {}
        for id, party in self.parties_list.items():
            if cond.apply(party):
                res[id] = party
        return res

    def get_n_parties(self):
        """
        Return the number of parties as specified in the aggregator config
        file, which determines the legnth of parties_list

        :return: Number of parties that the aggregator expects to connect to
        :rtype: `int`
        """
        return len(self.parties_list)

    def register_party(self, message):
        """
        Register data party.

        :param message: Request received by the server
        :type message: `Message`
        :return: Message with success response
        :rtype: `Message`
        """
        message_info = message.get_sender_info()
        new_party = PartyConnection(message_info)
        # TODO: check for duplicates and raise exception/send error response
        # TODO: handle invalid requests
        id = self.add_party(new_party)

        # for vertical FL setting, add active party symbol
        if isinstance(message.get_data(), dict) and \
                'is_active' in message.get_data() and \
                message.get_data()['is_active']:
            self.active_party_id = id
            logger.info('active party is registered, assigned id {}'.format(id))
        # end setting for vertical FL

        data = {'status': 'success', 'id': id}
        message.set_data(data)

        return message

    def process_model_update_requests(self, message):
        """
        Save model update request send from party

        :param message: request send by party
        :type message: `Message`
        :return: Message with appropriate response
        :rtype: `Message`
        """

        data = message.get_data()
        header = message.get_header()
        info = header['sender_info']
        id_request = header['id_request']

        party = self.get_party_by_info(info)
        party.add_new_reply(id_request, data)

        data = {'status': 'success'}
        message.set_data(data)
        return message

    def quorum_failed_party(self, party_ids,
                            id_request=None,
                            id_request_list=None):
        """
        Verifies the parties that failed during quorum

        :param party_ids: List of parties that have received query
        :type party_ids: `list`
        :param id_request: Current id_request
        :type id_request: `int`
        :param id_request_list: A list of id number link to the query that \
        needs quorum verification
        :type id_request_list: `list`
        :return: None
        """
        if id_request is None and id_request_list is None:
            logger.error('No request id provided.')
            raise GlobalTrainingException(
                'No request id provided.')
        elif id_request and id_request_list:
            raise GlobalTrainingException(
                'Both id_request and id_request_list cannot be provided at once.')
        if id_request:
            for p in party_ids:
                if not self.get_party_by_id(p).has_party_replied(id_request):
                    self.add_dropped_party(p)
        else:
            for p, p_id in zip(party_ids, id_request_list):
                if not self.get_party_by_id(p).has_party_replied(p_id):
                    self.add_dropped_party(p)

    def has_reached_quorum(self, party_ids, id_request=None,
                           id_request_list=None, perc_quorum=1., quorum_offset=0):
        """
        Verifies if quorum has been reached. If it has, it returns True,
        otherwise it returns False.

        :param party_ids: List of parties that have received query
        :type party_ids: `list`
        :param id_request: current id_request
        :type id_request: `int`
        :param id_request_list: A list of id number link to the query that \
        needs quorum verification
        :type id_request_list: `list`
        :param perc_quorum: A float to specify percentage of parties that \
        are needed for quorum to reach
        :type perc_quorum: `float`
        :param quorum_offset: Number of parties that dropped out \
        so we can calculate total registered parties
        :type quorum_offset: `int`
        :return: Quorum status
        :rtype: `boolean`
        """
        if id_request is None and id_request_list is None:
            logger.error('No request id provided.')
            raise GlobalTrainingException(
                'No request id provided.')
        elif id_request and id_request_list:
            raise GlobalTrainingException(
                'Both id_request and id_request_list cannot be provided '
                'at once. ')
        quorum = 0
        target_quorum = int(math.ceil((len(party_ids) + quorum_offset) * perc_quorum))

        if id_request:
            for p in party_ids:
                if self.get_party_by_id(p).has_party_replied(id_request):
                    quorum += 1
            if quorum >= target_quorum:
                return True
            return False
        else:
            for p, p_id in zip(party_ids, id_request_list):
                if self.get_party_by_id(p).has_party_replied(p_id):
                    quorum += 1
            if quorum >= target_quorum:
                return True
            return False

    def periodically_verify_quorum(self, party_ids, id_request=None,
                                   id_request_list=None, perc_quorum=1., quorum_offset=0):
        """
        Periodically verifies if enough replies from parties in party_ids
        for the current query identified by `id_request` have been received.
        If it has, returns True, and if it doesn't in a maximum pre-defined
        time, it throws an exception

        :param party_ids: List of parties selected for an epoch
        :type party_ids: `list`
        :param id_request: An id number link to the query that \
        needs quorum verification
        :param id_request_list: A list of id number link to the query that \
        needs quorum verification
        :type id_request_list: `list`
        :param perc_quorum: A float to specify percentage of parties that \
        are needed for quorum to reach
        :type perc_quorum: `float`
        :param quorum_offset: Number of parties that dropped out \
        so we can calculate total registered parties
        :type quorum_offset: `int`
        :return: boolean indicating if quorum has been reached
        :rtype: `boolean`
        """
        self.state = States.QUORUM_WAIT
        logger.info("State: " + str(self.state))
        if id_request is None and id_request_list is None:
            logger.error('No request id provided.')
            raise GlobalTrainingException(
                'No request id provided.')
        start = time.time()
        if id_request is not None and id_request_list is None:
            while not self.has_reached_quorum(party_ids, id_request,
                                              perc_quorum=1,
                                              quorum_offset=quorum_offset):
                # For now it we simply sleep for a while
                time.sleep(5)
                logger.info("Timeout:" + str(self.max_timeout)
                            + " Time spent:"
                            + str(round(time.time()-start)))
                if self.max_timeout:
                    if (round(time.time()-start) >= self.max_timeout):
                        if not self.has_reached_quorum(party_ids, id_request,
                                                perc_quorum=perc_quorum,
                                                quorum_offset=quorum_offset):
                            target_quorum = int(math.ceil((len(party_ids) + quorum_offset) * perc_quorum))
                            logger.error('Party did not reply in time [Registered=' + repr(len(party_ids) + quorum_offset) +
                                '] [Quorum=' + repr(target_quorum) + ']')
                            logger.info("Target Quorum was: " + str(target_quorum))

                            # Update dropped out parties list
                            self.quorum_failed_party(party_ids, id_request=id_request)
                            raise QuorumException(
                                'Max-timeout: quorum not reached. Party did not reply in time. Registered=' + 
                                repr(len(party_ids) + quorum_offset) + '. Quorum=' + repr(target_quorum) + '.')
                        else:
                            return True

        elif id_request_list is not None and id_request is None:
            while not self.has_reached_quorum(party_ids,
                                              id_request_list=id_request_list,
                                              perc_quorum=perc_quorum,
                                              quorum_offset=quorum_offset):
                time.sleep(5)
                logger.info("Timeout:" + str(self.max_timeout)
                            + " Time spent:"
                            + str(round(time.time()-start)))
                if self.max_timeout:
                    # Update dropped out parties list
                    if (round(time.time()-start) >= self.max_timeout):
                        target_quorum = int(math.ceil((len(party_ids) + quorum_offset) * perc_quorum))
                        logger.error('Party did not reply in time [Registered=' + repr(len(party_ids) + quorum_offset) +
                            '] [Quorum=' + repr(target_quorum) + ']')
                        self.quorum_failed_party(party_ids, id_request_list=id_request_list)
                        raise QuorumException(
                            'Max-timeout: quorum not reached. Party did not reply in time. Registered=' + 
                            repr(len(party_ids) + quorum_offset) + '. Quorum=' + repr(target_quorum) + '.')
        self.state = States.PROC_RSP
        logger.info("State: " + str(self.state))
        return True


class ProtoHandlerRabbitMQ(ProtoHandler):
    """
    Extended class for ProtoHandler (global federated learning algorithm),
    for using with RabbitMQ connection
    """

    def __init__(self, connection, synch=False, max_timeout=None, **kwargs):
        """
        Initializes an `ProtoHandlerRabbitMQ` object

        :param connection: Connection that will be used to send messages
        :type connection: `Connection`
        :param synch: Get model update synchronously
        :type synch: `boolean`
        """
        if synch:
            super(ProtoHandlerRabbitMQ, self).__init__(
                connection,
                synch,
                max_timeout,
                **kwargs
            )

        else:
            raise FLException(
                'RabbitMQ connection currently only supports synchronous mode'
            )

    def send_message(self, message):
        """
        Send a message to all parties

        :param message: Message to be sent to parties
        :type message: `Message`
        """
        self.connection.send_message('', message)

    def send_messages(self, party_ids, messages):
        """
        Send multiple messages to multiple parties

        :param party_ids: List of parties selected for an epoch
        :type party_ids: `list`
        :param messages: Message to be sent to parties
        :type messages: `Message`
        """
        self.connection.send_messages(party_ids, messages)

    def receive_messages(self, party_ids):
        """
        Receive model updates from parties

        :param party_ids: List of parties selected for an epoch
        :type party_ids: `list`
        """
        results = {}
        id_requests = {}
        for _ in range(len(party_ids)):
            res = self.connection.receiver.receive_message()
            results[res[0]] = res[1].get_data()
            id_requests[res[0]] = res[1].get_header()['id_request']

        for id in party_ids:
            party = self.get_party_by_id(id)
            party.add_new_reply(id_requests[party.info], results[party.info])

    def query_parties_with_same_payload(self, party_ids, data, msg_type=MessageType.TRAIN):
        """
        Query a list of parties by constructing a message with MessageType
        as `TRAIN` and packaging the data as payload

        :param party_ids: List of parties selected for an epoch
        :type party_ids: `list`
        :param data: query information which is sent to each party
        :type data: `dict`
        :param msg_type: The type of message the query should belong in. \
        The default type is TRAIN, which allows the router to direct \
        this type of queries to the `train` method defined \
        inside the `LocalTrainingHandler` class. \
        See `MessageType` class for other possible messages types.
        :type msg_type: `MessageType`
        :return: message number
        :rtype: `int``
        """
        self.connection.receiver.set_stopable()
        self.connection.receiver_thread.join()

        if not isinstance(msg_type, MessageType):
            raise FLException(
                "Provided message type should be of type MessageType. "
                "Instead it is of type " + str(type(msg_type))
            )

        try:
            message = Message(msg_type.value, data={'payload': data})
        except Exception as ex:
            logger.exception(ex)
            raise FLException("Error occurred when constructing message.")

        id_request = message.get_header()['id_request']

        try:
            self.send_message(message)

        except Exception as fl_ex:
            logger.exception(
                'Error occurred while sending TRAIN request to parties'
                + str(fl_ex)
            )
            raise FLException(
                'Error occurred while sending TRAIN request to parties'
            )

        try:
            self.receive_messages(party_ids)

        except Exception as fl_ex:
            logger.exception(
                'Error occurred while receiving local TRAIN results from'
                ' parties' + str(fl_ex)
            )
            raise FLException(
                'Error occurred while receiving local TRAIN results from'
                ' parties'
            )

        return id_request

    def query_parties_with_different_payloads(self, party_ids, data_queries,
                                              msg_type=MessageType.TRAIN):
        """
        Query a list of parties by constructing a message with MessageType
        as `TRAIN` and packaging the data as payload

        :param party_ids: List of parties selected for an epoch
        :type party_ids: `list`
        :param data_queries: Query information which is sent to each party
        :type data_queries: `list`
        :param msg_type: The type of message the query should belong in. \
        The default type is TRAIN, which allows the router to direct \
        this type of queries to the `train` method defined \
        inside the `LocalTrainingHandler` class. \
        See `MessageType` class for other possible messages types.
        :type msg_type: `MessageType`
        :return: list of message numbers
        :rtype: `list`
        """
        self.connection.receiver.set_stopable()
        self.connection.receiver_thread.join()

        messages = []
        id_request_lst = []

        if len(party_ids) != len(data_queries):
            logger.error('Number of parties and messages are not equal.')
            raise GlobalTrainingException(
                'Number of parties and messages are not equal.'
            )

        for data in data_queries:
            try:
                message = Message(msg_type.value, data={'payload': data})
            except Exception as ex:
                logger.exception(ex)
                raise FLException(
                    "Error occurred when constructing message."
                )

            id_request = message.get_header()['id_request']

            messages.append(message)
            id_request_lst.append(id_request)

        try:
            self.send_messages(party_ids, messages)

        except Exception as fl_ex:
            logger.exception(
                'Error occurred while sending TRAIN request to parties'
                + str(fl_ex)
            )
            raise FLException(
                'Error occurred while sending TRAIN request to parties'
            )

        try:
            self.receive_messages(party_ids)

        except Exception as fl_ex:
            logger.exception(
                'Error occurred while receiving local TRAIN results from'
                ' parties' + str(fl_ex)
            )
            raise FLException(
                'Error occurred while receiving local TRAIN results from'
                ' parties'
            )

        return id_request_lst

    def eval_model_parties(self, party_ids, data):
        """
        Send eval request to parties by constructing a message with
        MessageType as `EVAL_MODEL` and packaging the data as payload

        :param party_ids: List of parties selected for an epoch
        :type party_ids: `list`
        :param data: query information which is sent to each party
        :type data: `dict`
        :return: message number
        :rtype: `int`
        """
        self.connection.receiver.set_stopable()
        self.connection.receiver_thread.join()

        message = Message(MessageType.EVAL_MODEL.value, data={'payload': data})
        id_request = message.get_header()['id_request']

        try:
            self.send_message(message)

        except Exception as fl_ex:
            logger.exception(
                'Error occurred while sending EVAL request to parties'
                + str(fl_ex)
            )
            raise FLException(
                'Error occurred while sending EVAL request to parties'
            )

        return id_request

    def sync_model_parties(self, party_ids, data):
        """
        Send global model to a list of parties by constructinga message
        with MessageType as `SYNC` and packaging the model as payload

        :param party_ids: List of parties selected for an epoch
        :type party_ids: `lst`
        :param data: query information which is sent to each party
        :type data: `dict`
        :return: message number
        :rtype: `int`
        """
        self.connection.receiver.set_stopable()
        self.connection.receiver_thread.join()

        message = Message(MessageType.SYNC_MODEL.value, data={'payload': data})
        id_request = message.get_header()['id_request']

        try:
            self.send_message(message)

        except Exception as fl_ex:
            logger.exception(
                'Error occurred while sending SYNC request to parties'
                + str(fl_ex)
            )
            raise FLException(
                'Error occurred while sending SYNC request to parties'
            )

        return id_request

    def save_model_parties(self, party_ids, data):
        """
        Send save model request to parties by constructing a message with
        MessageType as `SAVE_MODEL` and packaging the data as payload

        :param party_ids: List of parties selected for an epoch
        :type party_ids: `lst`
        :param data: query information which is sent to each party
        :type data: `dict`
        :return: message number
        :rtype: `int`
        """
        self.connection.receiver.set_stopable()
        self.connection.receiver_thread.join()

        message = Message(MessageType.SAVE_MODEL.value, data={'payload': data})
        id_request = message.get_header()['id_request']

        try:
            self.send_message(message)

        except Exception as fl_ex:
            logger.exception(
                'Error occurred while sending SAVE request to parties'
                + str(fl_ex)
            )
            raise FLException(
                'Error occurred while sending SAVE request to parties'
            )

        return id_request

    def stop_parties(self):
        """
        STOP all available parties.

        :return: message number
        :rtype: `int`
        """
        self.connection.receiver.set_stopable()
        self.connection.receiver_thread.join()

        party_ids = self.get_available_parties()

        message = Message(MessageType.STOP.value, None)
        id_request = message.get_header()['id_request']

        try:
            messages = [message for _ in range(len(party_ids))]

            self.send_messages(party_ids, messages)
        except Exception as fl_ex:
            logger.exception(
                'Error occurred while sending STOP request to parties'
                + str(fl_ex)
            )

        return id_request
