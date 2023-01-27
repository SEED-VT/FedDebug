"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import random
import logging
from datetime import datetime
from ibmfl.aggregator.protohandler.proto_handler import ProtoHandler
from ibmfl.message.message_type import MessageType
from ibmfl.crypto.crypto_library import Crypto
from ibmfl.crypto.crypto_exceptions import KeyDistributionCommunicationException
from ibmfl.exceptions import QuorumException

logger = logging.getLogger(__name__)

class CryptoKeysProtoAgg:
    """
    This class implements the aggregator side of the protocol for generating and distributing
    cryptographic keys among the parties via the aggregator.
    """

    def __init__(self, config: dict = None, 
        protocol_handler: ProtoHandler = None, crypto_sys: Crypto = None, perc_quorum: float = None):
        """
        Initializes the aggregator side protocol object.

        :param config: configuration for the keys distribution protocol.
        :type config: `dict`
        :param protocol_handler: Protocol handler used for handling requests for communication.
        :type protocol_handler: `ProtoHandler`
        :param crypto_sys: Crypto system object that supports homomorphic encryption.
        :type crypto_sys: `Crypto`
        :param perc_quorum: Percentage representing the minimum number of parties to form a quorum.
        :type perc_quorum: `int`
        """

        self.party_certs = {} # Maps party ID to cert.
        self.he_pb = None # HE public key.
        self.keys_id = None
        self.keys_time = None
        self.parties_formation_time = None
        self.ph = protocol_handler
        self.crypto_sys = crypto_sys
        self.perc_quorum = perc_quorum
        if config is not None and \
            'allowed_parties_ids' in config and isinstance(config['allowed_parties_ids'], str) and \
            'allowed_parties_cp_signature' in config and isinstance(config['allowed_parties_cp_signature'], str) and \
            'allowed_parties_cp_cert' in config and isinstance(config['allowed_parties_cp_cert'], str) and \
            'allowed_parties_cp_id' in config and isinstance(config['allowed_parties_cp_id'], str):
            self.allowed_parties_info = {}
            self.allowed_parties_info['ids'] = config['allowed_parties_ids']
            self.allowed_parties_info['cp_signature'] = config['allowed_parties_cp_signature']
            self.allowed_parties_info['cp_cert'] = config['allowed_parties_cp_cert']
            self.allowed_parties_info['cp_id'] = config['allowed_parties_cp_id']
        else:
            self.allowed_parties_info = None
        return

    def get_he_pb(self):
        """
        Returns the HE public key.
        """
        return self.he_pb

    def get_keys_id(self):
        """
        Returns the ID of the keys.
        """
        return self.keys_id
        
    def get_keys_time(self):
        """
        Returns the generation time of the keys.
        """
        return self.keys_time

    def get_party_certs(self) -> dict:
        """
        Returns the certificates dict.
        """
        return self.party_certs

    def get_allowed_parties_info(self):
        """
        Returns the configuration information on the allowed parties.
        """
        return self.allowed_parties_info

    def add_party_cert(self, msg: dict):
        """
        Adds a party record to the certificates dict.

        :param msg: Party record mapping its ID to its certificate.
        :type msg: `dict`
        """
        self.party_certs[msg['id']] = msg['cert']
        return

    def __process_party_certs(self, cert_msgs: list, available_parties_ids: list = None):
        """
        Processes a list of certificates of new parties and a list of IDs of existing available parties
        into a dictionary of new parties and a list of existing available parties which are not
        new parties. Updates the internal dictionary of certificates with the new parties certificates.

        :param cert_msgs: Incoming messages specifying the certificates on new parties.
        :type cert_msgs: `list`
        :param available_parties_ids: A list of IDs of existing available parties.
        :type available_parties_ids: `list`
        :return: Tuple that includes a dictionary of new parties and a list of existing available parties which are not new parties.
        :rtype: `tuple`
        """

        new_parties_certs = {}
        available_parties = []
        for msg in cert_msgs:
            new_parties_certs[msg['id']] = msg['cert']
        self.party_certs.update(new_parties_certs)
        if available_parties_ids is not None:
            for id in available_parties_ids:
                if id not in new_parties_certs:
                    available_parties.append(id)
        return new_parties_certs, available_parties

    def parse_keys(self, keys_msg: dict, update_self = True) -> dict:
        """
        Parses a keys distribution message coming from a generating party into
        a dictionary of distribution messages - one message per party.
        In addition, updates the internal HE public key and properties.
        Incoming message structure:
        {gen_cert|he_pb|he_pb_sign|he_pr_enc|he_pr_enc_sign|{id->(sym_enc|sym_enc_sign)}}
        Outgoing message structure:
        {id->{gen_cert|he_pb|he_pb_sign|he_pr_enc|he_pr_enc_sign|sym_enc|sym_enc_sign}}

        :param keys_msg: Keys distribution message payload.
        :type keys_msg: `dict`
        :param update_self: Indication if to update the internal key and properties.
        :type update_self: `bool`
        :return: Keys distribution message payload per party. 
        :rtype: `dict`
        """

        msg = {}

        # Parse the he public key.
        if update_self:
            self.he_pb = keys_msg['he_pb']
            self.keys_id = keys_msg['keys_id']
            self.keys_time = keys_msg['keys_time']

        # Generate a dictionary of per party payload.
        if 'party_sym' in keys_msg:
            for id, sym in keys_msg['party_sym'].items():
                payload = {}
                payload['gen_cert'] = keys_msg['gen_cert']
                payload['he_pb'] = keys_msg['he_pb']
                payload['he_pb_sign'] = keys_msg['he_pb_sign']
                payload['he_pr_enc'] = keys_msg['he_pr_enc']
                payload['he_pr_enc_sign'] = keys_msg['he_pr_enc_sign']
                payload['sym_enc'] = sym[0]
                payload['sym_enc_sign'] = sym[1]
                payload['keys_id'] = keys_msg['keys_id']
                payload['keys_time'] = keys_msg['keys_time']
                msg[id] = payload

        return msg

    def manage_keys(self) -> bool:
        """
        Generates new keys if required, and distributes the keys to the parties.
        """
        if self.keys_id is None:
            return self.__generate_and_distribute_keys()
        else:
            return self.__distribute_existing_keys()

    def __generate_and_distribute_keys(self) -> bool:
        """
        Generates and distributes new keys to the parties.
        """

        # Record time of parties formation.
        self.parties_formation_time = datetime.now()

        # Wait for parties' acknowledgement that their models are initialized.
        lst_parties = self.ph.get_registered_parties()
        logger.info('generate_and_distribute_keys: start [lst_parties: ' + str(lst_parties) + ']')
        self.__await_parties_init(lst_parties)

        # Query parties for their IDs and certificates.
        payloads = []
        for id in lst_parties:
            payloads.append({'id': id})
        logger.debug('generate_and_distribute_keys: [payloads: ' + str(payloads) + ']')
        lst_response_cert = self.ph.query_parties(payload = payloads,
            lst_parties = lst_parties, perc_quorum = 1,
            msg_type=MessageType.REQUEST_CERT)
        logger.debug('generate_and_distribute_keys: [lst_response_cert: ' + str(lst_response_cert) + ']')
        party_certs, _ = self.__process_party_certs(cert_msgs = lst_response_cert)
        logger.info('generate_and_distribute_keys: received certs from party ids ' + str(party_certs.keys()))
        #logger.debug('generate_and_distribute_keys: [party_certs: ' + str(party_certs) + ']')

        # Request one party to generate keys, and receive a keys distribution message from the party.
        payload_dict = {'party_certs': party_certs}
        allowed_parties_info = self.get_allowed_parties_info()
        if allowed_parties_info is not None:
             payload_dict['allowed_parties_info'] = allowed_parties_info
        payload = [payload_dict]
        lst_parties_avl = list(party_certs.keys())
        gen_prt_idx = random.randint(0, len(lst_parties_avl)-1)
        lst_1_party = [lst_parties_avl[gen_prt_idx]]
        #logger.debug('generate_and_distribute_keys: [lst_1_party: ' + str(lst_1_party) + '] [payload: ' + str(payload) + ']')
        response_keys = self.ph.query_parties(payload = payload,
            lst_parties = lst_1_party, perc_quorum = 1,
            msg_type=MessageType.GENERATE_KEYS)
        if len(response_keys) != 1:
            raise KeyDistributionCommunicationException('generate_and_distribute_keys: the generating party did not '
                'respond to the key generation request message')
        logger.info('generate_and_distribute_keys: party id ' + str(lst_1_party[0]) + 
            ' generated keys and distribution message')
        #logger.debug('generate_and_distribute_keys: [response_keys: ' + str(response_keys[0].keys()) + ']')

        # Parse the keys distribution message into dedicated messages per parties,
        # and set the local HE public key. 
        msgs = self.parse_keys(response_keys[0])
        logger.debug('generate_and_distribute_keys: [msgs: ' + str(msgs.keys()) + ']')
        keys = {'pp': self.get_he_pb()}
        self.crypto_sys.set_keys(keys)

        # Send the keys distribution messages to parties.
        num_parties_updated = 1
        if len(lst_parties_avl) > 1:
            payloads = []
            lst_parties_wo_gen = lst_parties_avl[:gen_prt_idx]+lst_parties_avl[gen_prt_idx+1:]
            for id in lst_parties_wo_gen:
                payloads.append(msgs[id])
            lst_response_set = self.ph.query_parties(payload = payloads,
                lst_parties = lst_parties_wo_gen, perc_quorum = 1,
                msg_type=MessageType.SET_KEYS)
            num_parties_updated += len(lst_response_set)
            logger.debug('generate_and_distribute_keys: [lst_response_set: ' + str(lst_response_set) + ']')
            #if len(lst_response_set) != len(lst_parties_wo_gen):
            #    raise KeyDistributionCommunicationException('generate_keys: some parties did not respond to the keys '
            #        'generate_keys message [len(lst_response_set)=' + str(len(lst_response_set)) + '] [len(lst_parties_wo_gen)=' +
            #        str(len(lst_parties_wo_gen)) + ']')
            for rsp in lst_response_set:
                if rsp is not True:
                    raise KeyDistributionCommunicationException('generate_and_distribute_keys: some parties failed in processing the keys '
                        'distribution message [lst_response_set: ' + str(lst_response_set) + ']')

        logger.info('generate_and_distribute_keys: end [number of parties updated: ' + str(num_parties_updated) + ']')
        return True

    def __distribute_existing_keys(self) -> bool:
        """
        Distributes existing keys to new joining parties, if such parties exist.
        """

        # Determine the new parties since the last keys generation / distribution.
        new_party_ids = self.__get_new_parties()
        if len(new_party_ids) == 0:
            return
        logger.info('distribute_existing_keys: begin [new_party_ids: ' + str(new_party_ids) + ']')

        # Record time of parties formation.
        self.parties_formation_time = datetime.now()

        # Wait for parties' acknowledgement that their models are initialized.
        self.__await_parties_init(new_party_ids)

        # Query the new parties for their IDs and certificates.
        payloads = []
        for id in new_party_ids:
            payloads.append({'id': id})
        logger.debug('distribute_existing_keys: [payloads: ' + str(payloads) + ']')
        lst_response_cert = self.ph.query_parties(payload = payloads,
            lst_parties = new_party_ids, perc_quorum = 1,
            msg_type=MessageType.REQUEST_CERT)
        #logger.debug('distribute_existing_keys: [lst_response_cert: ' + str(lst_response_cert) + ']')
        #if len(lst_response_cert) != len(new_party_ids):
        #    raise KeyDistributionCommunicationException('distribute_existing_keys: some parties did not respond to the query '
        #        'certificate message [len(lst_response_cert)=' + str(len(lst_response_cert)) + '] [len(new_party_ids)=' +
        #        str(len(new_party_ids)) + ']')
        party_certs, available_existing_parties_ids = \
            self.__process_party_certs(cert_msgs=lst_response_cert, 
                available_parties_ids=self.ph.get_available_parties())
        logger.info('distribute_existing_keys: received certs from new party ids ' + str(party_certs.keys()) +
            ' [available_existing_parties_ids: ' + str(available_existing_parties_ids) + ']')
        #logger.debug('distribute_existing_keys: [party_certs: ' + str(party_certs))

        # Request one party to provide a keys distribution message.
        payload_dict = {'party_certs': party_certs}
        allowed_parties_info = self.get_allowed_parties_info()
        if allowed_parties_info is not None:
             payload_dict['allowed_parties_info'] = allowed_parties_info
        payload = [payload_dict]
        newKeys = fallback = False
        if len(available_existing_parties_ids) > 0:
            #gen_prt_idx = random.randint(0, len(available_existing_parties_ids)-1)
            response_keys = None
            for gen_prt_id in available_existing_parties_ids:
                lst_1_party = [gen_prt_id]
                #logger.debug('distribute_existing_keys: [lst_1_party: ' + str(lst_1_party) + '] [payload: ' + str(payload) + ']')
                try:
                    response_keys = self.ph.query_parties(payload = payload,
                        lst_parties = lst_1_party, perc_quorum = 1,
                        msg_type=MessageType.DISTRIBUTE_KEYS)
                except QuorumException:
                    logger.info('distribute_existing_keys: party ' + str(gen_prt_id) + ' is not available to generate distribution message')
                    continue
                else:
                    break
            if response_keys is None or len(response_keys) != 1:
                fallback = True
                logger.info('distribute_existing_keys: no existing party responded to the '
                    'key distribution request message; generating new keys')
                #raise KeyDistributionCommunicationException('distribute_existing_keys: no party responded to the '
                #    'key distribution request message')
            else:
                logger.info('distribute_existing_keys: party id ' + str(gen_prt_id) + 
                    ' generated a distribution message')
        if len(available_existing_parties_ids) == 0 or fallback:
            newKeys = True
            gen_prt_idx = random.randint(0, len(new_party_ids)-1)
            lst_1_party = [new_party_ids[gen_prt_idx]]
            #logger.debug('distribute_existing_keys: [lst_1_party: ' + str(lst_1_party) + '] [payload: ' + str(payload) + ']')
            response_keys = self.ph.query_parties(payload = payload,
                lst_parties = lst_1_party, perc_quorum = 1,
                msg_type=MessageType.GENERATE_KEYS)
            if response_keys is None or len(response_keys) != 1:
                raise KeyDistributionCommunicationException('distribute_existing_keys: the generating party did not '
                    'respond to the key generation request message')
            logger.info('distribute_existing_keys: party id ' + str(lst_1_party[0]) + 
                ' generated keys and distribution message')
        logger.debug('distribute_existing_keys: [response_keys: ' + str(response_keys[0].keys()) + ']')

        # Parse the keys distribution message into dedicated messages per parties,
        # and if new keys were generated - set the local HE public key. 
        msgs = self.parse_keys(response_keys[0], newKeys)
        logger.debug('distribute_existing_keys: [msgs: ' + str(msgs.keys()) + ']')
        if newKeys:
            keys = {'pp': self.get_he_pb()}
            self.crypto_sys.set_keys(keys)

        # Send the keys distribution messages to the new parties.
        num_parties_updated = 1 if newKeys else 0
        if len(msgs.keys()) > 0:
            party_ids_from_msgs = []
            payloads = []
            for id, pyld in msgs.items():
                party_ids_from_msgs.append(id)
                payloads.append(pyld)
            lst_response_set = self.ph.query_parties(payload = payloads,
                lst_parties = party_ids_from_msgs, perc_quorum = 1,
                msg_type=MessageType.SET_KEYS)
            num_parties_updated += len(lst_response_set)
            logger.debug('distribute_existing_keys: [lst_response_set: ' + str(lst_response_set) + ']')
            #if len(lst_response_set) != len(party_ids_from_msgs):
            #    raise KeyDistributionCommunicationException('distribute_existing_keys: some parties did not respond to the keys '
            #        'distribution message [len(lst_response_set)=' + str(len(lst_response_set)) + '] [len(party_ids_from_msgs)=' +
            #        str(len(party_ids_from_msgs)) + ']')
            for rsp in lst_response_set:
                if rsp is not True:
                    raise KeyDistributionCommunicationException('distribute_existing_keys: some parties failed in processing the keys '
                        'distribution message [lst_response_set: ' + str(lst_response_set) + ']')

        logger.info('distribute_existing_keys: end [newKeys: ' + str(newKeys) + '] [number of parties updated: ' + 
            str(num_parties_updated) + ']')
        return newKeys

    def __await_parties_init(self, lst_parties_ids: list):
        """
        Waits for the parties to acknowledge that their models are initialized.
        """

        lst_response_init = self.ph.query_parties(payload = {'ack_init': 'True'},
            lst_parties = lst_parties_ids, perc_quorum = 1,
            msg_type=MessageType.TRAIN)
        logger.info('await_parties_init: received ack init from ' + str(len(lst_response_init)) + ' parties')
        for rsp in lst_response_init:
            if rsp is not True:
                raise KeyDistributionCommunicationException('await_parties_init: some parties failed in acknowledging '
                    'model init [lst_response_init: ' + str(lst_response_init) + ']')
        return

    class PartyCondCreationDate:
        """
        This class enables to find parties based on a creation date condition.
        """

        def __init__(self, base_date):
            self.base_date = base_date

        def apply(self, party):
            if party.creation_time > self.base_date:
                #logger.debug('PartyCondCreationDate: new party [creation_time: ' + str(party.creation_time) +
                #    '] [base: ' + str(self.base_date) + ']')
                return True
            else:
                #logger.debug('PartyCondCreationDate: existing party [creation_time: ' + str(party.creation_time) +
                #    '] [base: ' + str(self.base_date) + ']')
                return False

    def __get_new_parties(self):
        """
        Returns a list of IDs of the new parties.
        """
        new_party_ids = \
            self.ph.get_parties_ids_by_cond(CryptoKeysProtoAgg.PartyCondCreationDate(self.parties_formation_time))
        return new_party_ids
