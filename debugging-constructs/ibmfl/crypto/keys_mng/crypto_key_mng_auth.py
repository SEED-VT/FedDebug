"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
from ibmfl.util.config import get_class_by_name
from ibmfl.authority.authority_message_type import CryptoMessageType
from ibmfl.message.message import Message
from ibmfl.crypto.keys_mng.crypto_key_mng_int import KeyManager
from ibmfl.crypto.crypto_enumeration import CryptoEnum
from ibmfl.crypto.crypto_exceptions import KeyManagerException

class AuthorityKeyManager(KeyManager):

    def __init__(self, config):
        """
        Initialize a authority key manager that can request the key from
        an online authority key server
        """
        if config and 'connection' not in config:
            raise KeyManagerException('connection information is not provided.')
        if config and 'authority' not in config:
            raise KeyManagerException('authority information is not provided.')
        connection_cls_ref = get_class_by_name(config['connection']['path'],
                                               config['connection']['name'])
        self.connection = connection_cls_ref({})
        self.connection.initialize_sender()
        self.authority = config['authority']
        self.requester = config.get('requester', None)
        self.requester_id = config.get('idx', -1)

    def initialize_keys(self, config):
        """
        Initialize needed keys for the crypto system according its roles.
        """
        if self.requester:
            payload = {
                'type': config['type'],
                'keys': [CryptoEnum.KEY_PUBLIC_PARAMETER.value]
            }
            if self.requester_id:
                payload['idx'] = int(self.requester_id) if isinstance(self.requester_id, str) else self.requester_id
            if self.requester == 'party':
                payload['keys'].append(CryptoEnum.KEY_PRIVATE.value)
            if config['type'] == CryptoEnum.CRYPTO_THRESHOLD_PAILLIER.value and self.requester == 'aggregator':
                payload['keys'].append(CryptoEnum.KEY_DECTYPT.value)

            resp_status, resp_msg = self.request_cryptographic_keys(payload)
            if resp_status:
                return resp_msg.get_data()['payload']
            else:
                raise KeyManagerException('error occurred while requesting keys.')
        else:
            raise KeyManagerException('error occurred as no crypto role provided')

    def request_cryptographic_keys(self, payload):
        """
        request cryptographic keys from the authority through connection

        :param payload: payload that is used to request the decryption key
        :type payload: `dict`
        """
        try:
            resp_status = False
            message = Message(CryptoMessageType.CRYPTOGRAPHIC_KEYS.value, data={'payload': payload})
            resp_msg = self.connection.sender.send_message(self.authority, message)
            if resp_msg:
                data = resp_msg.get_data()
                if 'status' in data and data['status'] == 'error':
                    logger.error('Error occurred in key request.')
                else:
                    resp_status = True
            else:
                logger.error('Invalid response from Authority server.')
        except KeyManagerException as fl_ex:
            logger.exception("Error occurred while sending request to Authority:"
                             + str(fl_ex))
            return False
        return resp_status, resp_msg
