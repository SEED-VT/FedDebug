"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import os
import base64
import string
import random
import pytz
import logging
from datetime import datetime
from ibmfl.crypto.infra.crypto_cert_imp_rsa import CryptoCertRsa
from ibmfl.crypto.infra.crypto_sym_imp_fernet import CryptoSymFernet
from ibmfl.crypto.infra.crypto_asym_imp_rsa import CryptoAsymRsa
from ibmfl.crypto.infra.crypto_he_imp_hely import CryptoHeLy
from ibmfl.crypto.crypto_library import Crypto
from ibmfl.crypto.crypto_exceptions import KeyDistributionInputException, KeyDistributionVerificationException

logger = logging.getLogger(__name__)

class CryptoKeysProtoParty:
    """
    This class implements the party side of the protocol for generating and distributing
    cryptographic keys among the parties via the aggregator.
    """

    def __init__(self, config_dst: dict, config_crypto: dict = None, crypto_sys: Crypto = None):
        """
        Initializes the party side protocol object.
        Loads the CA cert, my cert, and additional configuration information 
        for the party to implement the keys generation and distribution protocol.

        :param config_dst: Configuration for the keys distribution protocol.
        :type config: `dict`
        :param config_crypto: Configuration for the crypto system.
        :type config: `dict`
        :param crypto_sys: Crypto system object that supports homomorphic encryption.
        :type crypto_sys: `Crypto`
        """

        self.id = None
        self.crypto_cert = CryptoCertRsa(config_dst['ca_cert_file_path'], config_dst['my_cert_file_path'])
        if 'asym_key_password' in config_dst:
            asym_pass = config_dst['asym_key_password']
            if isinstance(asym_pass, str):
                asym_pass = asym_pass.encode()
            elif not isinstance(asym_pass, bytes):
                raise KeyDistributionInputException("Invalid type of asym_pass=" + str(type(asym_pass)))
        else:
            asym_pass = None
        self.crypto_asym = CryptoAsymRsa(config_dst['asym_key_file_path'], asym_pass)
        self.crypto_sym = CryptoSymFernet()
        self.crypto_he = CryptoHeLy(config_crypto)
        self.he_pb = None
        self.he_pr = None
        self.keys_id = None
        self.keys_time = None
        self.crypto_sys = crypto_sys
        if 'cp_certs_folder_path' in config_dst and isinstance(config_dst['cp_certs_folder_path'], str):
            self.cp_certs_folder_path = config_dst['cp_certs_folder_path']
            self.cp_certs_str = {}
            for filename in os.scandir(self.cp_certs_folder_path):
                if filename.is_file():
                    id = os.path.basename(filename.path).split('_')[0]
                    with open(filename.path) as f:
                        self.cp_certs_str[id] = f.read()
        else:
            self.cp_certs_folder_path = None
            self.cp_certs_str = None
        return

    def get_he_keys(self) -> tuple:
        """
        Returns the HE key pair.
        """
        return (self.he_pb, self.he_pr)

    def get_my_cert(self, info: dict) -> dict:
        """
        Returns a payload including this party's ID and certificate.

        :param info: Dictionary that includes this party's ID.
        :type info: `dict`
        :return: Returns a payload including this party's ID and certificate.
        :rtype: `dict`
        """
        self.id = info['id']
        logger.info('get_my_cert: [id: ' + str(self.id) + '] [payload: ' + str(info) + ']')
        msg = {'id': self.id, 'cert': self.crypto_cert.get_my_cert("pem")}
        return msg

    def get_keys_id(self) -> str:
        """
        Returns the keys ID.
        """
        return self.keys_id

    def get_keys_time(self) -> datetime:
        """
        Returns the creation time of the keys.
        """
        return self.keys_time

    def __verify_parties_list(self, party_certs: dict, allowed_parties_info: dict):
        """
        Verifies that the parties in the provided list of parties' certificates are included in a
        verified list of allowed parties.

        :param party_certs: List of parties' certificates.
        :type party_certs: `dict`
        :param allowed_parties_info: Information on allowed parties.
        :type allowed_parties_info: `dict`
        """

        if self.cp_certs_str is None:
            raise KeyDistributionInputException("self.cp_certs_str is None")
        # Check that the received cp_cert is one of the configured cp_certs.
        cp_id = allowed_parties_info['cp_id']
        if cp_id not in self.cp_certs_str:
            raise KeyDistributionInputException("cp_id=" + str(cp_id) + " not in self.cp_certs_str=" + str(self.cp_certs_str.keys()))
        if allowed_parties_info['cp_cert'] != self.cp_certs_str[cp_id]:
            raise KeyDistributionInputException("allowed_parties_info['cp_cert']=" + str(allowed_parties_info['cp_cert']) + 
                " != self.cp_certs_str[cp_id]=" + str(self.cp_certs_str[cp_id]))
        # Verify the cp_cert.
        verf, cp_pbk = self.crypto_cert.verify_cert_signature(self.cp_certs_str[cp_id].encode())
        if not verf:
            raise KeyDistributionVerificationException("Invalid certificate=" + str(self.cp_certs_str[cp_id]))
        # Verify the cp signature on the list of IDs.
        signature = base64.b64decode(allowed_parties_info['cp_signature'].encode())
        verf = CryptoAsymRsa.verify_signature(cp_pbk, signature, allowed_parties_info['ids'].encode())
        if not verf:
            raise KeyDistributionVerificationException("Invalid signature=" + str(allowed_parties_info['cp_signature']))
        # Verify that all the received IDs are included in the list of allowed IDs.
        allowed_pids = allowed_parties_info['ids'].split(',')
        for pid in party_certs.keys():
            if pid not in allowed_pids:
                raise KeyDistributionInputException("pid=" + str(pid) + " not in allowed_pids=" + str(allowed_parties_info['ids']))
        return

    def generate_keys(self, info: dict) -> dict:
        """
        Generates keys and a payload of a keys distribution message.
        Outgoing message structure:
        {my_cert|he_pb|he_pb_sign|he_pr_enc|he_pr_enc_sign|keys_id|keys_time|{id->(sym_enc|sym_enc_sign)}}

        :param info: Dictionary containing party_certs (certificates of parties) and allowed_parties_info.
        :type info: `dict`
        :return: Payload of the generated keys distribution message.
        :rtype: `dict`
        """

        logger.info('generate_keys: [id: ' + str(self.id) + '] start')
        logger.debug('generate_keys: [id: ' + str(self.id) + '] [payload: ' + str(info) + ']')

        # Verify input.
        if 'party_certs' not in info or info['party_certs'] is None:
            raise KeyDistributionInputException("party_certs not in info or info['party_certs'] is None")
        party_certs = info['party_certs']

        # Verify the parties list.
        if 'allowed_parties_info' in info and info['allowed_parties_info'] is not None:
            self.__verify_parties_list(party_certs, info['allowed_parties_info'])

        # Verify the party certificates and extract their public keys.
        party_certs.pop(self.id, None)
        party_keys = self.crypto_cert.verify_certs(party_certs)

        # Generate keys and meta properties.
        self.__generate_keys_elements()
        logger.info('generate_keys: [id: ' + str(self.id) + '] generated keys')

        # Set the HE keys in the crypto system object.
        if self.crypto_sys is not None:
            keys = {'pp': self.he_pb, 'sk': self.he_pr}
            self.crypto_sys.set_keys(keys)

        # Generate a distribution message.
        msg = self.__generate_distribution_message(party_keys)

        logger.info('generate_keys: [id: ' + str(self.id) + '] [len(party_keys): ' + str(len(party_keys)) + '] end')
        return msg

    def distribute_keys(self, info: dict) -> dict:
        """
        Generates a keys distribution message to distribute the existing keys.
        Outgoing message structure:
        {my_cert|he_pb|he_pb_sign|he_pr_enc|he_pr_enc_sign|keys_id|keys_time|{id->(sym_enc|sym_enc_sign)}}

        :param info: Dictionary containing party_certs (certificates of parties) and allowed_parties_info.
        :type info: `dict`
        :return: Payload of the keys distribution message.
        :rtype: `dict`
        """

        logger.info('distribute_keys: [id: ' + str(self.id) + '] start')
        logger.debug('distribute_keys: [id: ' + str(self.id) + '] [payload: ' + str(info) + ']')

        # Verify input.
        if 'party_certs' not in info or info['party_certs'] is None:
            raise KeyDistributionInputException("party_certs not in info or info['party_certs'] is None")
        party_certs = info['party_certs']

        # Verify the parties list.
        if 'allowed_parties_info' in info and info['allowed_parties_info'] is not None:
            self.__verify_parties_list(party_certs, info['allowed_parties_info'])

        # Verify the party certificates and extract their public keys.
        party_certs.pop(self.id, None)
        if len(party_certs) > 0:
            party_keys = self.crypto_cert.verify_certs(party_certs)
        else:
            party_keys = None

        # Generate a distribution message.
        msg = self.__generate_distribution_message(party_keys)

        logger.info('distribute_keys: [id: ' + str(self.id) + '] [len(party_keys): ' + str(len(party_keys)) + '] end')
        return msg

    def __generate_distribution_message(self, party_keys:dict) -> dict:
        """
        Generates a keys distribution message in a dictionary form.

        :return: Keys distribution message in a dictionary form.
        :rtype: `dict`
        """

        msg = {}

        # Store meta properties in the message.
        msg['gen_cert'] = self.crypto_cert.get_my_cert("pem")
        msg['keys_id'] = self.keys_id
        msg['keys_time'] = self.keys_time

        # Store public keys in the message.
        msg['he_pb'] = self.he_pb
        msg['he_pb_sign'] = self.crypto_asym.get_signature(self.he_pb)

        # Store secret keys in the message.
        if party_keys is not None and len(party_keys) > 0:

            # Encrypted HE private key.
            he_pr_enc = self.crypto_sym.encrypt(self.he_pr)
            msg['he_pr_enc'] = he_pr_enc
            msg['he_pr_enc_sign'] = self.crypto_asym.get_signature(he_pr_enc)
    
            # Dict of sym key encrypted per party.
            party_sym = {}
            sym_bytes = self.crypto_sym.get_key()
            for id, pr_pbk in party_keys.items():
                sym_enc = CryptoAsymRsa.encrypt_wkey(pr_pbk, sym_bytes)
                sym_enc_sign = self.crypto_asym.get_signature(sym_enc)
                party_sym[id] = (sym_enc, sym_enc_sign)
            msg['party_sym'] = party_sym

        return msg

    def __generate_keys_elements(self):
        """
        Generates the various keys and meta properties and stores these in the object.
        """

        self.crypto_sym.generate_key()
        self.crypto_he.generate_keys()
        self.he_pb = self.crypto_he.get_public_key()
        self.he_pr = self.crypto_he.get_private_key()
        self.keys_id = generate_random_string(64)
        self.keys_time = datetime.now(pytz.timezone('America/New_York'))
        return

    def parse_keys(self, keys_msg: dict) -> bool:
        """
        Parses a keys distribution message.
        Incoming message structure:
        {gen_cert|he_pb|he_pb_sign|he_pr_enc|he_pr_enc_sign|sym_enc|sym_enc_sign|keys_id|keys_time}

        :param keys_msg: Keys distribution message payload.
        :type keys_msg: `dict`
        :return: Indication if the keys were parsed and set successfully.
        :rtype: `bool`
        """

        logger.info('parse_keys: [id: ' + str(self.id) + '] start')
        logger.debug('parse_keys: [id: ' + str(self.id) + '] [payload: ' + str(keys_msg) + ']')

        # Verify the certificate of the generating party.
        ver, gen_pbk = self.crypto_cert.verify_cert_signature(keys_msg['gen_cert'])
        if not ver:
            raise KeyDistributionVerificationException("Invalid generating party certificate=" + str(keys_msg['gen_cert']))

        # Parse the sym key.
        sym_enc = keys_msg['sym_enc']
        ver = CryptoAsymRsa.verify_signature(gen_pbk, keys_msg['sym_enc_sign'], sym_enc)
        if not ver:
            raise KeyDistributionVerificationException("Invalid signature on sym_enc=" + str(keys_msg['sym_enc_sign']))
        sym_bytes = self.crypto_asym.decrypt(sym_enc)
        self.crypto_sym = CryptoSymFernet(sym_bytes)

        # Parse the he public key.
        self.he_pb = keys_msg['he_pb']
        ver = CryptoAsymRsa.verify_signature(gen_pbk, keys_msg['he_pb_sign'], self.he_pb)
        if not ver:
            raise KeyDistributionVerificationException("Invalid signature on he_pb=" + str(keys_msg['he_pb_sign']))

        # Parse the he private key.
        he_pr_enc = keys_msg['he_pr_enc']
        ver = CryptoAsymRsa.verify_signature(gen_pbk, keys_msg['he_pr_enc_sign'], he_pr_enc)
        if not ver:
            raise KeyDistributionVerificationException("Invalid signature on he_pr_enc=" + str(keys_msg['he_pr_enc_sign']))
        self.he_pr = self.crypto_sym.decrypt(he_pr_enc)

        # Parse the keys ID and time.
        self.keys_id = keys_msg['keys_id']
        self.keys_time = keys_msg['keys_time']

        # Set the HE keys in the crypto system object.
        if self.crypto_sys is not None:
            keys = {'pp': self.he_pb, 'sk': self.he_pr}
            self.crypto_sys.set_keys(keys)

        logger.info('parse_keys: [id: ' + str(self.id) + '] end')
        return True

def generate_random_string(length) -> str:
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str
