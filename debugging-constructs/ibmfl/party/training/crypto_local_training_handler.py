"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging

from ibmfl.party.training.local_training_handler import LocalTrainingHandler
from ibmfl.model.model_update import ModelUpdate
from ibmfl.crypto.crypto_exceptions import CryptoException
from ibmfl.util.config import get_class_by_name
from ibmfl.crypto.keys_mng.crypto_keys_proto_party import CryptoKeysProtoParty

logger = logging.getLogger(__name__)

class CryptoLocalTrainingHandler(LocalTrainingHandler):

    def __init__(self, fl_model, data_handler, hyperparams=None, info=None, **kwargs):
        """
        Initialize LocalTrainingHandler with fl_model, data_handler
        :param fl_model: model to be trained
        :type fl_model: `model.FLModel`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param hyperparams: Hyperparameters used for training.
        :type hyperparams: `dict`
        :param info: A dictionary containing crypto configurations about \
        a crypto library object to help with encryption and decryption.
        :type info: `dict`
        :return None
        """
        super().__init__(fl_model, data_handler, hyperparams=hyperparams,
                         info=info, **kwargs)
        # retrieve crypto configuration from info section
        if info and 'crypto' in info and info['crypto'] and isinstance(info['crypto'], dict):
            crypto_config = info['crypto']
        else:
            raise CryptoException('A crypto configuration of type dictionary needs '
                                  'to be provided for crypto initialization!')

        # initialize crypto system
        self.crypto = self.load_crypto_from_config(crypto_config)

        # Initialize keys distribution protocol.
        if 'key_manager' in crypto_config and 'key_mgr_info' in crypto_config['key_manager'] and \
            'distribution' in crypto_config['key_manager']['key_mgr_info']:
            keys_dist_config = crypto_config['key_manager']['key_mgr_info']['distribution']
            self.keys_proto = CryptoKeysProtoParty(config_dst = keys_dist_config, \
                config_crypto = crypto_config.get('crypto_system_info'), crypto_sys = self.crypto)
        else:
            self.keys_proto = None

    def train(self,  fit_params=None):
        """
        Train locally using fl_model. At the end of training, a
        model_update with the new model information is generated and
        send through the connection.
        :param fit_params: (optional) Query instruction from aggregator
        :type fit_params: `dict`
        :return: ModelUpdate
        :rtype: `ModelUpdate`
        """

        # Check for ack init message.
        if self.check_ack_init(fit_params):
            return True

        train_data, (_) = self.data_handler.get_data()

        model_update = fit_params.get('model_update')

        # Check if received fused model update in ciphertext
        if model_update and model_update.exist_key('ct_weights'):
            logger.info('Received encrypted model update.')
            logger.info('Decrypting - ' + str(type(self.crypto)))
            model_update = self.crypto.decrypt(model_update, num_parties=fit_params.get('num_parties'))
            logger.info('Decryption done.')
            # Check if partial decryption occurred
            # - the case of Threshold Paillier decryption style
            if model_update.exist_key('partial_ct_weights'):
                logger.info('Partially decrypted model update, '
                            'send to aggregator for final operation.')
                return model_update

        self.update_model(model_update)

        logger.info('Local training started...')
        self.fl_model.fit_model(train_data, fit_params)
        update = self.fl_model.get_model_update()
        logger.info('Local training done, start to encrypt model update...')

        logger.info('Encrypting - ' + str(type(self.crypto)))
        update = self.crypto.encrypt(update)
        logger.info('Encryption done.')

        return update

    @staticmethod
    def load_crypto_from_config(config):
        """
        Returns a crypto class object according to the provided config file.
        :param config: dictionary of configuration
        :type config: `dict`
        :return: crypto with initialized cipher ready for encryption and \
        decryption operations.
        :rtype: `Crypto`
        """
        try:
            crypto_cls_ref = get_class_by_name(config['path'], config['name'])
            crypto = crypto_cls_ref(config)
        except CryptoException as ex:
            logger.exception(ex)
            raise CryptoException('Error occurred while loading the crypto config.')
        return crypto

    def sync_model_impl(self, payload=None):
        """
        Updates the local model with the global model update received
        from the Aggregator. If the model update is encrypted,
        this method decrypts it.

        :param payload: Data payload received from Aggregator.
        :type payload: `dict`
        :return: Status of the sync model request, and returned plaintext model if requested.
        :rtype: `dict`
        """        
        logger.info('sync_model_impl: start [payload: ' + str(payload) + ']')
        model_update = payload['model_update']
        if model_update.exist_key('ct_weights'):
            logger.info('sync_model_impl: decrypting model - ' + str(type(self.crypto)))
            model_update = self.crypto.decrypt(model_update)
            logger.info('sync_model_impl: decryption done')
        else:
            logger.info('sync_model_impl: model is not encrypted')
        status = False
        status = self.fl_model.update_model(model_update)
        model_return = None
        model_return_ind = False
        if 'model_return_party_ids' in payload and \
            isinstance(payload['model_return_party_ids'], list) and \
            (self.keys_proto is None or \
                (self.keys_proto.id is not None and \
                payload['model_return_party_ids'].count(self.keys_proto.id) > 0)):
            model_return = model_update
            model_return_ind = True
        logger.info('sync_model_impl: end - local model updated [model_return=' + str(model_return_ind) + ']')
        return {'status': status, 'model_return': model_return}

    def request_cert(self, payload=None):
        """
        Returns the id and certificate of this party.

        :param payload: Data payload received from Aggregator.
        :type payload: `dict`
        :return: Dictionary containing the id and certificate of this party.
        :rtype: `dict`
        """
        if self.keys_proto is None:
            raise CryptoException('Keys distribution protocol is not configured for party')
        return self.keys_proto.get_my_cert(payload)

    def generate_keys(self, payload=None):
        """
        Generate crypto keys and a keys distribution message.

        :param payload: Data payload received from Aggregator.
        :type payload: `dict`
        :return: Dictionary of keys distribution message.
        :rtype: `dict`
        """
        if self.keys_proto is None:
            raise CryptoException('Keys distribution protocol is not configured for party')
        return self.keys_proto.generate_keys(payload)

    def distribute_keys(self, payload=None):
        """
        Generate keys distribution message for existing crypto keys.

        :param payload: Data payload received from Aggregator.
        :type payload: `dict`
        :return: Dictionary of keys distribution message.
        :rtype: `dict`
        """
        if self.keys_proto is None:
            raise CryptoException('Keys distribution protocol is not configured for party')
        return self.keys_proto.distribute_keys(payload)

    def set_keys(self, payload=None):
        """
        Set crypto keys received from a generating party.

        :param payload: Data payload received from Aggregator.
        :type payload: `dict`
        :return: Indication if the keys were set successfully.
        :rtype: `bool`
        """
        if self.keys_proto is None:
            raise CryptoException('Keys distribution protocol is not configured for party')
        return self.keys_proto.parse_keys(payload)

    def check_ack_init(self, fit_params) -> bool:
        """
        Check for ack init message.
        """
        if fit_params is not None and 'ack_init' in fit_params:
            logger.info('Received ack_init message')
            return True
        else:
            return False
