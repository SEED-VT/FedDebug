"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Module where the crypto library are defined.
"""

import abc
import sys
import logging

from ibmfl.model.model_update import ModelUpdate
from ibmfl.crypto.crypto_exceptions import CryptoException, KeyManagerException
from ibmfl.util.config import get_class_by_name

logger = logging.getLogger(__name__)

# the default precision setting.
PRECISION = 5

class Crypto(abc.ABC):
    """
    The base class for a crypto system implementation that will be exposed to
    other FL modules.
    """

    def __init__(self, config, **kwargs):
        self.cipher = None
        if not config:
            raise CryptoException('No crypto config is provided')
        if 'precision' in config:
            if isinstance(config['precision'], int) and (config['precision'] > 0):
                self.precision = config['precision']
            else:
                logger.warning(
                    'Provided precision is not in the correct format. '
                    'An accepted precision should be a positive integer. '
                    'Setting the precision to default precision: '
                    + str(PRECISION))
                self.precision = PRECISION
        else:
            self.precision = PRECISION
        
        self.idx = None
        logger.info("Initializing a key manager")
        if 'key_manager' not in config:
            raise CryptoException('no key manager provided')
        key_mgr_config = config['key_manager']
        try:
            key_mgr_cls_ref = get_class_by_name(key_mgr_config['path'],
                                                key_mgr_config['name'])
            self.key_manager = key_mgr_cls_ref(key_mgr_config['key_mgr_info'])
            self.idx = key_mgr_config['key_mgr_info'].get('idx', None)
        except KeyManagerException as ex:
            logger.exception(ex)

        self.initialize(config)

        self.ph = kwargs.get('proto_handler')

    @abc.abstractmethod
    def initialize(self, config):
        """
        Initialize a `KeyManager` object to generate keys for `Cipher` object;
        Initialize a 'Cipher' object

        :param config: A dictionary containing the provided crypto_keys,
            including public and private key(s), for encryption and decryption;
         etc., to initialize a cipher of a specific cryptosystem.
        :type config: `dict`
        :param kwargs: Dictionary of arguments required by a crypto system \
        implementation for encryption and decryption.
        :return: None
        """
        raise NotImplementedError

    def set_keys(self, keys):
        """
        Set the keys in use cases where the keys are not set during initialization
        of this object, e.g. for the keys generation and distribution protocol use cases.
     
        :param keys: The provided keys.
        :type keys: `dict`
        :return: None
        """
        self.cipher.set_keys(keys)
        return

    def encrypt(self, model_update, key='weights', **kwargs):
        """
        Method for the encryption operations FL party will perform
        during training. It currently calls the `encrypt_weights` method
        defined inside the Cipher class.
        
        :param model_update: ModelUpdate object to be encrypted.
        :type model_update: `ModelUpdate`
        :param key: model
        :type key: `str`
        :return: The resulting ciphertext of model update.
        :type: `ModelUpdate`
        """
        if not isinstance(model_update, ModelUpdate):
            raise CryptoException("not a ModelUpdate instance to be encrypted")

        model_weights = model_update.get(key)
        ct_model_update = ModelUpdate()
        if isinstance(model_weights, list):
            ct_model_update.add("ct_{}".format(key), self.cipher.encrypt_weights(model_weights))
        else:
            ct_model_update.add("ct_{}".format(key), self.cipher.encrypt_weights([model_weights]))
        return ct_model_update

    @abc.abstractmethod
    def decrypt(self, ct_model_update, key='ct_weights', **kwargs):
        """
        Abstract method to perform decryption or partial decryption operations
        FL party will perform during training.
        Decrypts the ciphertext included inside the provided `model_update`.
        The ciphertext is indicated by the dict_key.
        :param ct_model_update: A `ModelUpdate` object containing the provided \
        ciphertext to be decrypted. The ciphertext should be of \
        type `list` of `np.ndarray`.
        :type ct_model_update: `ModelUpdate`
        :param key: A dictionary key value indicating what encrypted \
        values inside the `model_update` objects the method will decrypt.
        :type key: `str`
        :param kwargs: Dictionary of arguments required to perform decryption \
        or partial decryption operations.
        :type kwargs: `dict`
        :return: The resulting decrypted plain text.
        :rtype: `list` of `np.ndarray`
        """
        raise NotImplementedError

    def avg_collected_ciphertext_response(self, lst_model_updates, **kwargs):
        """
        Receives a list of `model update`, where a `model_update` is of type
        `ModelUpdate`, using the encrypted values (indicating by the `key`)
        included in each `model_update`,
        it returns the decrypted resulting mean.

        :param: lst_model_updates. List of `model_update` of type
            `ModelUpdate` including the encrypted values (indicating by the `key`)
            to be averaged.
        :type lst_model_updates:  `list`

        :return: results of fused encrypted model update
        :rtype: `ModelUpdate`
        """
        fused_model_update = self.aggregate_collected_ciphertext_response(lst_model_updates)
        if fused_model_update.exist_key('weights'):
            avg_model_weights = fused_model_update.get('weights')
            for i in range(len(avg_model_weights)):
                avg_model_weights[i] = avg_model_weights[i] / len(lst_model_updates)
            avg_model_update = ModelUpdate(weights=avg_model_weights)
            return avg_model_update
        else:
            # fused model update is still in ciphertext format
            return fused_model_update

    @abc.abstractmethod
    def aggregate_collected_ciphertext_response(self, lst_model_updates, key='ct_weights', **kwargs):
        raise NotImplementedError


class Cipher(abc.ABC):
    """
    This class creates a crypto `Cipher` that encrypts and decrypts
    the message content exchanged during FL training.
    """

    def __init__(self, crypto_keys, precision, **kwargs):
        """
        Initializes a crypto `Cipher` for a crypto system with
        provided crypto keys, and set the `NEGATIVE_THRESHOLD` as the
        maximum size allowed by the system.

        :param crypto_keys: Provided crypto_keys, including public and \
        private key(s), for encryption and decryption.
        :type crypto_keys: `dict`
        :param precision: Provided precision for encoding and decoding, \
        default value is set to 6.
        :type precision: `int`
        :param kwargs: A dictionary contains additional arguments to \
        initialize a crypto `Cipher`.
        :type kwargs: `dict`
        """
        self.NEGATIVE_THRESHOLD = sys.maxsize
        self.keys = crypto_keys
        self.precision = precision

    def set_keys(self, keys):
        """
        Set the keys in use cases where the keys are not set during initialization
        of this object, e.g. for the keys generation and distribution protocol use cases.
     
        :param keys: The provided keys.
        :type keys: `dict`
        :return: None
        """
        raise NotImplementedError

    def encrypt(self, plaintext, **kwargs):
        raise NotImplementedError

    def decrypt(self, ciphertext, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def encrypt_weights(self, model_weights, **kwargs):
        """
        Encrypts the provided plain_text according to the crypto system setup,
         the encoding precision, and other encryption requirements defined
         inside the kwargs.

        :param model_weights: Plain text values to be encrypted.
        :type model_weights: `list` of `np.ndarray`
        :param kwargs: Encryption requirements
        :type kwargs: `dict`
        :return: Encrypted text
        :rtype: `list`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def decrypt_weights(self, model_weights_ciphertext, **kwargs):
        """
        Decrypts the provided cipher_text according to the crypto system
        setup, the encoding precision, and other decryption requirements
        defined inside the kwargs.

        :param model_weights_ciphertext: Cipher text to be decrypted.
        :type model_weights_ciphertext: `list`
        :param kwargs: Decryption requirements
        :type kwargs: `dict`
        :return: Decrypted text
        :rtype: `list`
        """
        raise NotImplementedError

    def _has_public_key(self):
        """
        Checks the existence of the public key(s).

        :return: True if the public key(s) exists otherwise returns False.
        :rtype: `boolean`
        """
        return 'pk' in self.keys or 'pp' in self.keys

    def _has_private_key(self):
        """
        Checks the existence of the private key(s).

        :return: True if the private key(s) exists otherwise returns False.
        :rtype: `boolean`
        """
        return 'sk' in self.keys
