"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Fully homomorphic encryption cryptosystem, built on helayer library.

* type:     public-key encryption
* setting:  Integer based

:Date: 2021
"""
import logging
import importlib

if importlib.util.find_spec("pyhelayers") is not None:
    import pyhelayers as pyhe

import numpy as np

from ibmfl.crypto.crypto_library import Crypto, Cipher
from ibmfl.crypto.crypto_enumeration import CryptoEnum
from ibmfl.crypto.crypto_exceptions import CryptoException, KeyManagerException
from ibmfl.model.model_update import ModelUpdate

logger = logging.getLogger(__name__)

class CryptoFHE(Crypto):
    """
    The crypto class for FHE crypto system, where its KeyGenerator and
    Cipher can be initialized.
    """
    def initialize(self, config):
        """
        Initializes cryptographic keys to generate the cipher object

        :param config: A dict that contains required configuration
        :type config: `dict`
        :return: None
        """
        logger.info("Initializing keys")
        try:
            key_config = {
                'type': CryptoEnum.CRYPTO_FHE.value,
            }
            keys = self.key_manager.initialize_keys(config=key_config)
        except KeyManagerException as ex:
            logger.exception(ex)
            raise KeyManagerException("Initialization of FHE context failed.")

        logger.info("Initializing a FHE Cipher")
        try:
            self.cipher = CipherFHE(keys, 0)
        except CryptoException as ex:
            logger.error(str(ex))
            raise CryptoException('Initialization of FHE Cipher failed.')
        
        self.private_fusion_weight = config.get('private_fusion_weights')
        if self.private_fusion_weight is None:
            logger.info("No flag for privacy of fusion weights in config. "
                        "Setting to default value of False.")
            self.private_fusion_weight = False
        if not isinstance(self.private_fusion_weight, bool):
            raise CryptoException('Flag private_fusion_weights must be boolean.')

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
        
        ct_model_update = ModelUpdate()
        weights = model_update.get(key)     
        counts = kwargs.get('counts')
        if self.private_fusion_weight:
            ct_weights, ct_counts = self.cipher.encrypt_weights(weights, counts=counts)
            ct_model_update.add('ct_weights', ct_weights)
            ct_model_update.add("ct_counts", ct_counts)
        else:
            ct_weights = self.cipher.encrypt_weights(weights)
            ct_model_update.add('ct_weights', ct_weights)
            ct_model_update.add('counts', counts)
        ct_model_update.add('weights_dtype', self.cipher.weights_dtype)

        return ct_model_update

    def decrypt(self, ct_model_update, **kwargs):
        """
        Method for processing fused model update in ciphertext that
        FL party will perform during training.

        :param ct_model_update: A `ModelUpdate` object containing the provided
        ciphertext to be decrypted. The ciphertext should be of
        type `list` of `np.ndarray`.
        :type ct_model_update: `ModelUpdate`
        :return: A `ModelUpdate` object containing the resulting decrypted \
        plain text, where the plain text is of type `list` of `np.ndarray`.
        :rtype: `ModelUpdate`
        """
        ct_weights = ct_model_update.get("ct_weights")
        weights = self.cipher.decrypt_weights(ct_weights, weights_dtype=ct_model_update.get('weights_dtype'))
        res_model_update = ModelUpdate()
        res_model_update.add("weights", weights)

        return res_model_update

    def aggregate_collected_ciphertext_response(self, lst_model_updates, key='ct_weights', **kwargs):
        """
        Receives a list of `model update`, where a `model_update` is of type
        `ModelUpdate`, using the encrypted values (indicating by the `key`)
        included in each `model_update`, it returns the fused model udpate.
        NOTE: if `private_fusion_weight` is set to `False`, the fusion weight
        of aggregation (`lst_fusion_weights`) should provided in plaintext;
        otherwise, the fusion weight is protected by FHE and it is required to 
        provide `min_counts` and `max_counts` for counted tranining samples. 

        :param: lst_model_updates. List of `model_update` of type
            `ModelUpdate` including the encrypted values (indicating by
            the `key`)
            to be averaged.
        :type lst_model_updates:  `list`

        :return: results of fused encrypted model update
        :rtype: `ModelUpdate`
        """
        logger.info('Start ciphertext fusing')

        # private_fusion_weight defined as the trigger to protect counts (fusion weight).
        weights_dtype = None
        if not self.private_fusion_weight:
            lst_ct_weights = []
            lst_fusion_weights = []
            for model_update in lst_model_updates:
                lst_ct_weights.append(model_update.get('ct_weights'))
                if model_update.get('counts') is not None:
                    lst_fusion_weights.append(model_update.get('counts'))
                cur_weights_dtype = model_update.get('weights_dtype')
                if cur_weights_dtype is not None:
                    if weights_dtype is None:
                        weights_dtype = cur_weights_dtype
                    elif weights_dtype != cur_weights_dtype:
                        logger.warning('aggregate_collected_ciphertext_response: weights dtype mismatch: ' +
                            str(weights_dtype) + ' != ' + str(cur_weights_dtype))
            if len(lst_fusion_weights) > 0:
                lst_fusion_weights /= np.sum(lst_fusion_weights)
            else:
                lst_fusion_weights = [1.0/len(lst_model_updates) for _ in lst_model_updates]
            ct_fused_weights = self.cipher.fuse_weights_scalar(lst_ct_weights, lst_fusion_weights)
        else:
            lst_ct_weights = []
            lst_ct_counts = []
            for model_update in lst_model_updates:
                lst_ct_weights.append(model_update.get('ct_weights'))
                lst_ct_counts.append(model_update.get('ct_counts'))
                cur_weights_dtype = model_update.get('weights_dtype')
                if cur_weights_dtype is not None:
                    if weights_dtype is None:
                        weights_dtype = cur_weights_dtype
                    elif weights_dtype != cur_weights_dtype:
                        logger.warning('aggregate_collected_ciphertext_response: weights dtype mismatch: ' +
                            str(weights_dtype) + ' != ' + str(cur_weights_dtype))
            low_bound = kwargs.get('min_counts', 0)
            up_bound = kwargs.get('max_counts', 1000)
            eps = kwargs.get('eps', 1e-5)
            ct_fused_weights = self.cipher.fuse_weights(lst_ct_weights,
                                                        lst_ct_counts, 
                                                        low_bound, up_bound, 
                                                        eps)

        model_update = ModelUpdate(ct_weights=ct_fused_weights, weights_dtype=weights_dtype)
        logger.info('Ciphertext fusion done')
        return model_update


class CipherFHE(Cipher):
    """
    This class module implements the encryption and decryption functionality
    for general FHE crypto system assuming key(s) will be provided
    during initialization.
    """
    def __init__(self, keys, precision, **kwargs):
        """
        Initializes a crypto cipher for FHE crypto system with
        provided crypto keys.

        :param keys: Provided crypto keys
        :type keys: `dict`
        :return: None
        """
        if 'files' in keys:
            self.context = pyhe.DefaultContext()
            self.context.load_from_file(keys['files']['context'])
            if 'key' in keys['files']:
                self.context.load_secret_key_from_file(keys['files']['key'])
                self.crypto = pyhe.TTEncoder(self.context)
                self.slots = self.context.slot_count()
        elif 'distribution' in keys:
            self.context = None
            self.crypto = None
        else:
            self.context = pyhe.DefaultContext()
            self.context.load_from_buffer(keys['pp'])
            if 'sk' in keys:
                self.context.load_secret_key(keys['sk'])
                self.crypto = pyhe.TTEncoder(self.context)
                self.slots = self.context.slot_count()
        self.weights_dtype = None

    def set_keys(self, keys):
        logger.debug("CipherFHE set_keys start")
        self.context = pyhe.DefaultContext()
        self.context.load_from_buffer(keys['pp'])
        if 'sk' in keys:
            self.context.load_secret_key(keys['sk'])
            self.crypto = pyhe.TTEncoder(self.context)
            self.slots = self.context.slot_count()
        logger.debug("CipherFHE set_keys end")

    def encrypt_weights(self, model_weights, **kwargs):
        """
        Encrypts the provided plain text of type list of `np.ndarray`.

        :param model_weights: The provided plain text to be encrypted.
        :type model_weights: `list` of `np.ndarray`
        :return: The resulting ciphertext of the provided plain text.
        :type: `list` of `np.ndarray`
        """
        logger.debug("CipherFHE encrypt_weights start")
        cipher_weights = []
        shape = pyhe.TTShape([1, self.slots])
        for i in range(len(model_weights)):
            weight = model_weights[i]
            if not isinstance(weight, np.ndarray):
                weight = np.asarray(weight)
            ct = self.crypto.encode_encrypt(shape, weight.reshape((1, weight.size)))
            cipher_weights.append([ct.save_to_buffer(), weight.shape])
            if self.weights_dtype is None:
                self.weights_dtype = weight.dtype
            elif self.weights_dtype != weight.dtype:
                logger.warning('encrypt_weights: weights dtype for multiple layers is different: '
                    '[self.weights_dtype=' + str(self.weights_dtype) + '] [weight.dtype=' + str(weight.dtype) + ']')

        # protect counts for FedAvg fusion method
        counts = kwargs.get('counts', None)
        if counts is not None:
            shape_counts = shape.get_with_duplicated_dims([0,1])
            ct_counts = self.crypto.encode_encrypt(
                shape_counts, np.array(counts).reshape(1,1))
            return cipher_weights, ct_counts.save_to_buffer()
        logger.debug("CipherFHE encrypt_weights end")
        return cipher_weights

    def decrypt_weights(self, ct_model_weights, **kwargs):
        """
        Decrypts the provided ciphertext of type list of `np.ndarray`.

        :param ct_model_weights: The provided ciphertext to be decrypted.
        :type ct_model_weights: `list` of `np.ndarray`
        :return: The resulting decrypted plain text.
        :rtype: `list` of `np.ndarray`
        """
        logger.debug("CipherFHE decrypt_weights start")

        if self.weights_dtype is None:
            weights_dtype = kwargs.get('weights_dtype')
            if weights_dtype is not None:
                self.weights_dtype = weights_dtype
                logger.info('decrypt_weights: weights dtype obtained from incoming model: ' +
                            str(self.weights_dtype))
            else:
                self.weights_dtype = np.zeros(1,np.float32).dtype
                logger.warning('decrypt_weights: weights dtype is None. '
                            'Using default dtype: ' + str(self.weights_dtype))
            
        dec_weights = list()
        for i in range(len(ct_model_weights)):
            target_shape = ct_model_weights[i][1]
            ct = pyhe.CTileTensor(self.context)
            ct.load_from_buffer(ct_model_weights[i][0])
            dec_weight = self.crypto.decrypt_decode_double(ct)
            if dec_weight.dtype != self.weights_dtype:
                dec_weight = dec_weight.astype(self.weights_dtype)
            dec_weights.append(dec_weight.reshape(target_shape))
        logger.debug("CipherFHE decrypt_weights end")
        return dec_weights

    def fuse_weights_scalar(self, ct_lst_weights, lst_fusion_weight):
        """
        Fuse a set of encrypted model weights with plaintext fusion
        weight for each model.

        :param ct_lst_weights: the list of ciphertext model weights
        :type ct_lst_weights: `list` of `np.ndarray`
        :param lst_fusion_weight: the list of fusion weight in plaintext
        :type lst_fusion_weight: `list`
        :return: fused ciphertext weights
        """
        logger.debug("CipherFHE fuse_weights_scalar start")
        if len(lst_fusion_weight) != len(ct_lst_weights):
            raise CryptoException('Size of fusion weight does not match'
                                  'the size of recevied ciphertext list')
        fused_ct_keras = ct_lst_weights[0]
        for layer_idx in range(len(fused_ct_keras)):
            fused_ct = pyhe.CTileTensor(self.context)
            fused_ct.load_from_buffer(fused_ct_keras[layer_idx][0])
            fused_ct.multiply_scalar(lst_fusion_weight[0])
            for party_idx in range(1, len(ct_lst_weights)):
                party_ct = pyhe.CTileTensor(self.context)
                party_ct.load_from_buffer(
                    ct_lst_weights[party_idx][layer_idx][0])
                party_ct.multiply_scalar(lst_fusion_weight[party_idx])
                fused_ct.add(party_ct)
            fused_ct_keras[layer_idx][0] = fused_ct.save_to_buffer()
        logger.debug("CipherFHE fuse_weights_scalar end")
        return fused_ct_keras

    def fuse_weights(self, ct_lst_weights, ct_lst_fusions, low_bound, up_bound,
                     eps):
        """
        Fuse a set of encrypted model weights with plaintext fusion
        weight for each model.

        :param ct_lst_weights: the list of ciphertext model weights
        :type ct_lst_weights: `list` of `np.ndarray`
        :param ct_lst_fusions: the list of encrypted fusion weight in plaintext
        :type ct_lst_fusions: `list` of `np.ndarray`
        :param low_bound: low bound for the input for the inverse function
        :type low_bound: `int`
        :param up_bound: up bound for the input for the inverse function
        :type up_bound: `int`
        :param eps: eps parameter
        :type eps: `double`
        :return: inversed encrypted fusion weights
        :rtype: `list` of `np.ndarray`
        """
        logger.debug("CipherFHE fuse_weights start")
        # address the encrypted fusion weight aggregation with eps
        logger.info('perform inverse of encrypted aggregated fusion weight with eps')
        ct_fusion_sum = pyhe.CTileTensor(self.context)
        ct_fusion_sum.load_from_buffer(ct_lst_fusions[0])
        for party_idx in range(1, len(ct_lst_fusions)):
            ct_fusion_party = pyhe.CTileTensor(self.context)
            ct_fusion_party.load_from_buffer(ct_lst_fusions[party_idx])
            ct_fusion_sum.add(ct_fusion_party)
        ct_fusion_sum.add_scalar(eps)
        func_eval = pyhe.TTFunctionEvaluator(self.context)
        ct_inverse = func_eval.inverse(ct_fusion_sum, low_bound, up_bound)

        # address weighted fusion over encryptd fusions and models
        logger.info('perform weighted sum over encrypted model weights')
        fused_ct_model = ct_lst_weights[0]
        ct_fusion_party0 = pyhe.CTileTensor(self.context)
        ct_fusion_party0.load_from_buffer(ct_lst_fusions[0])
        for layer_idx in range(len(fused_ct_model)):
            fused_ct = pyhe.CTileTensor(self.context)
            fused_ct.load_from_buffer(fused_ct_model[layer_idx][0])
            fused_ct.multiply(ct_fusion_party0)
            for party_idx in range(1, len(ct_lst_weights)):
                party_ct = pyhe.CTileTensor(self.context)
                party_ct.load_from_buffer(
                    ct_lst_weights[party_idx][layer_idx][0])
                party_ct_fusion = pyhe.CTileTensor(self.context)
                party_ct_fusion.load_from_buffer(ct_lst_fusions[party_idx])
                party_ct.multiply(party_ct_fusion)
                fused_ct.add(party_ct)
            fused_ct.multiply(ct_inverse)
            fused_ct_model[layer_idx][0] = fused_ct.save_to_buffer()
        logger.debug("CipherFHE fuse_weights end")
        return fused_ct_model

