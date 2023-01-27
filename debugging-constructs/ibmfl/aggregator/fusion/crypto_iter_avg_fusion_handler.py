"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Module to where fusion algorithms are implemented.
"""
import logging
import random
from ibmfl.model.model_update import ModelUpdate
from ibmfl.aggregator.fusion.iter_avg_fusion_handler import IterAvgFusionHandler
from ibmfl.crypto.crypto_exceptions import CryptoException
from ibmfl.util.config import get_class_by_name
from ibmfl.crypto.keys_mng.crypto_keys_proto_agg import CryptoKeysProtoAgg
from ibmfl.message.message_type import MessageType

logger = logging.getLogger(__name__)


class CryptoIterAvgFusionHandler(IterAvgFusionHandler):
    """
    Class for iterative averaging based fusion algorithms
    implemented with a crypto system.
    In particular, all parties encrypt their models' weights before sending it
    to the aggregator, and the aggregator utilizes `Crypto` to obtain the
    resulting average model weights without knowing the plaintext of the
    parties' local model weights. The global model's weights then are updated
    by the resulting average of all collected local models' weights.

    An iterative fusion algorithm here referred to a fusion algorithm that
    sends out queries at each global round to registered parties for
    information, and use the collected information from parties to update
    the global model.
    """

    def __init__(self, hyperparams,
                 protocol_handler,
                 data_handler=None,
                 fl_model=None,
                 info=None,
                 **kwargs):
        """
        Initializes an IterAvgFusionHandler object with provided information,
        such as protocol handler, fl_model, data_handler and hyperparams.

        :param hyperparams: Hyperparameters used for training.
        :type hyperparams: `dict`
        :param protocol_handler: Protocol handler used for handling learning \
        algorithm's request for communication.
        :type protocol_handler: `ProtoHandler`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param fl_model: model to be trained
        :type fl_model: `model.FLModel`
        :return: None
        """
        super().__init__(hyperparams,
                         protocol_handler,
                         data_handler=data_handler,
                         fl_model=fl_model,
                         info=info)
        # retrieve crypto configuration from info section
        if info and 'crypto' in info and info['crypto'] and isinstance(info['crypto'], dict):
            crypto_config = info['crypto']
        else:
            raise CryptoException('A crypto configuration of type dictionary needs '
                                  'to be provided for crypto initialization!')
        # initialize crypto system
        self.crypto = self.load_crypto_from_config(crypto_config)
        self.name = "Crypto-Iterative-Average"
        self.current_model_update = ModelUpdate(weights=self.current_model_weights) \
            if self.current_model_weights else None

        # Initialize keys distribution protocol.
        if 'key_manager' in crypto_config and 'key_mgr_info' in crypto_config['key_manager'] and 'distribution' in crypto_config['key_manager']['key_mgr_info']:
            self.keys_proto = \
                CryptoKeysProtoAgg(config = crypto_config['key_manager']['key_mgr_info']['distribution'], 
                    protocol_handler = protocol_handler, crypto_sys = self.crypto, perc_quorum = self.perc_quorum)
        else:
            self.keys_proto = None

    def start_global_training(self):
        """
        Starts an iterative global federated learning training process.
        """

        self.curr_round = 0
        while not self.reach_termination_criteria(self.curr_round):

            logger.info('start_global_training: [round: ' + str(self.curr_round) +
                '] [model update: ' + str(self.current_model_update) + ']')

            # Activate keys management.
            if self.keys_proto is not None:
                newKeys = self.keys_proto.manage_keys()
                if newKeys and self.current_model_update is not None and self.current_model_update.exist_key('ct_weights'):
                    # This is the case where there is an existing encrypted model update and new keys were now generated,
                    # therefore the existing model update cannot be used. This does not affect using an initial plaintext model.
                    # This case can happen only if all the parties drop during training and reconnect.
                    self.current_model_update = None

            # Query all available parties.
            payload = {
                'hyperparams': {
                    'local': self.params_local,
                },
                'model_update': self.current_model_update,
                'num_parties': len(self.ph.get_available_parties())
            }
            lst_replies = self.query_all_parties(payload)

            # Perform fusion.
            self.update_weights(lst_replies)

            # Update aggregator model (optional).
            if self.fl_model is not None and \
                    self.current_model_weights is not None:
                self.fl_model.update_model(
                    ModelUpdate(weights=self.current_model_weights))

            self.curr_round += 1
            self.save_current_state()

    def update_weights(self, lst_model_updates):
        """
        Update the global model's weights with the list of collected
        model_updates from parties.
        In this method, it calls the
        self.crypto.crypto_client.avg_collected_response to obtain the average
        the encrypted local model weights collected from parties and
        update the current global model weights
        by the results from self.crypto.avg_collected_response.

        :param lst_model_updates: List of model updates of type `ModelUpdate` \
        to be averaged.
        :type lst_model_updates: `list`
        :return: None
        """
        num_parties = len(self.ph.get_available_parties())
        self.current_model_update = self.crypto.avg_collected_ciphertext_response(lst_model_updates)
        self.current_model_weights = self.current_model_update.get('weights') \
            if self.current_model_update.exist_key('weights') else None

    def load_crypto_from_config(self, config):
        """
        Returns a crypto class object according to the provided config file.
        A cipher class is also initialized to provide encryption and
        decryption service.

        :param config: dictionary of configuration
        :type config: `dict`
        :return: crypto with initialized cipher ready for encryption and \
        decryption operations.
        :rtype: `crypto`
        """
        try:
            crypto_cls_ref = get_class_by_name(config['path'], config['name'])
            crypto = crypto_cls_ref(config, proto_handler=self.ph)
        except CryptoException as ex:
            logger.exception(ex)
            raise CryptoException('Error occurred while loading the crypto config.')
        return crypto

    def get_global_model(self):
        """
        Returns last model_update.

        :return: The last model_update.
        :rtype: `ModelUpdate`
        """
        logger.debug('get_global_model: sending model update [current_model_update: ' + str(self.current_model_update) + ']')
        return self.current_model_update

    def send_global_model(self):
        """
        Send global model to all the parties, 
        and get back plaintext global model from a subset of the parties.
        """
        if self.current_model_update is None:
            raise CryptoException('send_global_model: No current model')
        parties_lst = self.ph.get_available_parties()
        if self.current_model_update.exist_key('ct_weights'):
            N_SLCT_PARTIES = 4
            if len(parties_lst) <= N_SLCT_PARTIES:
                slct_parties_lst = parties_lst
            else:
                slct_parties_idx = random.sample(population=range(len(parties_lst)), k=N_SLCT_PARTIES)
                slct_parties_lst = [parties_lst[idx] for idx in slct_parties_idx]
        else:
            slct_parties_lst = None
        payload = \
            {
                'model_update': self.current_model_update,
                'model_return_party_ids': slct_parties_lst
            }
        logger.info('send_global_model: sending to parties ' + str(payload))
        replies = self.ph.query_parties(payload=payload, lst_parties=parties_lst, msg_type=MessageType.SYNC_MODEL)
        model_return = local_model_updated = False
        fail_status = 0
        for reply in replies:
            if 'status' in reply and isinstance(reply['status'], bool) and reply['status'] == False:
                fail_status += 1
            if not model_return and 'model_return' in reply and reply['model_return'] is not None:
                model_return = True
                model = reply['model_return']
                if not model.exist_key('weights'):
                    raise CryptoException('send_global_model: Returned model does not have a weights attribute')
                self.current_model_weights = model.get('weights')
                if self.current_model_weights is None:
                    raise CryptoException('send_global_model: Returned model weights are empty')
                self.current_model_update = model
                if self.fl_model is not None:
                    self.fl_model.update_model(ModelUpdate(weights=self.current_model_weights))
                    local_model_updated = True
        if fail_status > 0:
            raise CryptoException('send_global_model: ' + str(fail_status) + ' parties returned a failure reply')
        logger.info('send_global_model: end [model_return=' + str(model_return) + 
            '] [local_model_updated=' + str(local_model_updated) + ']')
        return
