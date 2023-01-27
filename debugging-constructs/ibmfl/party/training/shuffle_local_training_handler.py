"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging

from ibmfl.model.model_update import ModelUpdate
from ibmfl.party.training.local_training_handler import LocalTrainingHandler
from ibmfl.util import shuffle

from ibmfl.util.shuffle_config import get_seed, \
    get_seed_filename

from ibmfl.exceptions import LocalTrainingException, \
    InvalidConfigurationException

logger = logging.getLogger(__name__)


class ShuffleLocalTrainingHandler(LocalTrainingHandler):

    def __init__(self, fl_model, data_handler, hyperparams=None, **kwargs):
        """
        Initialize LocalTrainingHandler with fl_model, data_handler

        :param fl_model: model to be trained
        :type fl_model: `model.FLModel`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param hyperparams: Hyperparameters used for training.
        :type hyperparams: `dict`
        :param kwargs: Additional arguments to initialize a local training \
        handler, e.g., permutation seed file.
        :type kwargs: `dict`
        :return None
        """
        super().__init__(fl_model, data_handler, hyperparams, **kwargs)

        # initial curr_seed is 0 and it will be updated with the aggregator's seed generated for every round
        self.curr_seed = 0
        self.permute_secret = 0

        if not kwargs:
            raise InvalidConfigurationException('No local_training info given at runtime')

        seed_file = get_seed_filename(kwargs)
        self.permute_secret = get_seed(seed_file)

    def shuffle_model(self, model_update, seed):
        """
        Shuffle model updates received from FusionHandler

        :param model_update: ModelUpdate
        :type model_update: `ModelUpdate`
        :param seed: seed dispatched from aggregator of current training round
        :type seed: `int`
        """
        allw = model_update.get('weights')
        new_update = shuffle.shuffleweight(allw, self.permute_secret ^ seed)
        return ModelUpdate(weights=new_update)

    def unshuffle_model(self, model_update, seed):
        """
        Unshuffle model updates received from FusionHandler

        :param model_update: ModelUpdate
        :type model_update: `ModelUpdate`
        :param seed: seed dispatched from aggregator of current training round
        :type seed: `int`
        """
        allw = model_update.get('weights')
        new_update = shuffle.unshuffleweight(allw, self.permute_secret ^ seed)
        return ModelUpdate(weights=new_update)

    def update_model(self, model_update, seed=0):
        """
        Update local model with shuffled model updates received from Shuffle Iter Avg FusionHandler

        :param model_update: ModelUpdate
        :type model_update: `ModelUpdate`
        :param seed: seed dispatched from aggregator of current training round
        :type seed: `int`
        """
        try:
            if model_update is not None:
                unshuffled_update = self.unshuffle_model(model_update, seed)
                self.fl_model.update_model(unshuffled_update)
                logger.info('Local model updated.')
            else:
                logger.info('No model update was provided.')
        except Exception as ex:
            raise LocalTrainingException('No query information is provided. ' + str(ex))

    def train(self, fit_params=None):
        """
        Train locally using fl_model. At the end of training, a
        model_update with the new model information is generated and
        sent through the connection.

        :param fit_params: (optional) Query instruction from aggregator
        :type fit_params: `dict`
        :return: ModelUpdate
        :rtype: `ModelUpdate`
        """
        train_data, (_) = self.data_handler.get_data()

        # training first get model updates from the aggregator
        # update our local model
        # If it is the first time for the aggregator to start the training,
        #   model_update will be NONE and the curr_seed will be 0
        logger.info("update model before local training curr_seed:" + str(self.curr_seed))

        # When we update the model in train(), we should use the last curr_seed.
        # Because we shuffled with the last curr_seed and need to unshuffle with the same seed
        self.update_model(fit_params.get('model_update'), seed=self.curr_seed)
        self.get_train_metrics_pre()

        # After that, we can update our local curr_seed with the new seed from aggregator
        self.curr_seed = fit_params.get('aggseed')
        logger.info('update curr_seed to aggregator seed:' + str(self.curr_seed))
        logger.info('Local training started...')
        self.fl_model.fit_model(train_data, fit_params)

        # send local updates to aggregator for fusion
        update = self.fl_model.get_model_update()
        logger.info('Local training done, generating model update...')
        
        # shuffle the update here and return the shuffled update
        shuffled_model = self.shuffle_model(update, self.curr_seed)
        logger.info('Finish model updates shuffling')

        self.get_train_metrics_post()
        return shuffled_model

    def sync_model_impl(self, payload=None):
        """
        Update the local model with global ModelUpdate received
        from the Aggregator. This function is meant to be 
        overridden in base classes as opposed to sync_model, which
        contains boilerplate for exception handling and metrics.

        :param payload: data payload received from Aggregator
        :type payload: `dict`
        :return: Status of sync model request
        :rtype: `boolean`
        """
        status = False
        model_update = payload['model_update']

        # unshuffle the model update from aggregator before updating the model
        unshuffled_model = self.unshuffle_model(model_update, self.curr_seed)

        logger.info('Sync whole model')
        status = self.fl_model.update_model(unshuffled_model)

        return status
