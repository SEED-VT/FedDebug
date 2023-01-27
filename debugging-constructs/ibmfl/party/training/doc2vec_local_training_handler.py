"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
from __future__ import print_function
import inspect
import logging
from collections import defaultdict
from gensim.utils import simple_preprocess
from ibmfl.exceptions import LocalTrainingException, FLException
from ibmfl.party.training.local_training_handler import LocalTrainingHandler

logger = logging.getLogger(__name__)


class Doc2VecLocalTrainingHandler(LocalTrainingHandler):
    """
    Class implementation for Doc2Vec Local Training Handler
    """
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
        handler, e.g., a crypto library object to help with encryption and decryption.
        :type kwargs: `dict`
        :return None
        """
        super().__init__(fl_model, data_handler, hyperparams, **kwargs)
        self.train_data = self.data_handler.get_data()

    def update_model(self, model_update):
        """
        Update local model with model updates received from FusionHandler

        :param model_update: ModelUpdate
        :type model_update: `ModelUpdate`
        :return: `None`
        """
        try:
            if model_update is not None:
                self.fl_model.update_model(model_update)
                logger.info('Local model updated.')
            else:
                logger.info('No model update was provided.')
        except Exception as ex:
            raise LocalTrainingException(
                'No query information is provided', str(ex))

    def train(self, fit_params=None):
        """
        Primary wrapper function used for routing internal remote function calls
        within the Local Training Handler functions.

        :param fit_params: A dictionary payload structure containing two key \
        signatures, `func` and `args`, which respectively are the target \
        function defined within the Local Training Handler and the arguments \
        defined within the executing function, which is defined as a dictionary \
        containing key-value pairs of matching arguments.
        :type fit_params: `dict`
        :return: Returns the corresponding values depending on the function \
        remotely called from the aggregator.
        """
        try:
            # Validate Incoming Payload Parameter
            if fit_params is None:
                raise LocalTrainingException('Provided fit_params is None, no '
                                             'functions were executed.')
            # Validate Payload Signature
            if 'func' in fit_params and 'args' in fit_params:
                # Validate Defined Function Header
                if not (isinstance(fit_params['func'], str) and
                        hasattr(self, fit_params['func'])):
                    raise LocalTrainingHandler('Function header is not valid or is '
                                               'not defined within the scope of the '
                                               'local training handler.')

                # Validate Payload Argument Parameter Mappings Against Function
                spec = inspect.getargspec(eval('self.'+fit_params['func']))
                for k in fit_params['args'].keys():
                    if k not in spec.args:
                        raise LocalTrainingHandler('Specified parameter argument is '
                                                   'not defined in the function.')

                # Construct Function Call Command
                result = eval('self.'+fit_params['func'])(**fit_params['args'])
                return result

            else:
                self.update_model(fit_params.get('model_update'))
                logger.info('Local training started...')

                train_data = self.train_data
                self.fl_model.fit_model(train_data, fit_params, rounds=fit_params.get('rounds'))

                update = self.fl_model.get_model_update()
                logger.info('Local training done, generating model update...')

                return update

        except Exception as ex:
            raise LocalTrainingException('Error processing remote function ' +
                                         'call: ' + str(ex))

    def eval_model(self, payload=None):
        """
        Evaluate the local model based on data provided from the Aggregator

        :param payload: data payload received from Aggregator
        :type payload: `dict`
        :return: list of evaluation results
        :rtype: `list`
        """
        if payload is None or 'data' not in payload:
            raise FLException("Missing data to infer from Aggregator")

        try:
            eval = None
            data = payload['data']
            eval = self.fl_model.evaluate(data)
            logger.info('done with evaluations')
        except Exception as ex:
            logger.info("Exception occurred during evaluation")
            logger.exception(ex)

        payload = {'evaluation': eval}
        logger.info(payload)
        return payload

    def get_vocabulary(self, round_zero):
        """
        A remote function call which performs the following procedures:
        1. Creates dictionary of words and their frequency in training data set
        :return: Dictionary containing training dataset's vocabulary and frequencies
        :rtype: `dict`
        """
        logger.info('Obtaining list of vocabulary words and frequency.')

        documents = self.data_handler.get_data()
        vocab = defaultdict(int)
        # check if get data returns tuple, or assume a list of TaggedDocument
        if type(documents) is tuple:
            documents = documents[0]
            for doc in documents:
                tokens = simple_preprocess(doc)
                for word in tokens:
                    vocab[word] += 1

        else:
            for doc in documents:
                for word in doc[0]:
                    vocab[word] += 1

        payload = {'vocab': vocab}
        return payload

    def set_initial_model(self, initial_model):
        """
        Sets an initial Doc2Vec model received from the aggregator
        :param initial_model: An initialized doc2vec model
        :type initial_model: 'gensim.models.doc2vec.Doc2Vec'
        """
        logger.info('Initializing model retrieved from aggregator')
        self.fl_model.model = initial_model

        # check if training data is a tuple (docs, doc_id), or assume a list of TaggedDocument
        if type(self.train_data) is tuple:
            training_corpus = self.fl_model.build_tagged_documents(self.train_data[0], self.train_data[1])
        else:
            training_corpus = self.train_data

        total_words, corpus_count = self.fl_model.model.vocabulary.scan_vocab(documents=training_corpus,
                                                                              docvecs=self.fl_model.model.docvecs)
        self.fl_model.model.corpus_count = corpus_count
        self.fl_model.model.corpus_total_words = total_words
        self.fl_model.model.trainables.reset_doc_weights(self.fl_model.model.docvecs)

        return True
