"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Doc2Vec FL: Fusion handler
"""
import logging
import numpy as np
from collections import defaultdict
from gensim.models import doc2vec
from ibmfl.model.model_update import ModelUpdate
from ibmfl.message.message_type import MessageType
from ibmfl.aggregator.fusion.iter_avg_fusion_handler import IterAvgFusionHandler
from ibmfl.exceptions import FLException

logger = logging.getLogger(__name__)


class Doc2VecFusionHandler(IterAvgFusionHandler):
    """
    Class for fusion doc2vec models. Utilizes iterative averaging algorithm.
    An iterative fusion algorithm here refers to a fusion algorithm that
    sends out queries at each global round to registered parties for
    information, and use the collected information from parties to update
    the global model.
    The type of queries sent out at each round is the same. For example,
    at each round, the aggregator send out a query to request local model's
    weights after parties local training ends.

    For doc2vec, the aggregator first requests a dictionary of all parties vocabulary
    and word frequency, and merges them before sending initial model
    Afterwards, the aggregator requests local model's weights from all
    parties at each round, and the averaging aggregation is performed over
    collected model weights. The global model's weights then are updated by
    the mean of all collected local models' weights.
    """

    def __init__(self, hyperparams,
                 protocol_handler,
                 data_handler=None,
                 fl_model=None,
                 **kwargs):
        """
        Initializes a Doc2VecFusionHandler object with provided information,
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
        :param kwargs: Additional arguments to initialize a fusion handler.
        :type kwargs: `Dict`
        :return: None
        """
        super().__init__(hyperparams,
                         protocol_handler,
                         data_handler,
                         fl_model,
                         **kwargs)
        self.name = "Doc2Vec-Fusion"
        self.params_global = hyperparams.get('global') or {}
        self.params_local = hyperparams.get('local') or None

        self.rounds = self.params_global.get('rounds') or 1
        self.curr_round = 0

        if fl_model and fl_model.is_fitted():
            model_update = fl_model.get_model_update()
        else:
            model_update = None

        self.current_model_weights = \
            model_update.get('weights') if model_update else None

    def start_global_training(self):
        """
        Starts an iterative global federated learning training process.
        """
        self.curr_round = 0
        # [RPC] Perform Initialization of the Local Worker by first obtaining their training vocabulary
        logger.info('Perform Local Training Handler Initialization Process')
        vocabulary = self.query('get_vocabulary', {'round_zero': True})

        # Unpack vocabulary and merge
        vocab_lists = []
        for v in vocabulary:
            vocab_lists.append(v['vocab'])

        merged_vocab = self.merge_vocab(vocab_lists)

        # Set initial model with merged vocabulary
        initial_model = self.set_initial_model(merged_vocab)

        # distribute initial model
        self.query('set_initial_model', {'initial_model': initial_model})

        while not self.reach_termination_criteria(self.curr_round):
            if self.current_model_weights:
                model_update = ModelUpdate(weights=self.current_model_weights)
            else:
                model_update = None

            payload = {'hyperparams': {'local': self.params_local},
                       'model_update': model_update,
                       'rounds': self.rounds}

            logger.info('Model update' + str(model_update))

            # query all available parties
            lst_replies = self.query_all_parties(payload)

            self.update_weights(lst_replies)

            # Update model if we are maintaining one
            if self.fl_model is not None:
                self.fl_model.update_model(ModelUpdate(weights=self.current_model_weights))

            self.curr_round += 1
            self.save_current_state()

    def evaluate_model(self, data=None):
        """
        Requests all parties to send model evaluations.

        :param data: data to be evaluated by the registered parties' models
        :type data: 'str or 'TaggedDocument'
        """
        if data is None:
            if self.data_handler is None:
                raise FLException('Data or Data Handler must be provided for evaluation.')
            else:
                batch = self.data_handler.get_data()
                if type(batch) is tuple:
                    batch = batch[0]
                data = batch[0]

        lst_parties = self.get_registered_parties()
        lst_evals = self.ph.query_parties({'data': data},
                                          lst_parties,
                                          msg_type=MessageType.EVAL_MODEL,
                                          perc_quorum=self.perc_quorum,
                                          collect_metrics=True,
                                          metrics_party=self.metrics_party)

        logger.info('Finished evaluate model requests.')
        logger.info(lst_evals)
        return lst_evals

    def get_global_model(self):
        """
        Returns last model_update

        :return: model_update
        :rtype: `ModelUpdate`
        """
        return ModelUpdate(weights=self.current_model_weights)

    def get_current_metrics(self):
        """Returns metrics pertaining to current state of fusion handler

        :return: metrics
        :rtype: `dict`
        """
        fh_metrics = {}
        fh_metrics['rounds'] = self.rounds
        fh_metrics['curr_round'] = self.curr_round
        return fh_metrics

    def merge_vocab(self, vocab_lists):
        """
        Combines vocabulary from dictionary of word frequencies

        :param vocab_lists: list of dictionaries containing the vocabulary words used in the training corpus,
        and the number of times they occur
        :type vocab_lists: list<dict>
        :return: A dictionary with all words from each individual vocabulary dictionary
        :rtype dict
        """
        merged_dict = defaultdict(int)
        for d in vocab_lists:
            for word, freq in d.items():
                merged_dict[word] = freq

        return merged_dict

    def set_initial_model(self, merged_dict):
        """
        Sets an initial doc2vec model for parties to start with the same vocabulary and vector space
        :param merged_dict: a dictionary containing all words and frequencies from all parties' training sets
        :type merged_dict: 'dict'
        :return: an initialized doc2vec model
        :rtype: 'gensim.models.doc2vec.Doc2Vec'
        """
        # default values
        epochs = 10
        vector_size = 50
        min_count = 2
        algorithm = 1
        if self.params_local is not None:
            if 'epochs' in self.params_local:
                epochs = self.params_local['epochs']

            if 'vector_size' in self.params_local:
                vector_size = self.params_local['vector_size']

            if 'algorithm' in self.params_local:
                algorithm = self.params_local['algorithm']

            if 'min_count' in self.params_local:
                min_count = self.params_local['min_count']

        model = doc2vec.Doc2Vec(vector_size=vector_size,
                                dm=algorithm,
                                min_count=min_count,
                                epochs=epochs)

        model.build_vocab_from_freq(merged_dict)

        return model
