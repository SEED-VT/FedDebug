"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging
import copy
import time
import numpy as np
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from ibmfl.model.fl_model import FLModel
from ibmfl.model.model_update import ModelUpdate
from ibmfl.util import config
from ibmfl.exceptions import LocalTrainingException, FLException
logger = logging.getLogger(__name__)


class Doc2VecFLModel(FLModel):
    """
    Wrapper class for importing a Gensim Doc2Vec model
    """
    def __init__(self, model_name, model_spec=None, doc2vec_model=None, **kwargs):
        """
        Create a `Doc2VecFLModel` instance from a Gensim Doc2vec model.
        If doc2vec_model is provided, it will use it; otherwise it will take
        the model_spec to create the model.

        :param model_name: String specifying the type of model e.g., gensim_doc2vec
        :type model_name: 'str'
        :param model_spec: A dictionary specifying path to saved gensim doc2vec model spec
        :type model_spec: 'dict'
        :param doc2vec_model: An instantiated gensim doc2vec model
        :type: 'gensim.models.doc2vec.Doc2Vec'
        """
        self.model = None
        self.model_type = model_name
        if doc2vec_model is None:
            if model_spec is None or (not isinstance(model_spec, dict)):
                raise ValueError('Initializing model requires '
                                 'a model specification or '
                                 'instantiated gensim doc2vec model class reference '
                                 'None was provided')
            # In this case we need to recreate the model from model_spec
            self.model = self.load_model_from_spec(model_spec)
        else:
            self.model = doc2vec_model

        self.training_corpus = None
        self.total_epochs = None
        self.alpha_delta = None
        self.start_alpha = self.model.alpha
        self.end_alpha = self.model.min_alpha
        self.cur_epoch = 0

    def fit_model(self, train_data, fit_params=None, rounds=1, **kwargs):
        """
        Fits current model with provided training data.

        :param train_data: corpus to train on given in the form of (documents, document_ids).
        otherwise, a list of 'gensim.models.doc2vec.TaggedDocument'
        :type train_data: (list<str>, list<str>)
        :param fit_params: (optional) Dictionary with hyperparameters \
        that will be used to call Keras fit function. \
        Hyperparameter parameters should match keras expected values \
        e.g., `epochs`, which specifies the number of epochs to be run.
        :type fit_params: `dict`
        :return: None
        """
        epochs = 3
        if fit_params is not None and 'hyperparams' in fit_params:
            local_params = fit_params['hyperparams']['local']
            hyperparams = local_params['training']
            epochs = hyperparams.get('epochs', epochs)
            rounds = rounds

        if self.total_epochs is None:
            self.total_epochs = epochs * rounds
            self.alpha_delta = (self.start_alpha - self.end_alpha) / max(self.total_epochs - 1, 1)

        if type(train_data) is tuple:
            self.training_corpus = self.build_tagged_documents(train_data[0], train_data[1])

        else:
            # otherwise, expect that input is a list of tagged Documents
            self.training_corpus = train_data

        if not self.vocabulary_set():
            print("building vocab from train_data")
            self.model.build_vocab(self.training_corpus)

        try:
            for i in range(epochs):
                self.model.min_alpha = self.model.alpha
                self.model.train(self.training_corpus,
                                 total_examples=len(self.training_corpus),
                                 epochs=1)
                self.model.alpha -= self.alpha_delta

        except Exception as e:
            logger.exception(str(e))
            raise LocalTrainingException(
                'Error occurred while performing model.fit')

    def update_model(self, model_update):
        """
        Update model with provided model_update, where model_update
        should be generated according to `Doc2VecFLModel.get_model_update()`.

        :param model_update: `ModelUpdate` object that contains the weights \
        that will be used to update the model.
        :type model_update: `ModelUpdate`
        :return: None
        """
        if isinstance(model_update, ModelUpdate):
            update_weights = model_update.get('weights')

            if isinstance(update_weights, list):
                update_weights = np.asarray(update_weights, dtype=np.float32)

            self.model.trainables.syn1neg = update_weights

        else:
            raise ValueError('Provided model_update should be of type Model.'
                             'Instead they are:{0}'.format(str(type(model_update))))

    def get_model_update(self):
        """
        Generates a `ModelUpdate` object that will be sent to other entities.

        :return: ModelUpdate
        :rtype: `ModelUpdate`
        """
        weights = self.get_weights()
        update = {'weights': weights}

        return ModelUpdate(**update)

    def predict(self, x):
        """
        Perform prediction for test corpus. Note that for word embeddings, this means vectorizing \
        the input data, and returning the resulting vector representation

        :param x: Sample of corpus to be vectorized
        :type x: 'list<str>'
        :return: Array of vector representations of the original text data
        :rtype: `list<np.ndarray`>
        """
        return self.vectorize(x)

    def evaluate(self, x, y=None, topn=10, **kwargs):
        """
        Evaluate the model given sample, x. Note that for word embeddings, this means evaluating \
        the cosine similarity (or other metric) of the sample to other embeddings saved in the model
        :param x: Data sample to be evaluated. Can be a vector, string, or TaggedDocument format
        :type x: 'np.ndarry', 'str', or TaggedDocument
        :param y: N/A for embedding models
        :param topn: The number of results to return from the evaluation
        :type topn: 'int'
        :param kwargs: additional arguments
        :type kwargs: 'dict'
        :return: A list of the evaluation results
        :rtype: list<tuple>
        """
        if isinstance(x, np.ndarray):
            return self.find_most_similar_by_vector(x, topn=topn)

        else:
            return self.find_most_similar(x, topn=topn)

    def save_model(self, filename=None):
        """
        Saves an embedding model under provided filename and path

        :param filename: Determines the name to save the model
        :type filename: str
        :param path: Determines where to save the model
        :type path: str
        :return: filename
        :rtype: str
        """
        if filename is None:
            filename = 'doc2vec_model_{}.model'.format(time.time())

        full_path = super().get_model_absolute_path(filename)
        with open(full_path, "wb") as file:
            self.model.save(file)
            logger.info('Model saved in path: %s.', full_path)
            return filename

    def load_model(self, filename):
        """
        Loads an embedding model using provided filename and path

        :param filename: The filename the model was saved under
        :type filename: str
        :param path: The path where the model was saved
        :type path: str
        :return: None
        """
        full_path = super().get_model_absolute_path(filename)
        try:
            self.model = doc2vec.Doc2Vec.load(full_path)

        except FLException as e:
            logger.exception(str(e))
            raise FLException('Unable to load the provided compiled model!')

    def vectorize(self, data):
        """
        Transforms provided data sample(s) into a vector representation
        :param data: 'list', 'str', or 'TaggedDocument'
        :return: Vector representation(s) of original input data
        """
        if isinstance(data, list):
            return self.vectorize_list(data)

        elif isinstance(data, TaggedDocument):
            return self.model.infer_vector(data[0])

        else:
            tokens = simple_preprocess(data)
            return self.model.infer_vector(tokens)

    def vectorize_list(self, data):
        """
        Transforms provided data into a vector representation
        :param data: list of documents to be vectorized
        :type data: 'list<str>' or 'list<TaggedDocument>'
        :return: A list of vector representations of original input data
        :rtype: `list<np.ndarray`>
        """
        vectors = []

        if isinstance(data[0], TaggedDocument):
            for doc in data:
                v = self.model.infer_vector(doc[0])
                vectors.append(v)

        else:
            for doc in data:
                tokens = simple_preprocess(doc)
                vector = self.model.infer_vector(tokens)
                vectors.append(vector)

        return vectors

    def find_most_similar(self, query_doc, topn=10):
        """
        Searches model for top n items most similar to query

        :param query_doc: A new document used to query for the most similar in the training corpus
        :type query_doc: str
        :param topn: specifies the top n search results to return
        :type topn: 'int'
        :return: list of top n search results and their similarity score
        :rtype: list
        """
        vector = self.vectorize(query_doc)
        return self.model.docvecs.most_similar([vector], topn=topn)

    def find_most_similar_by_id(self, document_id, topn=10):
        """
        Searches model for top n items most similar to a document id in training corpus

        :param document_id: The id of the document to find most similar items to
        :type document_id: int
        :param topn: specifies the top n search results to return
        :type topn: 'int'
        :return: list of top n search results and their similarity score; None if id not found in training corpus
        :rtype: list
        """
        try:
            return self.model.docvecs.most_similar(document_id, topn=topn)
        except Exception as e:
            logger.exception(str(e))
            return None

    def find_most_similar_by_vector(self, vector, topn=10):
        """
        Searches model for top n vectors most similar to a provided vector
        :param vector: A vector to search most similar to
        :type vector: 'numpy array'
        :param topn: specifies the top n search results to return
        :type topn: 'int'
        :return: list of top n search results and their similarity score; None if id not found in training corpus
        :rtype: list
        """
        try:
            return self.model.docvecs.most_similar([vector], topn=topn)
        except Exception as e:
            logger.exception(str(e))
            return None

    def get_vector(self, document_id):
        """
        Returns the vector representation of the document with provided id
        :param document_id: The id of the document to retrieve its vector representation
        :type document_id: 'str'
        :return: the vectorized document with matching id
        :rtype: 'numpy array'
        """
        try:
            return self.model.docvecs[document_id]
        except Exception as e:
            logger.exception(str(e))
            return None

    def is_fitted(self):
        """
        Return a boolean value indicating if the model is fitted or not.
        In particular, check if the model has weights.
        If it has, return True; otherwise return false.

        :return: res
        :rtype: `bool`
        """
        try:
            weights = self.model.trainables.syn1neg
        except AttributeError:
            return False
        return weights.any()

    def get_loss(self, dataset):
        """
        Return the resulting loss computed based on the provided dataset.

        :param dataset: Provided dataset, a tuple given in the form \
        (x_test, y_test) or a datagenerator
        :type dataset: `np.ndarray`
        :return: The resulting loss.
        :rtype: `float`
        """
        raise NotImplementedError

    def get_weights(self):
        """
        Returns the Doc2Vec model's inner neural network's weights

        :return: A copy of the model's neural network weights
        :rtype: numpy array
        """
        return copy.deepcopy(self.model.trainables.syn1neg)

    def get_vector_size(self):
        """
        Returns the dimensionality of the model's vectors

        :return: The model's vector size
        :rtype: 'int'
        """
        return self.model.vector_size

    def build_tagged_documents(self, documents, document_ids):
        """
        Creates a list of TaggedDocument objects from a list of strings and ids
        :param documents: A list of texts representing documents
        :type documents: list<str>
        :param document_ids: A list of string ids corresponding to each document in the list
        :type document_ids: list<str>
        :return: A list of TaggedDocument objects
        :rtype: list<TaggedDocument>
        """
        tagged_documents = []
        for doc, doc_id in zip(documents, document_ids):
            tokens = simple_preprocess(doc)
            tagged_documents.append(TaggedDocument(words=tokens, tags=[str(doc_id)]))

        return tagged_documents

    def load_model_from_spec(self, model_spec):
        """
        Loads model from provided model_spec, where model_spec is a `dict`
        that contains one item: model_spec['model_definition'], which has a
        pointer to the file where a serialized gensim doc2vec model is saved
        :param model_spec: A dictionary that contains a pointer to a saved gensim doc2vec model
        :type model_spec: A d
        :return: model
        :rtype: 'gensim.models.doc2vec.Doc2Vec'
        """

        model_file = model_spec['model_definition']
        model_absolute_path = config.get_absolute_path(model_file)
        return doc2vec.Doc2Vec.load(model_absolute_path)

    def vocabulary_set(self):
        """
        Determines whether the doc2vec model's vocabulary has been set before training

        :return: True if the vocabulary has been built; False otherwise
        :rtype: 'bool'
        """
        return hasattr(self.model.trainables, 'syn1neg')

    def get_loss(self):
        """
        loss tallying not implemented in gensim doc2vec, so method returns None

        :return: None
        """
        return None
