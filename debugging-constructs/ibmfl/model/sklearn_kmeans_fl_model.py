"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError

from ibmfl.util import config
from ibmfl.model.sklearn_fl_model import SklearnFLModel
from ibmfl.model.model_update import ModelUpdate
from ibmfl.exceptions import LocalTrainingException, \
    ModelInitializationException, ModelException

logger = logging.getLogger(__name__)


class SklearnKMeansFLModel(SklearnFLModel):
    """
    Wrapper class for sklearn.cluster.KMeans
    """

    def __init__(self, model_name, model_spec, sklearn_model=None, **kwargs):
        """
        Create a `SklearnKMeansFLModel` instance from a
        sklearn.cluster.KMeans model.
        If sklearn_model is provided, it will use it; otherwise it will take
        the model_spec to create the model.

        :param model_name: A name specifying the type of model, e.g., \
        clustering_KMeans
        :type model_name: `str`
        :param model_spec: A dictionary contains model specification
        :type model_spec: `dict`
        :param sklearn_model: Complied sklearn model
        :type sklearn_model: `sklearn.cluster.KMeans`
        :param kwargs: A dictionary contains other parameter settings on \
         to initialize a sklearn KMeans model.
        :type kwargs: `dict`
        """
        super().__init__(model_name, model_spec,
                         sklearn_model=sklearn_model,
                         **kwargs)
        self.model_type = 'Sklearn-KMeans'
        if sklearn_model:
            if not issubclass(type(sklearn_model), KMeans):
                raise ValueError('Compiled sklearn model needs to be provided'
                                 '(sklearn.cluster.KMeans). '
                                 'Type provided: ' + str(type(sklearn_model)))

            self.model = sklearn_model

    def fit_model(self, train_data, fit_params=None, **kwargs):
        """
        Fits current model with provided training data.

        :param train_data: Tuple with first elements being the training data \
        (x_train,)
        :type train_data: `np.ndarray`
        :param fit_params: (optional) Dictionary with hyperparameters that \
        will be used to call sklearn.cluster fit function. \
        Provided hyperparameter should only contains parameters that \
        match sklearn expected values, e.g., `n_clusters`, which provides \
        the number of clusters to fit. \
        If no `fit_params` is provided, default values as defined in sklearn \
        definition are used.
        :return: None
        """

        # Extract x_train by default,
        # Only x_train is extracted since Clustering is unsupervised

        x_train = train_data[0]
        
        hyperparams = fit_params.get('hyperparams', {}) or {} if fit_params else {}
        local_hp = hyperparams.get('local', {}) or {}
        training_hp = local_hp.get('training', {}) or {}

        try:
            self.model.set_params(**training_hp)
        except Exception as err:
            logger.exception(str(err))
            raise LocalTrainingException(
                'Error occurred while setting up model parameters')

        try:
            self.model.fit(x_train)
        except Exception as err:
            logger.info(str(err))
            raise LocalTrainingException(
                'Error occurred while performing model.fit'
            )

    def update_model(self, model_update):
        """
        Update sklearn model with provided model_update, where model_update
        should contain `cluster_centers_` having the same
        dimension as expected by the sklearn.cluster model.
        `cluster_centers_` : np.ndarray, shape (n_clusters, n_features)

        :param model_update: `ModelUpdate` object that contains the \
        cluster_centers vectors that will be used to update the model.
        :type model_update: `ModelUpdate`
        :return: None
        """
        if isinstance(model_update, ModelUpdate):
            cluster_centers_ = model_update.get('weights')

            try:
                if cluster_centers_ is not None:
                    self.model.cluster_centers_ = np.array(cluster_centers_)
            except Exception as err:
                raise LocalTrainingException('Error occurred during '
                                             'updating the model weights. ' +
                                             str(err))
        else:
            raise LocalTrainingException('Provided model_update should be of '
                                         'type ModelUpdate. '
                                         'Instead they are: ' +
                                         str(type(model_update)))

    def get_model_update(self):
        """
        Generates a `ModelUpdate` object that will be sent to other entities.

        :return: ModelUpdate
        :rtype: `ModelUpdate`
        """
        try:
            cluster_centers_ = self.model.cluster_centers_
        except AttributeError:
            cluster_centers_ = None

        return ModelUpdate(weights=cluster_centers_)

    def evaluate(self, test_dataset, **kwargs):
        """
        Evaluates the model given testing data.
        :param test_dataset: Testing data, a tuple given in the form \
        (x_test, test) or a datagenerator of of type `keras.utils.Sequence`, 
        `keras.preprocessing.image.ImageDataGenerator`
        :type test_dataset: `np.ndarray`

        :param kwargs: Dictionary of metrics available for the model
        :type kwargs: `dict`
        """

        if type(test_dataset) is tuple:
            x_test = test_dataset[0]
            y_test = test_dataset[1]

            return self.evaluate_model(x_test, y_test)

        else:
            raise ModelException("Invalid test dataset!")

    def evaluate_model(self, x, y, **kwargs):
        """
        Evaluates the model given test data x and the corresponding labels y.

        :param x: Samples with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Not used for evaluation since this is an unsupervised model
        :type y: `None`
        :param kwargs: Dictionary of model-specific arguments \
        for evaluating models. For example, sample weights accepted \
        by model.score.
        :return: Dictionary with all evaluation metrics provided by \
        specific implementation.
        :rtype: `dict`
        """
        acc = {}

        try:
            acc['score'] = self.model.score(x, **kwargs)
        except NotFittedError:
            logger.info('Model evaluated before fitted. '
                        'Returning accuracy as 0')
            acc['score'] = 0

        return acc

    def save_model(self, filename=None, path=None):
        """
        Save a sklearn.cluster.KMeans model to file in the format specific
        to the framework requirement. Meanwhile, initialize the attribute
        `_n_threads` to 1 if the model does not have it.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path \
        is specified, the model will be stored in the default data location of \
        the library `DATA_PATH`.
        :type path: `str`
        :return: filename
        """
        if not hasattr(self.model, '_n_threads'):
            logger.info("Attribute _n_threads does not exist. "
                        "Setting it to default value.")
            self.model._n_threads = 1

        return super().save_model(filename, path)

    @staticmethod
    def load_model_from_spec(model_spec):
        """
        Loads model from provided model_spec, where model_spec is a `dict`
        that contains the following items: model_spec['model_definition']
        contains the model definition as type sklearn.cluster.KMeans.

        :param model_spec: Model specification contains \
        a compiled sklearn model.
        :type model_spec: `dict`
        :return: model
        :rtype: `sklearn.cluster`
        """
        model = None
        try:
            if 'model_definition' in model_spec:
                model_file = model_spec['model_definition']
                model_absolute_path = config.get_absolute_path(model_file)

                with open(model_absolute_path, 'rb') as f:
                    model = joblib.load(f)

                if not issubclass(type(model), KMeans):
                    raise ValueError('Provided complied model in model_spec '
                                     'should be of type sklearn.cluster.'
                                     'Instead they are: ' + str(type(model)))
        except Exception as ex:
            raise ModelInitializationException('Model specification was '
                                               'badly formed. '+ str(ex))
        return model
