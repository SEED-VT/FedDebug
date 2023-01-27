"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging
import time
import joblib
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from ibmfl.util import config
from ibmfl.model.fl_model import FLModel
from ibmfl.exceptions import ModelInitializationException

logger = logging.getLogger(__name__)


class SklearnFLModel(FLModel):
    """
    Wrapper class for sklearn models.
    """

    def __init__(self, model_name, model_spec, sklearn_model=None, **kwargs):
        """
        Create a `SklearnFLModel` instance from a sklearn model.
        If sklearn_model is provided, it will use it; otherwise it will take
        the model_spec to create the model.

        :param model_name: A name specifying the type of model, e.g., \
                linear_SVM
        :type model_name: `str`
        :param model_spec: A dictionary contains model specification
        :type model_spec: `dict`
        :param sklearn_model: A compiled sklearn model
        :type sklearn_model: `sklearn`
        :param kwargs: A dictionary contains other parameter settings on \
         to initialize a sklearn model.
        :type kwargs: `dict`
        """
        super().__init__(model_name, model_spec, **kwargs)

        if sklearn_model is None:
            if model_spec is None or (type(model_spec) is not dict):
                raise ValueError('Initializing model requires '
                                 'a model specification or '
                                 'compiled sklearn model. '
                                 'None was provided.')
            self.model = self.load_model_from_spec(model_spec)
        else:
            self.model = sklearn_model

        self.is_classification = True if not(model_spec and model_spec.get(
            'is_classification')) else model_spec.get('is_classification')

    def fit_model(self, train_data, fit_params=None, **kwargs):
        """
        Fits current model with provided training data.

        :param train_data: Training data a tuple
        given in the form (x_train, y_train).
        :type train_data: `np.ndarray`
        :param fit_params: (Optional) Dictionary with hyperparameters that \
        will be used to train the model. \
        If the corresponding sklearn fit function is called, \
        the provided hyperparameter should only contains parameters that \
        match sklearn expected values, e.g., `learning_rate`, which provides \
        the learning rate schedule
        :return: None
        """
        raise NotImplementedError

    def update_model(self, model_update):
        """
        Update sklearn model with provided model_update.

        :param model_update: `ModelUpdate` object that contains \
        the required information to update the sklearn model.
        :type model_update: `ModelUpdate`
        :return: None
        """
        raise NotImplementedError

    def get_model_update(self):
        """
        Generates a `ModelUpdate` object that will be sent to other entities.

        :return: ModelUpdate
        :rtype: `ModelUpdate`
        """
        raise NotImplementedError

    def predict(self, x):
        """
        Perform prediction for the given input.

        :param x: Samples with shape as expected by the model.
        :type x: `np.ndarray`
        :return: Array of predictions
        :rtype: `np.ndarray`
        """
        return self.model.predict(x)

    def evaluate_model(self, x, y, **kwargs):
        """
        Evaluates the model given test data x and the corresponding labels y.

        :param x: Samples with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Corresponding true labels to x
        :type y: `np.ndarray`
        :param kwargs: Dictionary of model-specific arguments \
        for evaluating models. For example, sample weights accepted \
        by model.score.
        :return: Dictionary with all evaluation metrics provided by \
        specific implementation.
        :rtype: `dict`
        """
        raise NotImplementedError

    def save_model(self, filename=None, path=None):
        """
        Save a sklearn model to file in the format specific
        to the framework requirement.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path \
        is specified, the model will be stored in the default data location of \
        the library `DATA_PATH`.
        :type path: `str`
        :return: filename
        """
        if filename is None:
            file = self.model_name if self.model_name else self.model_type
            filename = '{}_{}.pickle'.format(file, time.time())

        full_path = super().get_model_absolute_path(filename)

        with open(full_path, 'wb') as f:
            joblib.dump(self.model, f)
        logger.info('Model saved in path: %s.', full_path)
        return filename

    @staticmethod
    def load_model(file_name):
        """
        Load a sklearn from the disk given the filename.

        :param file_name: Name of the file that contains the model to be loaded.
        :type file_name: `str`
        :return: A sklearn model
        :rtype: `sklearn`
        """
        absolute_path = config.get_absolute_path(file_name)

        with open(absolute_path, 'rb') as f:
            model = joblib.load(f)

        return model

    @staticmethod
    def load_model_from_spec(model_spec):
        """
        Loads model from provided model_spec, where model_spec is a `dict`
        that contains the following items: model_spec['model_definition']
        contains the model definition as a specific sklearn model type,
        e.g., `sklearn.linear_model.SGDClassifier`.

        :param model_spec: Model specification contains \
        a compiled sklearn model.
        :type model_spec: `dict`
        :return: model
        :rtype: `sklearn`
        """
        model = None
        try:
            if 'model_definition' in model_spec:
                path = model_spec['model_definition']
                with open(path, 'rb') as f:
                    model = joblib.load(f)
        except Exception as ex:
            raise ModelInitializationException('Model specification was '
                                               'badly formed. '+ str(ex))
        return model

    def is_fitted(self):
        """
        Return a boolean value indicating if the model is fitted or not.

        :return: res
        :rtype: `bool`
        """
        try:
            check_is_fitted(self.model)
        except NotFittedError:
            return False
        return True
