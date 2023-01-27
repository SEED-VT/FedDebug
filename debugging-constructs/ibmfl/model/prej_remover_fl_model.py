"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import logging
import numpy as np
import joblib
import importlib
import time
import aif360


from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from ibmfl.model.fl_model import FLModel
from ibmfl.model.model_update import ModelUpdate
from ibmfl.exceptions import LocalTrainingException, \
    ModelInitializationException, ModelException
pr = importlib.import_module('aif360.algorithms.inprocessing.kamfadm-2012ecmlpkdd.fadm.lr.pr')
LRwPRType4 = pr.LRwPRType4

logger = logging.getLogger(__name__)

class PrejRemoverFLModel(FLModel):

    def __init__(self, model_name, model_spec, **kwargs):
        """
        Create a `SklearnPRFLModel` instance.

        :param model_name: A name specifying the type of model, e.g.,
        linear_SVM
        :type model_name: `str`
        """

        self.eta = model_spec['eta']
        self.C = model_spec['C']
        self.coef = None
        self.model = LRwPRType4()

    def fit_model(self, train_data, fit_params=None, **kwargs):
        """
        Learns the LRwPRType4 model.

        :param dataset: training dataset
        :type model_name: `np.array`
        :return: Prejudice Remover model
        :rtype: LRwPRType4
        """
        x = train_data[0]
        y = train_data[1]
        model = LRwPRType4(eta=self.eta, C=self.C)
        model.fit(x, y, 1, itype=0)
        self.model = model

        return self

    def update_model(self, model_update):
        """
        Update model with provided model_update, where model_update
        should contains `coef_` and `intercept_`.

        `coef_` : np.ndarray, shape (1, n_features) if n_classes == 2
        else (n_classes, n_features)
        `intercept_` : np.ndarray, shape (1,) if n_classes == 2 else (n_classes,)

        :param model_update: `ModelUpdate` object that contains the coef_ and \
        the intercept vectors that will be used to update the model.
        :type model_update: `ModelUpdate`
        :return: None
        """
        if isinstance(model_update, ModelUpdate):
            weights = model_update.get('weights')
            coef = np.array(weights)
            intercept = np.array(weights)[-1].reshape(1,)

            try:
                self.model.coef_ = coef
                self.model.intercept_ = intercept
            except Exception as e:
                raise LocalTrainingException('Error occurred during '
                                             'updating the model weights. ' +
                                             str(e))
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
        model = self.model
        if hasattr(model, 'coef_'):
            coef = model.coef_
        else:
            coef = None
        if hasattr(model, 'intercept_'):
            intercept = model.intercept_
        else:
            intercept = None
        if intercept is None:
            w = coef
        else:
            w = np.append(coef, intercept)
        intercept = np.reshape(intercept, [1, 1])
        w = np.append(coef, intercept)

        return ModelUpdate(weights=w.tolist(),
                           coef=coef,
                           intercept=intercept)

    def predict(self, x):
        """
        Perform prediction for the given input.

        :param x: Samples with shape as expected by the model.
        :type x: `np.ndarray`
        :return: Array of predictions
        :rtype: `np.ndarray`
        """
        return self.model.predict(x)

    def predict_pr(self, train_data, fit_params=None):
        """
        Predicts class labels for testing dataset via trained LRwPRType4 model.

        :param dataset: testing dataset
        :type dataset: `np.array`
        :return: labels
        :rtype: `np.array`
        """
        x = train_data[0]
        p = self.model.predict_proba(x)
        labels = []
        for i in range(p.shape[0]):
            c = np.argmax(p[i, :])
            labels = np.append(labels, c)

        return labels

    def get_classes(self, labels=None):
        """
        Returns an array of shape (n_classes,). If self.classes is not None,
        return self.classes, else obtains the array based on provided labels.

        :param labels: Provided class labels to obtain the array of classes.
        :type labels: `numpy.ndarray`
        :return: An array of shape `(n_classes,)`.
        :rtype: `numpy.ndarray`
        """
        if self.classes is not None:
            return self.classes
        elif hasattr(self.model, "classes_"):
            return self.model.classes_
        elif labels is not None:
            return np.unique(labels)

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

            return self.evaluate_model(x_test, y_test, **kwargs)

        else:
            raise ModelException("Invalid test dataset!")

    def evaluate_model(self, x, y, **kwargs):
        acc = {}
        acc['score'] = self.model.score(x, y, **kwargs)
        return acc

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
