"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import joblib
import numpy as np
import logging

from diffprivlib.models import GaussianNB

from ibmfl.exceptions import ModelInitializationException, ModelException
from ibmfl.model.model_update import ModelUpdate
from ibmfl.model.sklearn_fl_model import SklearnFLModel

logger = logging.getLogger(__name__)


class NaiveBayesFLModel(SklearnFLModel):
    """
    Wrapper class for diffprivlib.models.GaussianNB classifier, which itself
    is an implementation of sklearn.naive_bayes.GaussianNB with differential
    privacy.
    """

    def __init__(self, model_name, model_spec, model=None, **kwargs):
        """
        Create a `NaiveBayesFLModel` instance from a diffprivlib.models.GaussianNB
        model. If model is provided, it will use it; otherwise it will take
        the model_spec to create the model.

        :param model_name: A name specifying the type of model, e.g., \
        naive_bayes
        :type model_name: `str`
        :param model_spec: A dictionary contains model specification
        :type model_spec: `dict`
        :param model: Compiled GaussianNB model
        :type model: `diffprivlib.models.GaussianNB`
        :param kwargs: A dictionary contains other parameter settings on \
         to initialize a Naive Bayes model.
        :type kwargs: `dict`
        """
        if model:
            if not issubclass(type(model), GaussianNB):
                raise ValueError('Compiled GaussianNB needs to be provided'
                                 '(diffprivlib.models.GaussianNB). '
                                 'Type provided' + str(type(model)))

        super().__init__(model_name,
                         model_spec,
                         sklearn_model=model,
                         **kwargs)

        self.old_vals = {"theta": None, "var": None, "class_count": None}
        self.model_trained = False
        self.model_type = 'Sklearn-NaiveBayes'

    def fit_model(self, train_data, fit_params=None, **kwargs):
        """
        Fits current model with provided training data.

        :param train_data: Training data a tuple given in the form \
        (x_train, y_train).
        :type train_data: `np.ndarray`
        :param fit_params: Not used, present here for API consistency by \
        convention.
        :return: None
        """
        self.model.fit(train_data[0], train_data[1])
        self.model_trained = True

    def update_model(self, model_update, **kwargs):
        """
        Update GaussianNB model with provided model_update.

        :param model_update: `ModelUpdate` object that contains \
        the required information to update the GaussianNB model.
        :type model_update: `ModelUpdate`
        :return: None
        """
        if model_update.get("class_count") is None:
            return

        self.model.class_count_ = model_update.get("class_count")
        self.model.theta_ = model_update.get("theta")

        try:
            self.model.sigma_ = model_update.get("var")
        except AttributeError:
            self.model.var_ = model_update.get("var")

        self._store_old_values(model_update.get("theta"), model_update.get("var"),
                               model_update.get("class_count"))

    def get_model_update(self):
        """
        Generates a `ModelUpdate` object that will be sent to other entities.

        :return: ModelUpdate
        :rtype: `ModelUpdate`
        """
        if (not self.model_trained or
                not hasattr(self.model, "class_count_") or
                np.all(self.old_vals['class_count'] ==
                       self.model.class_count_)):
            return ModelUpdate(theta=None, var=None, class_count=None)

        try:
            model_var = self.model.var_
        except AttributeError:
            model_var = self.model.sigma_

        if self.old_vals['class_count'] is None:
            return ModelUpdate(theta=self.model.theta_,
                               var=model_var,
                               class_count=self.model.class_count_)

        class_count_update = self.model.class_count_ - \
            self.old_vals['class_count']

        nm = self.model.class_count_[:, np.newaxis]
        n = self.old_vals['class_count'][:, np.newaxis]
        m = class_count_update[:, np.newaxis]

        theta_update = self.model.theta_ * nm - self.old_vals['theta'] * n
        theta_update = theta_update / m

        var_update = nm * model_var - n * self.old_vals['var'] - \
            (n * m) / nm * (self.old_vals['theta'] - theta_update) ** 2

        return ModelUpdate(theta=theta_update,
                           var=var_update,
                           class_count=class_count_update)

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
        :param y: Corresponding true labels to x
        :type y: `np.ndarray`
        :param kwargs: Dictionary of model-specific arguments for evaluating \
        models. For example, sample weights accepted by model.score.
        :return: Dictionary with all evaluation metrics provided by \
        specific implementation.
        :rtype: `dict`
        """
        acc = {'score': self.model.score(x, y, **kwargs)}
        return acc

    def _store_old_values(self, theta, var, class_count):
        """
        Store old values of training parameters to allow reconstruction of
        parameters for newly-trained examples when model update is sought.
        """
        self.old_vals = {"theta": theta,
                         "var": var,
                         "class_count": class_count}

    @staticmethod
    def load_model_from_spec(model_spec):
        """
        Loads model from provided model_spec, where model_spec is a `dict`
        that contains the following items: model_spec['model_definition']
        contains the model definition as type diffprivlib.GaussianNB.

        :param model_spec: Model specification contains \
        a complied sklearn model.
        :param model_spec: `dict`
        :return: model
        :rtype: `sklearn.cluster`
        """
        model = None
        try:
            if 'model_definition' in model_spec:
                path = model_spec['model_definition']
                with open(path, 'rb') as f:
                    model = joblib.load(f)

                if not issubclass(type(model), GaussianNB):
                    raise ValueError('Provided complied model in model_spec '
                                     'should be of type GaussianNB.'
                                     'Instead they are:' + str(type(model)))
        except Exception as ex:
            raise ModelInitializationException('Model specification was '
                                               'badly formed. ', str(ex))
        return model
