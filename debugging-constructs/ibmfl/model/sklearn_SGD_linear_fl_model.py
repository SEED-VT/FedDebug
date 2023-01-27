"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging
import joblib
import numpy as np
import time

from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


from ibmfl.util import fl_metrics
from ibmfl.util import config
from ibmfl.model.sklearn_fl_model import SklearnFLModel
from ibmfl.model.model_update import ModelUpdate
from ibmfl.exceptions import LocalTrainingException, \
    ModelInitializationException, ModelException

logger = logging.getLogger(__name__)


class SklearnSGDFLModel(SklearnFLModel):
    """
    Wrapper class for sklearn.linear_model.SGDClassifier and
    sklearn.linear_model.SGDRegressor.
    """

    def __init__(self, model_name, model_spec, sklearn_model=None, **kwargs):
        """
        Create a `SklearnSGDFLModel` instance from a
        sklearn.linear_model.SGDClassifier or a
        sklearn.linear_model.SGDRegressor model.
        If sklearn_model is provided, it will use it; otherwise it will take
        the model_spec to create the model.

        :param model_name: A name specifying the type of model, e.g., \
        linear_SVM
        :type model_name: `str`
        :param model_spec: A dictionary contains model specification
        :type model_spec: `dict`
        :param sklearn_model: Compiled sklearn model
        :type sklearn_model: `sklearn.linear_model`
        :param kwargs: A dictionary contains other parameter settings on \
         to initialize a sklearn.linear_model model.
        :type kwargs: `dict`
        """
        super().__init__(model_name, model_spec,
                         sklearn_model=sklearn_model,
                         **kwargs)
        self.model_type = 'Sklearn-linear-SGD'
        if sklearn_model:
            if not issubclass(type(sklearn_model), (SGDClassifier,
                                                    SGDRegressor)):
                raise ValueError('Compiled sklearn model needs to be provided'
                                 '(sklearn.linear_model). '
                                 'Type provided: ' + str(type(sklearn_model)))

            self.model = sklearn_model
            
        if type(self.model) is SGDClassifier:
            self.is_classification = True
        elif type(self.model) is SGDRegressor:
            self.is_classification = False

        # checking if classes_ is provided for classification problem
        if isinstance(self.model, SGDClassifier) and \
                hasattr(self.model, "classes_"):
            # check the type
            if isinstance(self.model.classes_, np.ndarray) or \
                    self.model.classes_ is None:
                self.classes = self.model.classes_
            else:
                raise ValueError(
                    "Provided SGDClassifier model has wrong type of "
                    "`classes_` attribute. "
                    "`classes_` should be of type `numpy.ndarray`. "
                    "Instead it is of type " + str(type(self.model.classes_)))
        else:
            self.classes = None

    def fit_model(self, train_data, fit_params=None, **kwargs):
        """
        Fits current model with provided training data.

        :param train_data: Training data a tuple \
        given in the form (x_train, y_train).
        :type train_data: `np.ndarray`
        :param fit_params: (Optional) Dictionary with hyperparameters that \
        will be used to call sklearn.linear_model fit function. \
        Provided hyperparameter should only contains parameters that \
        match sklearn expected values, e.g., `learning_rate`, which provides \
        the learning rate schedule. \
        If no `learning_rate` or `max_iter` is provided, a default value will \
        be used ( `optimal` and `1`, respectively).
        :return: None
        """
        # Default values
        max_iter = 1
        warm_start = True

        # Extract x_train and y_train, by default,
        # label is stored in the last column
        x = train_data[0]
        y = train_data[1]

        hyperparams = fit_params.get('hyperparams', {}) or {} if fit_params else {}
        sample_weight = fit_params.get('sample_weight', None) if fit_params else {}

        local_hp = hyperparams.get('local', {}) or {}
        training_hp = local_hp.get('training', {}) or {}
        
        try:
            
            if 'max_iter' not in training_hp:
                training_hp['max_iter'] = max_iter
                logger.info('Using default max_iter: ' + str(max_iter))

            # set warm_start to True
            training_hp['warm_start'] = warm_start
            logger.info('Set warm_start as ' + str(warm_start))

            for key, val in training_hp.items():
                self.model.set_params(**{key: val})
        except Exception as e:
            logger.exception(str(e))
            raise LocalTrainingException(
                'Error occurred while setting up model parameters')

        try:
            # Compute the classes based on labels if classification problem
            if isinstance(self.model, SGDClassifier):
                if self.classes is None:
                    logger.warning(
                        "Obtaining class labels based on local dataset. "
                        "This may cause failures during aggregation "
                        "when parties have distinctive class labels. ")
                    self.classes = self.get_classes(labels=y)
                # sklearn `partial_fit` uses max_iter = 1,
                # manually start the local training cycles
                for iter in range(training_hp['max_iter']):
                    logger.info("Local training epoch " + str(iter+1) + ":")
                    self.model.partial_fit(x, y,
                                           classes=self.classes,
                                           sample_weight=sample_weight)
            elif isinstance(self.model, SGDRegressor):
                for iter in range(training_hp['max_iter']):
                    logger.info("Local training epoch " + str(iter + 1) + ":")
                    self.model.partial_fit(x, y, sample_weight=sample_weight)
            else:
                raise NotImplementedError
        except Exception as e:
            logger.info(str(e))
            raise LocalTrainingException(
                'Error occurred while performing model.fit'
            )

    def update_model(self, model_update):
        """
        Update sklearn model with provided model_update, where model_update
        should contains `coef_` and `intercept_` having the same dimension
        as expected by the sklearn.linear_model.
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

            if isinstance(self.model, SGDClassifier):
                coef = np.array(weights)[:, :-1]
                intercept = np.array(weights)[:, -1]
            elif isinstance(self.model, SGDRegressor):
                coef = np.array(weights)[:-1]
                intercept = np.array(weights)[-1].reshape(1,)
            else:
                raise LocalTrainingException(
                    "Expecting scitkit-learn model of "
                    "type either "
                    "sklearn.linear_model.SGDClassifier "
                    "or sklearn.linear_model.SGDRegressor."
                    "Instead provided model is of type "
                    + str(type(self.model)))
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

        coef = self.model.coef_
        intercept = self.model.intercept_
        if isinstance(self.model, SGDClassifier):
            n_classes = len(self.get_classes())
            # intercept is of shape (1,) if n_classes == 2 else (n_classes,)
            if n_classes == 2:
                intercept = np.reshape(intercept, [1, 1])
            else:
                intercept = np.reshape(intercept, [n_classes, 1])
            w = np.append(coef, intercept, axis=1)
        elif isinstance(self.model, SGDRegressor):
            w = np.append(coef, intercept)
        else:
            raise LocalTrainingException("Expecting scitkit-learn model of "
                                         "type either "
                                         "sklearn.linear_model.SGDClassifier "
                                         "or sklearn.linear_model.SGDRegressor."
                                         "Instead provided model is of type "
                                         + str(type(self.model)))

        return ModelUpdate(weights=w.tolist(),
                           coef=self.model.coef_,
                           intercept=self.model.intercept_)

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
        :param kwargs: Optional sample weights accepted by model.score
        :return: score, mean accuracy on the given test data and labels
        :rtype: `dict`
        """
        acc = {}
        acc['score'] = self.model.score(x, y, **kwargs)

        y_pred = self.predict(x, **kwargs)

        if self.is_classification:
            acc['acc'] = acc['score']
            additional_metrics = fl_metrics.get_eval_metrics_for_classificaton(
                y, y_pred)
        else:
            additional_metrics = fl_metrics.get_eval_metrics_for_regression(
                y, y_pred)

        dict_metrics = {**acc, **additional_metrics}

        return dict_metrics

    def predict_proba(self, x):
        """
        Perform prediction for the given input.

        :param x: Samples with shape as expected by the model.
        :type x: `np.ndarray`
        :return: Array of predictions
        :rtype: `np.ndarray`
        """
        return self.model.predict_proba(x)

    def get_classes(self, labels=None):
        """
        Returns an array of shape (n_classes,). If self.classes is not None,
        return self.classes, else obtains the array based on provided labels.

        :param labels: Provided class labels to obtain the array of classes.
        :type labels: `numpy.ndarray`
        :return: An array of shape `(n_classes,)`.
        :rtype: `numpy.ndarray`
        """
        if isinstance(self.model, SGDRegressor):
            raise ValueError(
                "SGDRegressor is used for regression problems and "
                "does not have `classes_`.")
        if self.classes is not None:
            return self.classes
        elif hasattr(self.model, "classes_"):
            return self.model.classes_
        elif labels is not None:
            return np.unique(labels)
        else:
            raise NotFittedError(
                "The scikit-learn model has not been initialized with "
                "`classes_` attribute, "
                "please either manually specify `classes_` attribute as "
                "an array of shape (n_classes,) or "
                "provide labels to obtain the array of classes. ")

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

        # check if the classes_ attribute exists for SGDClassifier
        # this attribute is required for prediction and scoring
        if isinstance(self.model, SGDClassifier) and \
                not hasattr(self.model, "classes_"):
            logger.warning(
                "The classification model to be saved has no `classes_` "
                "attribute and cannot be used for prediction!")

        with open(full_path, 'wb') as f:
            joblib.dump(self.model, f)
        logger.info('Model saved in path: %s.', full_path)
        return filename

    @staticmethod
    def load_model_from_spec(model_spec):
        """
        Loads model from provided model_spec, where model_spec is a `dict`
        that contains the following items: model_spec['model_definition']
        contains the model definition as
        type sklearn.linear_model.SGDClassifier
        or sklearn.linear_model.SGDRegressor.

        :param model_spec: Model specification contains \
        a compiled sklearn model.
        :param model_spec: `dict`
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

                if not issubclass(type(model), (SGDClassifier, SGDRegressor)):
                    raise ValueError('Provided compiled model in model_spec '
                                     'should be of type sklearn.linear_model.'
                                     'Instead it is: ' + str(type(model)))
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
            check_is_fitted(self.model, ("coef_", "intercept_"))
        except NotFittedError as ex:
            logger.warning(
                "Model has no attribute `coef_` and `intercept_`, "
                "and hence is not fitted yet.")
            return False
        return True

