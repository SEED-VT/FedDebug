"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging
import time
import json
import numpy as np
import tensorflow as tf
import inspect
from pandas.core.frame import DataFrame
# if tf.__version__ != "2.1.0":
# raise ImportError("This function requires TensorFlow v2.1.0.")

from ibmfl.util import config
from ibmfl.util import fl_metrics
from ibmfl.model.fl_model import FLModel
from ibmfl.model.model_update import ModelUpdate
from ibmfl.exceptions import FLException, LocalTrainingException, \
    ModelException
from tensorflow.keras import backend as K

logger = logging.getLogger(__name__)


class TensorFlowFLModel(FLModel):
    """
    Wrapper class for importing tensorflow models.
    """

    def __init__(self, model_name, model_spec, tf_model=None, **kwargs):
        """
        Create a `TensorFlowFLModel` instance from a tensorflow model.\
        If `tf_model` is provided, it will use it; otherwise it will take\
        the model_spec to create the model.\
        Assumes the `tf_model` passed as argument is compiled.

        :param model_name: String specifying the type of model e.g., tf_CNN
        :type model_name: `str`
        :param model_spec: Specification of the `tf_model`
        :type model_spec: `dict`
        :param tf_model: Compiled TensorFlow model.
        :type tf_model: `tf.keras.Model`
        """

        super().__init__(model_name, model_spec, **kwargs)
        if tf_model is None:
            if model_spec is None or (not isinstance(model_spec, dict)):
                raise ValueError('Initializing model requires '
                                 'a model specification or '
                                 'compiled TensorFlow model. '
                                 'None was provided')
            # In this case we need to recreate the model from model_spec
            self.model = self.load_model_from_spec(model_spec)
        else:
            if not issubclass(type(tf_model), tf.keras.Model):
                raise ValueError('Compiled TensorFlow model needs to be '
                                 'provided of type `tensorflow.keras.models`.'
                                 ' Type provided: ' + str(type(tf_model)))

            if self.use_gpu_for_training and self.num_gpus >= 1:
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    self.model = tf_model
            else:
                self.model = tf_model

        self.model_type = 'TensorFlow-2.1.0'
        # Default values for local training
        self.batch_size = 128
        self.epochs = 1
        self.validation_split = 0
        self.steps_per_epoch = None
        self.is_classification = True if not (model_spec and model_spec.get(
            'is_classification')) else model_spec.get('is_classification')

    def fit_model(self, train_data, fit_params=None, validation_data=None, **kwargs):
        """
        Fits current model with provided training data.

        :param train_data: Training data, a tuple\
        given in the form (x_train, y_train).
        :type train_data: `np.ndarray`
        :param fit_params: (optional) Dictionary with hyperparameters\
        that will be used to call fit function.\
        Hyperparameter parameters should match  expected values\
        e.g., `epochs`, which specifies the number of epochs to be run.\
        If no `epochs` or `batch_size` are provided, a default value\
        will be used (1 and 128, respectively).
        :type fit_params: `dict`
        :return: None
        """
        fit_args = self.get_fit_args(fit_params, **kwargs)
        if validation_data is not None:
            fit_args.pop('validation_split', None)
            fit_args['validation_data'] = validation_data

        logger.info('Training hps for this round => '
                    'batch_size: {}, epochs {}, steps_per_epoch {}'
                    .format(fit_args.get('batch_size'),
                    fit_args.get('epochs'), fit_args.get('steps_per_epoch')))

        try:
            if type(train_data) is tuple and type(train_data[0]) is np.ndarray:
                # Extract x_train and y_train, by default,
                # label is stored in the last column
                x = train_data[0]
                y = train_data[1]
                self.model.fit(x, y, **fit_args)
            else:
                if isinstance(train_data, (tf.keras.utils.Sequence)) and \
                    hasattr(train_data, 'set_batch_size'):
                    train_data.set_batch_size(fit_args.get('batch_size'))

                fit_args.pop('batch_size', None)
                fit_args.pop('validation_split', None)
                self.model.fit(train_data, **fit_args)

        except Exception as e:
            logger.exception(str(e))
            raise LocalTrainingException(
                'Error occurred while performing model.fit')

    def update_model(self, model_update):
        """
        Update TensorFlow model with provided model_update, where model_update \
        should be generated according to \
        `TensorFlowFLModel.get_model_update()`

        :param model_update: `ModelUpdate` object that contains the weights \
        that will be used to update the model.
        :type model_update: `ModelUpdate`, `numpy array`, or 'list'
        :return: None
        """
        if isinstance(model_update, ModelUpdate):
            w = model_update.get("weights")
            self.model.set_weights(w)
        else:
            raise LocalTrainingException('Provided model_update should be of '
                                         'type ModelUpdate. '
                                         'Instead they are: ' +
                                         str(type(model_update)))

    def update_model_gradient(self, grads):
        """
        Update TensorFlow model with provided gradients,
        where grads is a vector of gradients to be applied
        based on current optimizer.

        :param grads: Numpy array or list of gradient arrays that contains \
        gradients to update model weights.
        :type grads: `numpy array`, or 'list'
        :return: None
        """
        if isinstance(grads, np.ndarray):
            grads = self.reshape_to_model(grads, 'model')
            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        elif isinstance(grads, list):
            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        else:
            raise LocalTrainingException('Provided gradients should be of '
                                         'type np.ndarray or list of np.ndarrays. '
                                         'Instead they are: ' +
                                         str(type(grads)))

    def get_model_update(self):
        """
        Generates a `ModelUpdate` object that will be sent to other entities.

        :return: ModelUpdate
        :rtype: `ModelUpdate`
        """
        w = self.model.get_weights()
        return ModelUpdate(weights=w)

    def predict(self, x, **kwargs):
        """
        Perform prediction for a batch of inputs. Note that for classification \
        problems, it returns the resulting probabilities.

        :param x: Samples with shape as expected by the model.
        :type x: `np.ndarray`
        :param kwargs: Dictionary of tf-specific arguments.
        :type kwargs: `dict`

        :return: Array of predictions
        :rtype: `np.ndarray`
        """
        return self.model.predict(x, **kwargs)

    def evaluate(self, test_dataset, **kwargs):
        """
        Evaluates the model given testing data.

        :param test_dataset: Testing data, a tuple given in the form \
        (x_test, y_test) or a datagenerator of type `keras.utils.Sequence`, \
        `keras.preprocessing.image.ImageDataGenerator`
        :type test_dataset: `np.ndarray`
        :param kwargs: Dictionary of metrics available for the model
        :type kwargs: `dict`
        :return: metrics
        :rtype: `dict`
        """

        if type(test_dataset) is tuple:
            x_test = test_dataset[0]
            y_test = test_dataset[1]

            return self.evaluate_model(x_test, y_test, **kwargs)

        else:
            return self.evaluate_generator_model(
                test_dataset, **kwargs)

    def evaluate_model(self, x, y, batch_size=128, **kwargs):
        """
        Evaluates the model given x and y.

        :param x: Samples with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Corresponding labels to x
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param kwargs: Dictionary of metrics available for the model
        :type kwargs: `dict`
        :return: metrics
        :rtype: `dict`
        """

        metrics = self.model.evaluate(
            x, y, batch_size=batch_size, verbose=0, **kwargs)
        names = self.model.metrics_names
        dict_metrics = {}
        additional_metrics = {}
        if type(metrics) == list:
            for metric, name in zip(metrics, names):
                # metric = metric.item()
                if name == 'accuracy':
                    dict_metrics['acc'] = round(metric, 2)
                dict_metrics[name] = metric
        else:
            dict_metrics[names[0]] = metrics

        y_pred = self.predict(x, batch_size=batch_size)
        if self.is_classification:
            additional_metrics = fl_metrics.get_eval_metrics_for_classificaton(
                y, y_pred)
        else:
            additional_metrics = fl_metrics.get_eval_metrics_for_regression(
                y, y_pred)

        logger.info(additional_metrics)
        dict_metrics = {**dict_metrics, **additional_metrics}
        logger.info(dict_metrics)

        return dict_metrics

    def evaluate_generator_model(self, test_generator, **kwargs):
        """
        Evaluates the model based on the provided data generator.

        :param test_generator: Testing datagenerator of type \
        `keras.utils.Sequence`, or \
        `keras.preprocessing.image.ImageDataGenerator`.
        :type test_generator: `ImageDataGenerator` or `keras.utils.Sequence`
        :return: metrics
        :rtype: `dict`
        """

        steps = self.steps_per_epoch
        if steps in kwargs:
            steps = kwargs.get('steps')

        metrics = self.model.evaluate_generator(
            test_generator, steps=steps)
        names = self.model.metrics_names
        dict_metrics = {}
        additional_metrics = {}

        if type(metrics) == list:
            for metric, name in zip(metrics, names):
                # metric = metric.item()
                if name == 'accuracy':
                    dict_metrics['acc'] = round(metric, 2)
                dict_metrics[name] = metric
        else:
            dict_metrics[names[0]] = metrics

        return dict_metrics

    @staticmethod
    def load_model(file_name, custom_objects={}):
        """
        Loads a model from disk given the specified file_name

        :param file_name: Name of the file that contains the model to be loaded.
        :type file_name: `str`
        :return: TensorFlow model loaded to memory
        :rtype: `tf.keras.models.Model`
        """
        try:
            model = tf.keras.models.load_model(
                file_name, custom_objects=custom_objects)
        except Exception as ex:
            logger.exception(str(ex))
            logger.error(
                'Loading model via tf.keras.models.load_model failed!')
        return model

    def save_model(self, filename=None):
        """
        Save a model to file in the format specific to the backend framework.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :return: filename
        :rtype `string`
        """
        if filename is None:
            file = self.model_name if self.model_name else self.model_type
            filename = '{}'.format(file)

        full_path = super().get_model_absolute_path(filename)
        self.model.save(full_path)
        logger.info('Model saved in path: %s.', full_path)
        return filename

    @staticmethod
    def model_from_json_via_tf_keras(json_file_name, custom_objects={}):
        """
        Loads a model architecture from disk via tf.keras \
        given the specified json file name.

        :param json_file_name: Name of the file that contains \
        the model architecture to be loaded.
        :type json_file_name: `str`
        :param custom_objects: Dictionary of custom objects required for loading arch
        :type custom_objects: `dict`
        :return: tf.keras model with only model architecture loaded to memory
        :rtype: `tf.keras.models.Model`
        """
        model = None
        json_file = open(json_file_name, 'r')
        f = json_file.read()
        json_file.close()
        try:
            model = tf.keras.models.model_from_json(
                f, custom_objects=custom_objects)
        except Exception as ex:
            logger.error(
                'Loading model via tf.keras.models.model_from_json failed! ')

        return model

    def load_model_from_spec(self, model_spec):
        """
        Loads model from provided model_spec, where model_spec is a `dict` \
        that contains the following items: \
            'model_definition': the path where the tf model is stored, \
                usually in a `SavedModel` format.
        :return: model
        :rtype: `keras.models.Model`
        """
        custom_objects = {}
        if 'custom_objects' in model_spec:

            custom_objects_config = model_spec['custom_objects']
            for custom_object in custom_objects_config:
                key = custom_object['key']
                value = custom_object['value']
                path = custom_object['path']
                custom_objects[key] = config.get_attr_from_path(
                    path, value)

        if 'model_definition' in model_spec:
            try:
                model_file = model_spec['model_definition']
                model_absolute_path = config.get_absolute_path(model_file)

                if self.use_gpu_for_training:
                    strategy = tf.distribute.MirroredStrategy()
                    with strategy.scope():
                        model = TensorFlowFLModel.load_model(
                            model_absolute_path, custom_objects=custom_objects)
                else:
                    model = TensorFlowFLModel.load_model(
                        model_absolute_path, custom_objects=custom_objects)

            except Exception as ex:
                logger.exception(str(ex))
                raise FLException('Failed to load TensorFlow model!')
        else:

            if self.use_gpu_for_training:
                strategy = tf.distribute.MirroredStrategy()
                model = self.load_model_from_architecture(
                    model_spec, custom_objects)

            else:
                model = self.load_model_from_architecture(
                    model_spec, custom_objects)

        return model

    def load_model_from_architecture(self, model_spec, custom_objects):
        """
        Loads model from provided model_spec, where model_spec is a `dict` \
        that contains the following items: \
            'model_architecture': the path where the tf model is stored, \
                usually in a `SavedModel` format. \
            'model_weights': the path to where the tf model weights are saved \
            'compile_model_options': attributes used to compile model.
        :param model_spec: Disctionary of spec provided by the user
        :type model_spec: `dict`
        :param custom_objects: Dictionary of custom objects required for loading arch
        :type custom_objects: `dict`
        :return: model
        :rtype: `keras.models.Model`
        """

        try:
            model = TensorFlowFLModel.model_from_json_via_tf_keras(
                model_spec['model_architecture'], custom_objects=custom_objects)

            if model is None:
                logger.error('An acceptable compiled model should be of type '
                             'tensorflow.keras.models!')
        except Exception as ex:
            logger.error(str(ex))
            raise FLException('Unable to load the provided uncompiled model!')

            # Load weights from provided path
        if 'model_weights' in model_spec:
            model.load_weights(model_spec['model_weights'])

        if 'compile_model_options' in model_spec:
            # Load compile options:
            try:
                compiled_options = model_spec['compile_model_options']
                optimizer = self.get_custom_attribute(
                    compiled_options.get('optimizer'))
                loss = self.get_custom_attribute(compiled_options.get('loss'))
                metrics = self.get_custom_attribute(
                    compiled_options.get('metrics'))
                metrics = [metrics] if not isinstance(
                    metrics, list) else metrics
                model.compile(optimizer=optimizer,
                              loss=loss,
                              metrics=metrics)
            except Exception as ex:
                logger.exception(str(ex))
                logger.exception(
                    'Failed to compiled the TensorFlow.keras model.')
        else:
            raise ModelException('Failed to compile keras model, '
                                 'no compile options provided.')

        return model

    def get_gradient(self, train_data, wrt='trainable_weights'):
        """
        Compute the gradient with the provided dataset at the current local \
        model's weights. Can calculate the gradient with respect to the trainable \
        weights, or with respect to the input to the model.

        :param train_data: Training data, a tuple \
        given in the form (x_train, y_train).
        :type train_data: `np.ndarray`
        :param wrt: Specifying which variable the gradient is computed for.
        The default 'trainable_weights' will calculate the gradient of the trainable weights.
        Setting wrt to 'model_input' will calculate the gradient with respect to the input to the model.
        :type wrt: `str`
        :return: gradients
        :rtype: `list` of `tf.Tensor`
        """
        try:
            x, y = train_data[0], train_data[1]
        except Exception as ex:
            logger.exception(str(ex))
            raise FLException('Provided dataset has incorrect format. '
                              'It should be a tuple in the form of '
                              '(x_train, y_train).')

        # Check if input length matches input shape of model
        if hasattr(self.model.layers[0], 'input_shape') and x.shape[1:] != self.model.layers[0].input_shape[1:]:
            raise FLException('Input data does not match model input shape.'
                              f'Input data shape: {x.shape[1:]}.'
                              f'Model input shape: {self.model.layers[0].input_shape[1:]}.')
        if wrt == 'trainable_weights':
            with tf.GradientTape() as tape:
                predictions = self.model(x, training=True)
                loss = self.model.loss(y, predictions)
                gradients = tape.gradient(loss, self.model.trainable_variables)
        elif wrt == 'model_input':
            # TODO see https://github.com/tensorflow/tensorflow/issues/36596
            x_input = tf.Variable(tf.convert_to_tensor(x))
            with tf.GradientTape() as tape:
                tape.watch(x_input)
                predictions = self.model(x_input, training=False)
                loss = self.model.loss(y, predictions)
                gradients = [tape.gradient(loss, x_input)]
        else:
            raise FLException('Gradient can only be computed '
                              'with respect to trainable_weights or '
                              'model_input. However, ' + str(wrt) +
                              'was provided.')
        return gradients

    def expand_model_by_layer_name(self, new_dimension, layer_name="dense"):
        """
        Expand the current Keras model with provided dimension of
        the hidden layers or model weights.
        This method by default expands the dense layer of
        the current neural network.
        It can be extends to expand other layers specified by `layer_name`,
        for example, it can be use to increase the number of CNN filters or
        increase the hidden layer size inside LSTM.

        :param new_dimension: New number of dimensions for \
        the fully connected layers
        :type new_dimension: `list`
        :param layer_name: layer's name to be expanded
        :type layer_name: `str`
        :return: None
        """
        if new_dimension is None:
            raise FLException('No information is provided for '
                              'the new expanded model. '
                              'Please provide the new dimension of '
                              'the resulting expanded model.')
        try:
            model_config = json.loads(self.model.to_json())
        except NotImplementedError:
            raise ModelException(
                "Please construct the model config for models in "
                "`SavedModel` format. "
                "Details about how to construct the model config can be found"
                " in TensorFlowFLModel tutorials.")
        except Exception as ex:
            logger.exception(str(ex))
            raise FLException("Error occurred during extracting "
                              "the model architecture.")
        i = 0

        for layer in model_config['config']['layers']:
            # find the specified layers
            if 'class_name' in layer and \
                    layer['class_name'].strip().lower() == layer_name:
                layer['config']['units'] = new_dimension[i]
                i += 1

        custom_obj = {
            self.model.__class__.__name__: self.model.__class__
        }

        try:
            new_model = tf.keras.models.model_from_json(
                json.dumps(model_config), custom_objects=custom_obj)
        except Exception as ex:
            logger.exception(str(ex))
            raise FLException("Error occurred during loading model from "
                              "the new config.")

        metrics = self.model.metrics_names
        if 'loss' in metrics:
            metrics.remove('loss')
        if not self.use_gpu_for_training or self.num_gpus == 1:
            new_model.compile(optimizer=self.model.optimizer,
                              loss=self.model.loss,
                              metrics=metrics)
        else:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                new_model.compile(optimizer=self.model.optimizer,
                                  loss=self.model.loss,
                                  metrics=metrics)

        self.model = new_model

    def is_fitted(self):
        """
        Return a boolean value indicating if the model is fitted or not. \
        In particular, check if the tensorflow model has weights. \
        If it has, return True; otherwise return false.

        :return: res
        :rtype: `bool`
        """
        try:
            self.model.get_weights()
        except Exception:
            return False
        return True

    def get_loss(self, dataset):
        """
        Return the resulting loss computed based on the provided dataset.

        :param dataset: Provided dataset, a tuple given in the form \
        (x_test, y_test) or a datagenerator of type `keras.utils.Sequence`, \
        `keras.preprocessing.image.ImageDataGenerator`.
        :type dataset: `np.ndarray`
        :return: The resulting loss.
        :rtype: `float`
        """
        if 'loss' not in self.model.metrics_names:
            self.model.metrics_names.append('loss')
        res = self.evaluate(dataset)

        if 'loss' in res:
            loss = round(res['loss'], 2)
            return loss
        else:
            raise FLException(
                "Loss is not listed in the model's metrics_names.")

    def get_custom_attribute(self, attr):
        """
        Load compiled options which are provided as config.
        :param attr: Attribute config provided in config
        :type attr: dict or key
        :return: Attribute loaded and returned back for compilation
        :rtype: `str` or python attr

        """
        if attr is None:
            raise ModelException("Invalid Model config exception")

        if isinstance(attr, dict):
            try:

                value = attr.get('value')
                path = attr.get('path')
                args = attr.get('args') if 'args' in attr else {}
                attribute = config.get_attr_from_path(
                    path, value)

            except Exception as ex:
                logger.error(
                    "Error occurred while loading the custom attribute!")
                logger.error("Custom attribute : " + attr)
                logger.error()

            logger.debug(type(attribute))

            if inspect.isclass(attribute):
                return attribute(**args)
            else:
                return attribute

        else:
            return attr

    def get_fit_args(self, global_params, **kwargs):

        fit_args = {}
        local_params = kwargs.get('local_params', {}) or {} if global_params else {}
        hyperparams = global_params.get('hyperparams', {}) or {} if global_params else {}
        local_hp = hyperparams.get('local', {}) or {}
        training_hp = local_hp.get('training', {}) or {}
        optimizer_hp = local_hp.get('optimizer', {}) or {}

        lr = optimizer_hp.get('lr', None)
        if lr:
            K.set_value(self.model.optimizer.learning_rate, lr)
        logger.info("Learning rate of optimizer is set as {}".format(
            self.model.optimizer.learning_rate))

        validation_split = training_hp.get('validation_split',
                                           self.validation_split)
        try:
            if float(validation_split) != 0:
                fit_args['validation_split'] = float(validation_split)
        except (TypeError, ValueError):
            raise ValueError('Validation split cannot be a NoneType')
        if 'validation_split' in local_params:
            validation_split = local_params.get('validation_split')
            try:
                if float(validation_split) != 0:
                    fit_args['validation_split'] = float(validation_split)
            except (TypeError, ValueError):
                raise ValueError('Validation split cannot be a NoneType')

        fit_args['batch_size'] = training_hp.get('batch_size', self.batch_size)
        if 'batch_size' in local_params:
            fit_args['batch_size'] = local_params.get('batch_size')

        fit_args['epochs'] = training_hp.get('epochs', self.epochs)
        if 'epochs' in local_params:
            fit_args['epochs'] = local_params.get('epochs')

        fit_args['steps_per_epoch'] = training_hp.get('steps_per_epoch',
                                                      self.steps_per_epoch)
        if 'steps_per_epoch' in local_params:
            fit_args['steps_per_epoch'] = local_params.get('steps_per_epoch')

        return fit_args

    def get_model_output(self, x, learning_phase_flag=False):
        """
        Return the resulting last layer output of the model when passing the
        provided set of features.

        :param x: The provided set of features to obtain the last layer output.
        :type x: `np.ndarray`
        :param learning_phase_flag: The keras.backend.learning_phase flag to \
        indicate if this is training or inference phase, \
        as 'some Keras layers (e.g. Dropout, BatchNormalization) behave \
        differently at training time and testing time' quoted from Keras.io.
        :type learning_phase_flag: `boolean`
        :return: The resulting last layer output
        :rtype: `list`
        """
        x_train = self._data_format_check(x)
        last_layer = self.model(x_train, training=learning_phase_flag)
        return [last_layer]

    def get_gradient_of_output(self, x):
        """
        Compute the gradient of output with respect to model weights.
        Uses tape.jacobian to obtain per-sample gradients.

        :param x: The provided set of features to compute gradients.
        :type x: `np.ndarray`
        :return: The resulting gradient.
        :rtype: `list`
        """
        with tf.GradientTape(persistent=True) as tape:
            preds = self.model(x, training=False)

        gradients = tape.jacobian(preds, self.model.trainable_variables)
        gradients = self.reshape_to_model(gradients, 'flatten')
        return gradients

    def reshape_to_model(self, vals, reshape_type):
        """
        Reshapes input values into shape that matches the model weights,
        or flattens input values that are currently in the shape of model weights.

        :param vals: Input values to be reshaped.
        :type vals: `np.ndarray`
        :param reshape_type: Indicates if input should be flattened or placed into model weights shape.
        :type vals: `str`
        :return: Reshaped values.
        :rtype: `np.ndarray`
        """
        if reshape_type == 'flatten':
            l = []
            for val in vals:
                batch_size = val.shape[0]
                embedding_length = val.shape[1]
                # Flatten only axis 2 and higher,
                # maintain the batch size and embedding length dimensions
                l.append(tf.reshape(val, (batch_size, embedding_length, -1)))
            return np.concatenate(l, axis=2)
        else:
            l = []
            elem = 0
            for weights in self.model.trainable_variables:
                end = elem + tf.size(weights)
                l.append(vals[elem:end].reshape(weights.shape))
                elem = end
            return l

    @staticmethod
    def _data_format_check(data):
        """
        Check the data format: if the data in format of `DataFrame`,
        covert it `numpy.ndarray`
        """
        return data.to_numpy(dtype=float) if isinstance(data, DataFrame) else data
