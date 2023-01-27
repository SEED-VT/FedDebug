"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import abc
import os
from pathlib import Path
import logging

import ibmfl.envs as fl_envs

logger = logging.getLogger(__name__)


class FLModel(abc.ABC):
    """
    Base class for all FLModels. This includes supervised and unsupervised ones.
    """

    def __init__(self, model_name, model_spec, **kwargs):
        """
        Initializes an `FLModel` object

        :param model_name: String describing the model e.g., keras_cnn
        :type model_name: `str`
        :param model_spec: Specification of the the model
        :type model_spec: `dict`
        :param kwargs: Dictionary of model-specific arguments.
        :type kwargs: `dict`
        """
        self.model_name = model_name
        # reserve to specify the model type by FLModel class
        self.model_type = None
        self.model_spec = model_spec if model_spec else {}
        self.use_gpu_for_training = False
        if 'info' in kwargs and kwargs['info'] is not None:
            gpu_config = kwargs['info'].get('gpu')
            # load GPU setting
            if gpu_config:
                self.use_gpu_for_training = True
                if 'selection' in gpu_config and gpu_config.get('selection') == 'auto':
                    self.num_gpus, self.gpu_ids = self.get_gpu_config_auto(gpu_config)
                else:
                    self.num_gpus, self.gpu_ids = self.get_gpu_config_manual(gpu_config)

                self.num_gpus, self.gpu_ids = self.get_gpu_config_auto(gpu_config)

                os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_ids

    def get_custom_metrics_pre(self):
        """
        Return pre-training metrics to log via metrics manager.

        :return: Dictionary of custom metrics to pass through straight to metrics manager.
        :rtype: `dict`
        """

        return {}

    def get_custom_metrics_post(self):
        """
        Return post-training metrics to log via metrics manager.

        :return: Dictionary of custom metrics to pass through straight to metrics manager.
        :rtype: `dict`
        """

        return {}

    def get_train_result(self):
        raise NotImplementedError

    @abc.abstractmethod
    def fit_model(self, train_data, **kwargs):
        """
        Fits current model with provided training data.

        :param train_data: Training data.
        :type train_data: Data structure containing training data, \
        varied based on model types.
        :param kwargs: Dictionary of model-specific arguments for fitting, \
        e.g., hyperparameters for local training, information provided \
        by aggregator, etc.
        :type kwargs: `dict`
        :return: None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update_model(self, model_update, **kwargs):
        """
        Updates model using provided `model_update`. Additional arguments
        specific to the model can be added through `**kwargs`

        :param model_update: Model with update. This is specific to each model \
        type e.g., `ModelUpdateSGD`. The specific type should be checked by \
        the corresponding FLModel class.
        :type model_update: `ModelUpdate`
        :param kwargs: Dictionary of model-specific arguments.
        :type kwargs: `dict`
        :return: None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_model_update(self):
        """
        Generate a `ModelUpdate` object specific to the FLModel being trained.
        This object will be shared with other parties.

        :return: Model Update. Object specific to model being trained.
        :rtype: `ModelUpdate`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Samples with shape as expected by the model.
        :type x: Data structure as expected by the model
        :param kwargs: Dictionary of model-specific arguments.
        :type kwargs: `dict`

        :return: Predictions
        :rtype: Data structure the same as the type defines labels \
        in testing data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, test_dataset, batch_size=128, **kwargs):
        """
        Evaluates model given the test dataset.
        Multiple evaluation metrics are returned in a dictionary

        :param test_dataset: Provided test dataset to evalute the model.
        :type test_dataset: `tuple` of `np.ndarray` or data generator.
        :param batch_size: batch_size: Size of batches.
        :type batch_size: `int`
        :param kwargs: Dictionary of model-specific arguments.
        :type kwargs: `dict`
        :return: Dictionary with all evaluation metrics provided by specific \
        implementation.
        :rtype: `dict`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, filename, path=None):
        """
        Save a model to file in the format specific to the backend framework.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path \
        is specified, the model will be stored in \
                     the default data location of the library `DATA_PATH`.
        :type path: `str`
        :return: filename
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, filename):
        """
        Load model from provided filename

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is specified, \
        the model will be stored in the default data location of the library `DATA_PATH`.
        :type path: `str`
        :return: model
        """
        raise NotImplementedError

    def is_fitted(self):
        """
        Return a boolean value indicating if the model is fitted or not.

        :return: res
        :rtype: `bool`
        """
        raise NotImplementedError

    @staticmethod
    def get_model_absolute_path(filename):
        """Construct absolute path using MODEL_DIR env variable

        :param filename: Name of the file
        :type filename: `str`
        :return: absolute_path; constructed absolute path using model_dir
        :rtype: `str``

        """
        if "MODEL_DIR" in os.environ:
            model_path = Path(os.environ["MODEL_DIR"])
        else:
            model_path = Path(fl_envs.model_directory)

        model_path.mkdir(parents=True, exist_ok=True)
        absolute_path = model_path.joinpath(filename)
        return str(absolute_path)

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


    def get_gpu_config_auto(self, gpu_config):
        """
        Extract num of gpus and gpu ids automatically using gpu util.
        :param gpu_config: Provided gpu config
        :type gpu_config:  `dict`
        """
        num_gpus = 1
        gpu_ids = ''

        import GPUtil
        try:
            _gpu_ids = GPUtil.getFirstAvailable(order='memory')
            num_gpus = gpu_config.get('num_gpus') if 'num_gpus' in gpu_config else len(_gpu_ids)
            gpu_ids = str(_gpu_ids[0:num_gpus]).replace('[', '').replace(']', '')
        except RuntimeError:
            logger.error("No gpu was detected on the node.")
        return num_gpus, gpu_ids

    def get_gpu_config_manual(self, gpu_config):
        """
        Extract num of gpus and gpu ids from user provided gpu configuration.
        :param gpu_config: Provided gpu config
        :type gpu_config:  `dict`
        """
        num_gpus = 1
        gpu_ids = ''
        if not gpu_config:
            logger.error("No gpu configuration provided for this job.")

        if 'num_gpus' not in gpu_config and 'gpu_id' not in gpu_config:
            logger.error("No valid gpu configuration provided for this job.")

        if 'gpu_id' in gpu_config:
            gpu_ids = gpu_config.get('gpu_ids')
            num_gpus = len(gpu_ids.split(','))
        
        elif 'num_gpus' in gpu_config:
            if not isinstance(gpu_config.get('num_gpus'), int):
                logger.error(
                    'Provided number of gpu should be of type int. '
                    'Instead it is of type ' +
                    str(type(gpu_config.get('num_gpus'))) +
                    'Set number of gpu to default value 1.')
                self.num_gpus = 1
            else:
                self.num_gpus = gpu_config.get('num_gpus')
        

        return num_gpus, gpu_ids
