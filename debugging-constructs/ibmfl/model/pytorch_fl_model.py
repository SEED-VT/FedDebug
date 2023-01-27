"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import torch
import logging
import time
import copy
from importlib import import_module

import numpy as np
from sklearn import metrics
from skorch import NeuralNet
from skorch.callbacks import EpochScoring
from skorch.exceptions import NotInitializedError
from skorch.dataset import CVSplit
from skorch.dataset import Dataset
from skorch.helper import predefined_split

from ibmfl.model.fl_model import FLModel
from ibmfl.model.model_update import ModelUpdate
from ibmfl.util import config
from ibmfl.exceptions import LocalTrainingException, ModelException, \
    FLException, ModelInitializationException

logger = logging.getLogger(__name__)


class PytorchFLModel(FLModel):
    """
    Wrapper class for importing a Pytorch based model
    """

    def __init__(self, model_name,
                 model_spec=None,
                 pytorch_module=None,
                 module_init_params=None,
                 **kwargs):
        """
        Create a `PytorchFLModel` instance from a Pytorch model.
        If pytorch_model is provided, it will use it; otherwise it will take
        the model_spec to create the model.

        :param model_name: String specifying the type of model e.g., Pytorch_NN
        :type model_name: `str`
        :param model_spec: A dictionary specifying path to saved nn.sequence container
        :type model_spec: 'dict'
        :param pytorch_module: uninstantiated pytorch model class
        :type pytorch_module: torch.nn.Module class reference \
        :param module_init_params: A dictionary with the values for the model's init \
        arguments. The key for the init parameters must be prefixed with module__ \
        (ex. if the parameter name is hidden_size, then key must be module__hidden_size)
        :type module_init_params: 'dict'
        :param kwargs: A dictionary contains other parameter settings on \
         to initialize a PyTorch model.
        :type kwargs: `dict`
        """
        super().__init__(model_name, model_spec, **kwargs)

        self.model_type = 'PyTorch'
        self.model = None
        self.module = None
        if pytorch_module is None:
            if model_spec is None or (not isinstance(model_spec, dict)):
                raise ValueError('Initializing model requires '
                                 'a model specification or uninstantiated '
                                 'pytorch model class reference '
                                 'None was provided')
            # In this case we need to recreate the model from model_spec
            self.module = self.load_model_from_spec(model_spec)
        else:
            self.module = pytorch_module
        
        self.optimizer = self.load_optimizer_from_spec(model_spec)
        self.criterion = self.load_loss_criterion_from_spec(model_spec)

        self.model = self.initialize_model(self.module, self.optimizer,
                                           self.criterion, module_init_params)

        if self.use_gpu_for_training and torch.cuda.device_count() > 0:
            if self.num_gpus > torch.cuda.device_count():
                logger.error('Selected number of gpus to use for training exceeds number of available gpus, ' +
                             str(torch.cuda.device_count()) +
                             'Set number of gpus to maximum available on device.')
                self.num_gpus = torch.cuda.device_count()

            device_ids = list(range(self.num_gpus))
            self.model.module_ = torch.nn.DataParallel(self.model.module_, device_ids=device_ids)
            self.model.set_params(device='cuda')
            self.model.module_.to('cuda')

    def initialize_model(self, pytorch_module, optimizer, criterion,
                         module_init_params=None):
        """
        Initializes a pytorch model via skorch library

        :param pytorch_module: uninstantiated pytorch model class
        :type pytorch_module: torch.nn.Module class reference
        :param optimizer: the optimizer to use
        :type optimizer: pytorch optimizer class
        :param criterion: the loss function to use
        :type criterion: pytorch loss function class
        :param module_init_params: A dictionary with the values for the model's init arguments. \
        The key for the init parameters must be prefixed with module__  \
        (ex. if the parameter name is hidden_size, then key must be module__hidden_size)
        :type module_init_params: 'dict'
        :return: an initialized skorch model
        :rtype: 'skorch.NeuralNet'
        """
        if module_init_params is None:
            module_init_params = {}
        model = NeuralNet(
            module=pytorch_module,
            optimizer=optimizer,
            criterion=criterion,
            warm_start=True,
            callbacks=[('valid_acc', EpochScoring(self.valid_acc,
                                                  lower_is_better=False,
                                                  use_caching=True,
                                                  name='test_valid_acc'))],
            **module_init_params,
        )
        model.initialize()
        return model

    def fit_model(self, train_data, fit_params=None, validation_data=None, **kwargs):
        """
        Fits current model with provided training data.

        :param train_data: Training data, a tuple \
        given in the form (x_train, y_train). otherwise, input compatible with skorch.dataset.Dataset
        :type train_data: `(np.ndarray, ”np.ndarray)`
        :param fit_params: (optional) Dictionary with hyperparameters \
        that will be used to call fit function. \
        Hyperparameter parameters should match pytorch expected values \
        e.g., `epochs`, which specifies the number of epochs to be run. \
        If no `epochs` or `batch_size` are provided, a default value \
        will be used (1 and 128, respectively).
        :type fit_params: `dict`
        :return: None
        """
        # Initialized with default values
        batch_size = 128
        epochs = 1
        lr = 0.01
        train_split = None
        optimizer_params = {}
        hyperparams = fit_params.get('hyperparams', {}) if fit_params else {}
        party_params = kwargs.get('local_params', {}) or {} if fit_params else {}

        if hyperparams:
            local_params = hyperparams.get('local', {}) or {}
            training_hp = local_params.get('training', {}) or {}

            epochs = training_hp.get('epochs', epochs)
            batch_size = training_hp.get('batch_size', batch_size)
            lr = training_hp.get('lr', lr)

            if 'validation_split' in training_hp:
                validation_split = training_hp.get('validation_split')
                try:
                    if float(validation_split) != 0:
                        train_split = CVSplit(float(validation_split), random_state=42)
                except (TypeError, ValueError):
                    raise ValueError("Validation split cannot be a NoneType")
            if 'validation_split' in party_params:
                validation_split = party_params.get('validation_split')
                try:
                    if float(validation_split) != 0:
                        train_split = CVSplit(float(validation_split), random_state=42)
                except (TypeError, ValueError):
                    raise ValueError("Validation split cannot be a NoneType")
            if validation_data is not None:
                validation_ds = Dataset(validation_data[0], validation_data[1])
                train_split = predefined_split(validation_ds)

            if 'optimizer_params' in local_params:
                optimizer_params = local_params['optimizer_params']
        self.model.set_params(batch_size=batch_size, lr=lr, train_split=train_split, **optimizer_params)

        try:
            if type(train_data) is tuple:
                # Extract x_train and y_train, by default,
                # label is stored in the last column
                x = train_data[0]
                y = train_data[1]
                self.model.fit_loop(x, y, epochs=epochs)

            else:
                # otherwise, expect that input is a pytorch Dataset generator
                self.model.fit_loop(train_data, epochs=epochs)

        except Exception as e:
            logger.exception(str(e))
            raise LocalTrainingException(
                'Error occurred while performing model.fit')

    def update_model(self, model_update):
        """
        Update model with provided model_update, where model_update
        should be generated according to `PytorchFLModel.get_model_update()`.

        :param model_update: `ModelUpdate` object that contains the weights \
        that will be used to update the model.
        :type model_update: `ModelUpdate`
        :return: None
        """
        if isinstance(model_update, ModelUpdate):
            for p1, p2 in zip(self.get_weights(), model_update.get('weights')):
                p1.data = torch.from_numpy(p2)
                p1.data.requires_grad = True

            if self.use_gpu_for_training and torch.cuda.device_count() > 0:
                self.model.module_.to(self.model.device)
        else:
            raise ValueError('Provided model_update should be of type Model.'
                             'Instead they are:{0}'.format(str(type(model_update))))

    def get_model_update(self):
        """
        Generates a `ModelUpdate` object that will be sent to other entities.

        :return: ModelUpdate
        :rtype: `ModelUpdate`
        """
        weights = self.get_weights(to_numpy=True)
        update = {'weights': weights}

        return ModelUpdate(**update)

    def predict(self, x):
        """
        Perform prediction for a batch of inputs. Note that for classification
        problems, it returns the resulting probabilities.

        :param x: Samples with shape as expected by the model.
        :type x: `np.ndarray`, input compatible with skorch.dataset.Dataset, or pytorch dataloader
        :return: Array of predictions
        :rtype: `np.ndarray`
        """
        if isinstance(x, torch.utils.data.DataLoader):
            return self.predict_generator(x)

        return self.model.predict(x)

    def predict_generator(self, dataloader):
        """
        Performs predictions using a pytorch dataloader

        :param x: A pytorch dataloader with a dataset for predicting
        :type x: 'torch.utils.data.DataLoader'
        :return: Array of predictions
        :rtype: 'np.ndarray'
        """
        predictions = None
        for x_batch, _ in dataloader:
            try:
                y_pred = self.model.evaluation_step(x_batch)
            except ValueError:
                # Skorch 0.11.0 throws a ValueError for line 253 above.
                # This is because Skorch 0.11.0 requires the input
                # of `evaluation_step()` to be a tuple with 2 entries. 
                # Therefore, a dummy tuple of size 2 is created by appending
                # an empty list and it is given to `evaluation_step()`
                batch = (x_batch, [])
                y_pred = self.model.evaluation_step(batch)
            y_pred = y_pred.numpy()
            if predictions is None:
                predictions = y_pred
            else:
                predictions = np.append(predictions, y_pred, 0)

        return predictions

    def evaluate(self, test_dataset, eval_metrics=None, **kwargs):
        """
        Evaluates the model given testing data.
        :param test_dataset: Testing data, a tuple given in the form \
        (x_test, y_test) or a pytorch DataLoader
        :type test_dataset: `np.ndarray`
        :param eval_metrics: a list of sklearn.metric class, or a function for evaluating a pytorch dataloader batch
        :type eval_metrics: 'sklearn.metrics' or 'function'
        :param kwargs: Dictionary of metrics available for the model
        :type kwargs: `dict`
        """

        if type(test_dataset) is tuple:
            x_test = test_dataset[0]
            y_test = test_dataset[1]

            return self.evaluate_model(x_test, y_test, eval_metrics)

        else:
            return self.evaluate_generator_model(test_dataset, eval_metrics)

    def evaluate_model(self, x, y, eval_metrics=None, **kwargs):
        """
        Evaluates the model given x and y.

        :param x:  Samples with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Corresponding labels to x
        :type y: `np.ndarray`
        :param eval_metrics: A list of sklearn.metric class or a functions with the signature \
                        (y_true, y_pred, **kwargs) => float
        :type eval_metrics: `sklearn.metrics` or 'function'
        :return: dictionary of metrics
        :rtype: 'dict'
        """
        if eval_metrics is None or len(eval_metrics) == 0:
            eval_metrics = [metrics.accuracy_score]

        # get names for eval_metrics
        metric_names = [getattr(metric, '__name__', repr(metric))
                        for metric in eval_metrics]

        y_pred = self.predict(x)
        y_pred_exp = np.exp(y_pred)
        y_pred_argmax = np.argmax(y_pred_exp, axis=1)

        try:
            metric_dict = dict(zip(metric_names, eval_metrics))
            # NOTE: Had to replace comprehension with for loop to tackle loss fn case which uses y_pred not argmax
            for metric, fn in metric_dict.items():
                if metric == 'loss':
                    metric_dict[metric] = fn(y, y_pred, self.model)
                else:
                    metric_dict[metric] = fn(y, y_pred_argmax, **kwargs)
            # metric_dict = {metric_name: metric(y, y_pred, **kwargs) for metric_name, metric in
            #                zip(metric_names, eval_metrics)}
            return metric_dict
        except TypeError as exc:
            logger.exception(str(exc))
            raise TypeError("eval_metrics must be an sklearn.eval_metrics class, or a function with the signature "
                            "(y_true, y_pred, **kwargs)'")
        except ValueError as exc:
            logger.exception(exc)
            raise ValueError("arguments not in the correct format for metric")

    def evaluate_generator_model(self, dataloader, eval_metrics=None, **kwargs):
        """
        evaluates the model based on the provided dataloader
        :param dataloader: a pytorch dataloader with test dataset and labels
        :type dataloader: torch.utils.data.DataLoader
        :param eval_metrics: a function for how to evaluate the batched examples and labels
        must have the signature (x_batch, y_batch) => float
        :type eval_metrics: 'function'
        :return: dictionary of metrics
        :rtype: 'dict'
        """
        metric_score = 0
        for x_batch, y_batch in dataloader:
            try:
                y_pred = self.model.evaluation_step(x_batch)
            except ValueError:
                y_pred = self.model.evaluation_step((x_batch, y_batch))

            if eval_metrics is None:
                y_pred = np.exp(y_pred)
                y_pred = np.argmax(y_pred, axis=1)
                equals = (y_pred == y_batch)
                metric_score += torch.mean(equals.type(torch.FloatTensor))

            else:
                try:
                    metric_score += eval_metrics(x_batch, y_batch)

                except TypeError as exc:
                    logger.exception(str(exc))
                    raise TypeError("eval_metrics must be a function "
                                    "with the signature (x_batch, y_batch, **kwargs)'")

        metric_dict = {'metric_score': metric_score / len(dataloader)}

        return metric_dict

    def save_model(self, filename=None, optimizer_filename=None, history_filename=None):
        """
        Save a model to file in the format specific to the backend framework.

        :param filename: Name of the file that contains the model to be loaded.
        :type filename: `str`
        :param optimizer_filename: Name of the file that contains the optimizer to be loaded.
        :type optimizer_filename: `str`
        :param history_filename: Name of the file that contains the model history to be loaded.
        :type history_filename: `str`
        :return: None
        """
        if filename is None:
            file = self.model_name if self.model_name else self.model_type
            filename = '{}_{}.pt'.format(file, time.time())

        f_params = super().get_model_absolute_path(filename)
        f_optimizer = None
        f_history = None
        if optimizer_filename is not None:
            f_optimizer = super().get_model_absolute_path(optimizer_filename)
        if history_filename is not None:
            f_history = super().get_model_absolute_path(history_filename)
        self.model.save_params(
            f_params=f_params, f_optimizer=f_optimizer, f_history=f_history)
        return filename

    def load_model(self, pytorch_module, model_filename, optimizer_filename=None, history_filename=None, optimizer=None,
                   module_init_params=None):
        """
        Loads a model from disk given the specified file_name

        :param pytorch_module: uninstantiated pytorch model class
        :type pytorch_module: torch.nn.Module class reference
        :param model_filename: Name of the file that contains the model to be loaded.
        :type model_filename: `str`
        :param optimizer_filename: Name of the file that contains the optimizer to be loaded.
        :type optimizer_filename: `str`
        :param history_filename: Name of the file that contains the model history to be loaded.
        :type history_filename: `str`
        :param optimizer: the optimizer that should be loaded
        :type optimizer: pytorch optimizer class
        :param module_init_params: A dictionary with the values for the model's init arguments. \
        The key for the init parameters must be prefixed with module__ \
        (ex. if the parameter name is hidden_size, then key must be module__hidden_size)
        :type module_init_params: 'dict'
        :return: None
        """
        f_params = model_filename
        f_optimizer = None
        f_history = None
        if optimizer is not None and optimizer_filename is not None:
            self.optimizer = optimizer
            f_optimizer = optimizer_filename
        if history_filename is not None:
            f_history = history_filename
        model = self.initialize_model(pytorch_module, self.optimizer, self.criterion,
                                      module_init_params=module_init_params)
        model.load_params(f_params=f_params,
                          f_optimizer=f_optimizer, f_history=f_history)

        if self.use_gpu_for_training and torch.cuda.device_count() > 0:
            if self.num_gpus > torch.cuda.device_count():
                logger.error('Selected number of gpus to use for training exceeds number of available gpus, ' +
                             str(torch.cuda.device_count()) +
                             'Set number of gpus to maximum available on device.')
                self.num_gpus = torch.cuda.device_count()
            device_ids = list(range(self.num_gpus))
            model.module_ = torch.nn.DataParallel(model.module_, device_ids=device_ids)
            model.set_params(device='cuda')
            model.module_.to('cuda')

        self.model = model

    def get_weights(self, to_numpy=False):
        """
        Returns the weights of the model

        :param to_numpy; Determines whether the weights should be returned as numpy array, or tensor
        :type to_numpy: `boolean`
        :return: list of model weights
        """
        if self.use_gpu_for_training and torch.cuda.device_count() > 0:
            module = copy.deepcopy(self.model.module_).cpu()
        else:
            module = self.model.module_
        if to_numpy:
            return self.parameters_to_numpy(module.parameters())
        return list(module.parameters())

    def parameters_to_numpy(self, params):
        """
        Transforms parameter tensors to numpy arrays

        :param params: The parameter tensor to be transformed
        :return: numpy array of parameters 
        """
        np_params = []
        for layer in params:
            np_params.append(layer.detach().numpy())
        return np_params

    def load_model_from_spec(self, model_spec):
        """
        Loads model from provided model_spec, where model_spec is a `dict`
        that contains one item: model_spec['model_definition'], which has a
        pointer to the file where an nn.sequence container is saved

        :return: model
        :rtype: `nn.sequence`
        """

        model_file = model_spec['model_definition']
        model_absolute_path = config.get_absolute_path(model_file)
        model = torch.load(model_absolute_path)
        return model

    def load_optimizer_from_spec(self, model_spec):
        """
        Loads optimizer class from provided model_spec, where model_spec is a `dict`
        that contains an item: model_spec['optimizer_class'], which has a
        pointer to the file where an optimizer class object is saved

        :return: model
        :rtype: `torch.optim`
        """
        if model_spec is None or (not isinstance(model_spec, dict)) \
            or model_spec.get('optimizer') is None:
            logger.info('No optimizer found in the config file. ' 
                         'Using default SGD optimizer class.')
            optimizer = torch.optim.SGD
            return optimizer
        else:
            optimizer = model_spec.get('optimizer')
        if not isinstance(optimizer, str):
            raise ModelInitializationException('Optimizer is not specified as a string')
        else:
            try:
                optimizer = getattr(import_module(
                    'torch.optim'), optimizer.split('.')[-1])
            except Exception as e:
                logger.exception(str(e))
                logger.info('Selected optimizer not found. '
                            'Using default SGD optimizer.')
                optimizer = torch.optim.SGD

        return optimizer

    def load_loss_criterion_from_spec(self, model_spec):
        """
        Loads loss criterion from provided model_spec, where model_spec is a `dict`
        that contains an item: model_spec['loss_criterion'], which has a
        pointer to the file where an loss criterion class object is saved

        :return: model
        :rtype: `torch.nn` loss class object
        """
        if model_spec is None or (not isinstance(model_spec, dict)) \
            or model_spec.get('loss_criterion') is None:
            logger.info('No loss criterion found in the config file. ' 
                         'Using default NLLLoss.')
            criterion = torch.nn.NLLLoss
            return criterion
        else:
            criterion = model_spec.get('loss_criterion')
        if not isinstance(criterion, str):
            raise ModelInitializationException('Criterion is not specified as a string')
        else:
            try:
                criterion = getattr(import_module(
                    'torch.nn'), criterion.split('.')[-1])
            except Exception as e:
                criterion = self.criterion
                logger.exception(str(e))
                logger.info('selected criterion not found. '
                      'Using default NLLoss criterion.')
                criterion = torch.nn.NLLLoss
        
        return criterion 

    def get_gradient(self, train_data):
        """
        Returns the gradients for each layer in the model

        :return: gradients
        :rtype: `list`. numpy array list of model's gradients
        """

        x = train_data[0]
        y = train_data[1]
        gradients = []
        y_infer = self.model.module_(torch.from_numpy(x))
        loss = self.model.criterion_(y_infer, torch.from_numpy(y))
        loss.backward()

        for layer in self.model.module_.parameters():
            gradients.append(layer.grad.numpy())

        return gradients

    @staticmethod
    def valid_acc(net, x, y):
        """
        Callback scoring function that returns the validation accuracy during training.
        :param net: The model that will be used
        :param x: the validation data set
        :param y: the training targets
        :return: the accuracy of the validation training pass
        """
        y_pred = net.predict(x)
        y_pred = np.exp(y_pred)
        y_pred = np.argmax(y_pred, axis=1)
        return metrics.accuracy_score(y, y_pred)

    @staticmethod
    def loss(y, y_pred, net):
        loss = net.get_loss(torch.from_numpy(y_pred), torch.from_numpy(y)).numpy()

        return loss

    @staticmethod
    def __expand_linear_layer__(net, layer_name, new_dimensions, layer_idx):
        """
        Expands a linear layer of a pytorch module
        :param net: an instantiated pytorch module
        :type net: `torch.nn.Module`
        :param layer_name: layer variable name which needs to be expanded
        :type layer_name: `str`
        :param new_dimensions: new dimensions
        :type new_dimensions: `list`
        :param layer_idx: layer idx which needs to be expanded. this corresponds with
        the index of `new_dimensions`
        :type layer_idx: `int`
        :return: None
        """

        original_lin_layer = getattr(net, layer_name)

        if not isinstance(original_lin_layer, torch.nn.Linear):
            raise FLException('Received a non-linear layer to expand '
                              'whereas the method expects a linear layer')

        new_ip_dim = original_lin_layer.in_features if layer_idx == 0 else new_dimensions[
            layer_idx-1]
        new_op_dim = new_dimensions[layer_idx]
        bias = original_lin_layer.bias is not None

        new_lin_layer = type(original_lin_layer)(
            in_features=new_ip_dim, out_features=new_op_dim, bias=bias
        )

        setattr(net, layer_name, new_lin_layer)

    def expand_model_by_layer_name(self, new_dimension, layer_name="dense"):
        """
        Expands the current PyTorch models layers with the provided dimensions
        :param new_dimension: new dimensions of the particular `layer_name`
        :type new_dimension: `list`
        :param layer_name: layer name which needs to be expanded
        :type layer_name: `str`
        :return: None
        """

        if new_dimension is None:
            raise FLException('No information is provided for '
                              'the new expanded model. '
                              'Please provide the new dimension of '
                              'the resulting expanded model')

        layer_maps = {
            'dense': {
                'class': torch.nn.Linear,
                'expansion_fn': self.__expand_linear_layer__
            }
        }

        layer_cls = layer_maps[layer_name]['class']
        layer_exp_fn = layer_maps[layer_name]['expansion_fn']
        net = self.model.module_    # get instantiated module
        idx = 0

        for layer_varname, layer in net.named_modules():
            # skip layer if not instance of layer desired to be expanded
            if not isinstance(layer, layer_cls):
                continue

            layer_exp_fn(net, layer_varname, new_dimension, idx)
            idx += 1

        self.model.initialize_optimizer()

    def is_fitted(self):
        """
        Return a boolean value indicating if the model is fitted or not.
        In particular, it calls `skorch.utils.check_is_fitted` to checks
        whether the net is initialized.
        If it has, return True; otherwise return false.

        :return: res
        :rtype: `bool`
        """
        try:
            self.model.check_is_fitted()
        except NotInitializedError:
            return False
        return True

    def get_loss(self, dataset):
        """
        Return the resulting loss computed based on the provided dataset.

        :param train_data: Training data, a tuple \
        given in the form (x_train, y_train). otherwise, input compatible with skorch.dataset.Dataset
        :type train_data: `(np.ndarray, ”np.ndarray)`
        :return: The resulting loss.
        :rtype: `float`
        """

        metrics_list = [PytorchFLModel.loss]
        orig_loss_value = self.evaluate(dataset, eval_metrics=metrics_list)[
                    'loss']

        return orig_loss_value
