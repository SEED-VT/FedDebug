"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging
import numpy as np


from ibmfl.data.data_handler import DataHandler
from ibmfl.util.datasets import load_mnist

logger = logging.getLogger(__name__)


class MnistKerasAggDataHandler(DataHandler):
    """
    Aggregator's data handler for MNIST dataset in embedding VFL setting.
    """

    def __init__(self, data_config=None):
        super().__init__()

        self.file_name = None
        if data_config is not None:
            if 'npz_file' in data_config:
                self.file_name = data_config['npz_file']

        # load the datasets
        self.y_train = self.load_dataset()

    def get_data(self):
        """
        Not implemented for aggregator side under VFL setting.
        """
        raise NotImplementedError

    def load_dataset(self, nb_points=100):
        """
        Loads the aggregator dataset from a given local path. \
        If no local path is provided, it will download the original MNIST \
        dataset online, and reduce the dataset size to contain \
        100 data points.

        Since this is the aggregator data handler, it will only returns labels.

        :param nb_points: Number of data points to be included in each set if
        no local dataset is provided.
        :type nb_points: `int`
        :return: An array of class labels
        :rtype: `np.ndarray`
        """
        if self.file_name is None:
            (_, y_train), (_) = load_mnist()
            # Reduce datapoints to make test faster
            y_train = y_train[:nb_points]
        else:
            try:
                logger.info('Loaded training data from ' + str(self.file_name))
                y_train = np.load(self.file_name)
            except Exception:
                raise IOError('Unable to load training data from path '
                              'provided in config file: ' +
                              self.file_name)
        return y_train

    def get_labels(self):
        """
        Returns class labels as in onehot encoding format for training.

        :return: An array of class labels
        :rtype: `np.ndarray`
        """
        return np.eye(10)[self.y_train]


class MnistKerasPartyDataHandler(DataHandler):
    """
    Data handler for parties in vertical FL under the embedding setting.
    """

    def __init__(self, data_config=None, channels_first=False):
        super().__init__()

        self.file_name = None
        self.sample_size = 100
        if data_config is not None:
            if 'npz_file' in data_config:
                self.file_name = data_config['npz_file']

            self.sample_size = data_config.get('sample_size') or 100
            self.img_rows = data_config.get('img_rows') or 28
            self.img_cols = data_config.get('img_cols') or 28

        self.channels_first = channels_first

        # load the datasets
        (self.x_train, self.x_test) = self.load_dataset(
            nb_points=self.sample_size)

        # pre-process the datasets
        self.preprocess()

    def get_data(self):
        """
        Gets pre-process mnist training and testing data without labels.

        :return: the training and testing data.
        :rtype: `tuple`
        """
        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def load_dataset(self, nb_points=100):
        """
        Loads the training and testing datasets from a given local path. \
        If no local path is provided, it will download the original MNIST \
        dataset online, and reduce the dataset size to contain \
        500 data points per training and testing dataset and \
        also remove the labels. \
        Because this method \
        is for testing it takes as input the number of datapoints, nb_points, \
        to be included in the training and testing set.

        :param nb_points: Number of data points to be included in each set if
        no local dataset is provided.
        :type nb_points: `int`
        :return: training and testing datasets
        :rtype: `tuple`
        """
        if self.file_name is None:
            (x_train, _), (x_test, _) = load_mnist()
            # Reduce datapoints to make test faster
            x_train = x_train[:nb_points]
            x_test = x_test[:nb_points]
        else:
            try:
                logger.info(
                    'Loaded training data from ' + str(self.file_name))
                data_train = np.load(self.file_name)
                x_train = data_train['x_train']
                x_test = data_train['x_test']
            except Exception:
                raise IOError('Unable to load training data from path '
                              'provided in config file: ' +
                              self.file_name)
        return x_train, x_test

    def preprocess(self):
        """
        Preprocesses the training and testing dataset, \
        e.g., reshape the images according to self.channels_first; \
        convert the labels to binary class matrices.

        :return: None
        """
        if self.channels_first:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 1,
                                                self.img_rows, self.img_cols)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 1,
                                              self.img_rows, self.img_cols)
        else:
            self.x_train = self.x_train.reshape(self.x_train.shape[0],
                                                self.img_rows, self.img_cols, 1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0],
                                              self.img_rows, self.img_cols, 1)


class MnistTFAggDataHandler(MnistKerasAggDataHandler):
    """
    Aggregator's data handler for MNIST dataset in embedding VFL setting.
    """
    def __init__(self, data_config=None):
        super().__init__(data_config)

    def get_labels(self):
        """
        Returns class labels.

        :return: An array of class labels
        :rtype: `np.ndarray`
        """
        return self.y_train, self.y_test

    def shuffle_data(self, seed):
        """
        Create random permutation of indices for random mini-batch selection.

        :param seed: Random seed agreed upon by all parties.
        :type seed: int
        :return: None
        """
        np.random.seed(seed)
        self.indices = np.random.permutation(len(self.y_train))

    def get_labels_batch(self, batch_size, batch_num):
        """
        Returns the labels for specified mini-batch in training dataset

        :param batch_size: Size of the mini-batch
        :type batch_size: int
        :param batch_num: Mini-batch number to return
        :type batch_num: int
        """
        y_train = self.y_train
        indices_train = self.indices[int(batch_size*batch_num):int(batch_size*(batch_num+1))]
        return y_train[indices_train]


class MnistTFPartyDataHandler(MnistKerasPartyDataHandler):
    """
    Data handler for parties in vertical FL under the embedding setting.
    """
    def __init__(self, data_config=None):
        super().__init__(data_config)

    def preprocess(self):
        """
        Reshapes feature set by appending one dimension to x_train and x_test.

        :return: None
        """
        # Add a channels dimension
        import tensorflow as tf
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
        self.x_train = self.x_train[..., tf.newaxis]
        self.x_test = self.x_test[..., tf.newaxis]
        self.indices = None

    def shuffle_data(self, seed):
        """
        Create random permutation of indices for random mini-batch selection.

        :param seed: Random seed agreed upon by all parties.
        :type seed: int
        :return: None
        """
        np.random.seed(seed)
        self.indices = np.random.permutation(len(self.x_train))

    def get_data_batch(self, batch_size, batch_num):
        """
        Get mini-batch of training data.

        :param batch_size: Size of the mini-batch
        :type batch_size: int
        :param batch_num: Mini-batch number to return
        :type batch_num: int
        :return: A dataset structure
        :rtype: `tuple` of `pandas.core.frame.DataFrame`
        """
        if self.indices is None:
            raise ValueError('Must call `shuffle_data` before loading mini-batch.')

        indices_train = self.indices[int(batch_size*batch_num):int(batch_size*(batch_num+1))]
        return self.x_train[indices_train]
