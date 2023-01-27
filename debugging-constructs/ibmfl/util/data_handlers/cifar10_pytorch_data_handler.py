"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging
import numpy as np
from ibmfl.data.data_handler import DataHandler
from ibmfl.util.datasets import load_cifar10

logger = logging.getLogger(__name__)


class Cifar10PytorchDataHandler(DataHandler):
    def __init__(self, data_config=None):
        super().__init__()
        self.file_name = None
        if data_config is not None:
            if 'npz_file' in data_config:
                self.file_name = data_config['npz_file']
        # load the datasets
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_dataset()

        # pre-process the datasets
        self.preprocess()

    def get_data(self):
        """
        Gets pre-processed CIFAR10 training and testing data.

        :return: training data
        :rtype: `tuple`
        """
        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def load_dataset(self, nb_points=500):
        """
        Loads the training and testing datasets from a given local path. \
        If no local path is provided, it will download the original cifar10 \
        dataset from Keras.datasets, and reduce the dataset size to contain \
        500 data points per training and testing dataset.

        :param nb_points: Number of data points to be included in each set if
        no local dataset is provided.
        :type nb_points: `int`
        :return: training and testing datasets
        :rtype: `tuple`
        """
        if self.file_name is None:
            (x_train, y_train), (x_test, y_test) = load_cifar10()
            x_train = x_train[:nb_points]
            y_train = y_train[:nb_points]
            x_test = x_test[:nb_points]
            y_test = y_test[:nb_points]
        else:
            try:
                logger.info('Loaded training data from ' + str(self.file_name))
                data_train = np.load(self.file_name)
                x_train = data_train['x_train']
                y_train = data_train['y_train']
                x_test = data_train['x_test']
                y_test = data_train['y_test']
            except Exception:
                raise IOError('Unable to load training data from path '
                              'provided in config file: ' +
                              self.file_name)
        return (x_train, y_train), (x_test, y_test)

    def preprocess(self):
        """
        Preprocesses the training and testing dataset, \
        e.g., reshape the images according to self.channels_first; \
        convert the labels to binary class matrices.

        :return: None
        """
        img_rows, img_cols = 32, 32
        self.x_train = self.x_train.astype('float32').\
            reshape(self.x_train.shape[0], 3, img_rows, img_cols)
        self.x_test = self.x_test.astype('float32').\
            reshape(self.x_test.shape[0], 3, img_rows, img_cols)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')

        self.y_train = self.y_train.astype('int64')
        self.y_test = self.y_test.astype('int64')
        print('y_train shape:', self.y_train.shape)
        print(self.y_train.shape[0], 'train samples')
        print(self.y_test.shape[0], 'test samples')
