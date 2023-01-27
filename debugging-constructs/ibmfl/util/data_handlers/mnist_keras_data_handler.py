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


class MnistKerasDataHandler(DataHandler):
    """
    Data handler for MNIST dataset.
    """

    def __init__(self, data_config=None, channels_first=False):
        super().__init__()

        self.file_name = None
        if data_config is not None:
            if 'npz_file' in data_config:
                self.file_name = data_config['npz_file']
        self.channels_first = channels_first

        # load the datasets
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_dataset()

        # pre-process the datasets
        self.preprocess()

    def get_data(self):
        """
        Gets pre-process mnist training and testing data.

        :return: the training and testing data.
        :rtype: `tuple`
        """
        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def load_dataset(self, nb_points=500):
        """
        Loads the training and testing datasets from a given local path. \
        If no local path is provided, it will download the original MNIST \
        dataset online, and reduce the dataset size to contain \
        500 data points per training and testing dataset. \
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
            (x_train, y_train), (x_test, y_test) = load_mnist()
            # Reduce datapoints to make test faster
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
        num_classes = 10
        img_rows, img_cols = 28, 28

        if self.channels_first:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, img_rows, img_cols)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, img_rows, img_cols)
        else:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], img_rows, img_cols, 1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], img_rows, img_cols, 1)

        # convert class vectors to binary class matrices
        self.y_train = np.eye(num_classes)[self.y_train]
        self.y_test = np.eye(num_classes)[self.y_test]


class MnistDPKerasDataHandler(MnistKerasDataHandler):
    """
    Data handler for MNIST dataset with differential privacy.
    Only changes from MNISTDataHandler is removal of one-hot encoding of target variable.
    Currently the differentially private SGD optimizer expects single dimensional y.
    """

    def __init__(self, data_config=None):
        super().__init__(data_config)
        self.y_train = np.argmax(self.y_train, axis=1)
        self.y_test = np.argmax(self.y_test, axis=1)


class MnistKerasDataGenerator(DataHandler):
    """
    Sample data handler for MNIST dataset in the form of Datagenerator class.
    """

    def __init__(self, data_config):
        super().__init__()
        from keras.utils import np_utils
        from keras.preprocessing.image import ImageDataGenerator
        # load the original MNIST dataset
        (X_train, y_train), (X_test, y_test) = load_mnist()
        # reshape the dataset
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_train = X_train.astype('float32')
        X_train /= 255
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        X_test = X_test.astype('float32')
        X_test /= 255

        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        train_gen = ImageDataGenerator(rotation_range=8,
                                       width_shift_range=0.08,
                                       shear_range=0.3,
                                       height_shift_range=0.08,
                                       zoom_range=0.08)
        test_gen = ImageDataGenerator(rotation_range=8,
                                      width_shift_range=0.08,
                                      shear_range=0.3,
                                      height_shift_range=0.08,
                                      zoom_range=0.08)

        self.train_datagenerator = train_gen.flow(
            X_train, y_train, batch_size=64)
        self.test_datagenerator = test_gen.flow(X_test, y_test, batch_size=64)

    def get_data(self):
        return self.train_datagenerator, self.test_datagenerator

    def set_batch_size(self, batch_size):
        self.train_datagenerator.set_batch_size(batch_size)


class MnistTFDataHandler(MnistKerasDataHandler):
    """
       Data handler for MNIST dataset.
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
        self.x_train = self.x_train[..., tf.newaxis].astype('float32')
        self.x_test = self.x_test[..., tf.newaxis].astype('float32')
