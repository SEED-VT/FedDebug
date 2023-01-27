"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging

import numpy as np

from ibmfl.data.data_handler import DataHandler
from ibmfl.util.datasets import load_leaf_femnist

logger = logging.getLogger(__name__)


class FemnistKerasDataHandler(DataHandler):
    def __init__(self, data_config=None, channels_first=False):
        super().__init__()
        self.file_name = None
        if data_config is not None:
            if 'npz_file' in data_config:
                self.file_name = data_config['npz_file']
            if 'data_folder' in data_config:
                self.data_folder = data_config['data_folder']
        self.channels_first = channels_first

        # load the datasets
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_dataset()

        # pre-process the datasets
        self.preprocess()

    def get_data(self):
        """
        Gets pre-processed femnist training and testing data.

        :return: training data, testing data
        :rtype: `tuple`, `tuple`
        """
        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def load_dataset(self, nb_points=500):
        """
        Loads the local dataset from a provided local data path. \
        If no local data path is provided, it loads the \
        femnist dataset from LEAF, and reduces the dataset size to contain \
        500 data points per training and testing dataset.

        :param nb_points: Number of data points to be included in each set if
        no local dataset is provided.
        :type nb_points: `int`
        :return: training and testing datasets
        :rtype: `tuple`
        """
        if self.file_name is None:
            (x_train, y_train), (x_test, y_test) = load_leaf_femnist(download_dir=self.data_folder)
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
        num_classes = 62
        img_rows, img_cols = 28, 28
        if self.channels_first:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 1,
                                                img_rows, img_cols)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 1,
                                              img_rows, img_cols)
        else:
            self.x_train = self.x_train.reshape(self.x_train.shape[0],
                                                img_rows, img_cols, 1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], img_rows,
                                              img_cols, 1)

        logger.info('x_train shape: {}'.format(self.x_train.shape))
        logger.info('Train Samples: {}'.format(self.x_train.shape[0]))
        logger.info('Test Samples: {}'.format(self.x_test.shape[0]))

        # convert class vectors to binary class matrices
        self.y_train = np.eye(num_classes)[self.y_train]
        self.y_test = np.eye(num_classes)[self.y_test]


class FemnistDPKerasDataHandler(FemnistKerasDataHandler):

    def __init__(self, data_config=None):
        super().__init__(data_config)
        self.y_train = np.argmax(self.y_train, axis=1)
        self.y_test = np.argmax(self.y_test, axis=1)
