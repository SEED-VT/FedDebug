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


class MnistSklearnDataHandler(DataHandler):
    """
    Data handler for mnist dataset to train a SGDClassifier on scikit-learn.
    """
    def __init__(self, data_config=None):
        super().__init__()
        self.file_name = None
        if data_config is not None:
            if 'npz_file' in data_config:
                self.file_name = data_config['npz_file']

        # load the datasets
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_dataset()

        # pre-process the datasets
        self.x_train = self.x_train.reshape((self.x_train.shape[0], -1))
        self.x_test = self.x_test.reshape((self.x_test.shape[0], -1))

        # change y labels to type int
        self.y_train = self.y_train.astype(int)
        self.y_test = self.y_test.astype(int)

    def get_data(self):
        """
        Gets pre-process mnist training and testing data. Because this method
        is for testing it takes as input the number of datapoints, nb_points,
        to be included in the training and testing set.

        :return: training data
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
            if nb_points is not None and nb_points > 0:
                x_train = x_train[:nb_points]
                y_train = y_train[:nb_points]

                x_test = x_test[:nb_points]
                y_test = y_test[:nb_points]
        else:
            try:
                logger.info('Loaded training data from ' + str(self.file_name))
                data_train = np.load(self.file_name, allow_pickle=True)
                x_train = data_train['x_train']
                y_train = data_train['y_train']
                x_test = data_train['x_test']
                y_test = data_train['y_test']
            except Exception:
                raise IOError('Unable to load training data from path '
                              'provided in config file: ' +
                              self.file_name)
        return (x_train, y_train), (x_test, y_test)
