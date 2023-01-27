"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
from __future__ import print_function
import logging
import pandas as pd

from ibmfl.util.datasets import load_linovf
from sklearn.model_selection import train_test_split
from ibmfl.data.data_handler import DataHandler

logger = logging.getLogger(__name__)

TEST_SIZE = 0.2
RANDOM_STATE = 42


class LinovfDataHandler(DataHandler):
    """
    Data handler for Binary Overfit dataset to train a Regression Model
    based on a Linear Relationship.
    TEST_SIZE is set to 0.2, and RANDOM_STATE is set to 42.
    """
    def __init__(self, data_config=None):
        super().__init__()
        self.file_name = None
        if data_config is not None:
            if 'txt_file' in data_config:
                self.file_name = data_config['txt_file']

        # load dataset
        X, y = self.load_dataset()
        # split the dataset into training and testing

        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=TEST_SIZE,
                             random_state=RANDOM_STATE)

    def get_data(self):
        """
        Obtains generated data and splits to test and train sets.

        :return: training data and testing data
        :rtype: `tuple`
        """
        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def load_dataset(self):
        """
        Loads the dataset from a given local path. \
        If no local data path is provided, it load the dataset from \
        `load_linovf()` defined in `util/datasets.py`.

        :return: the local dataset as in the format of (features, labels).
        :rtype: `tuple`
        """
        if self.file_name is None:
            X, y = load_linovf()
            X = X.reshape(-1, 1)
        else:
            try:
                logger.info('Loaded training data from '+ str(self.file_name))
                data = pd.read_csv(self.file_name, header=None).to_numpy()
                X, y = data[:, :-1], data[:, -1]
            except Exception:
                raise IOError('Unable to load training data from path '
                              'provided in config file: ' + self.file_name)
        return X, y
