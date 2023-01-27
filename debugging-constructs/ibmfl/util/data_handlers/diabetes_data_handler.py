"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
from __future__ import print_function
import sys
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from ibmfl.util.datasets import load_diabetes
from ibmfl.data.data_handler import DataHandler
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

logger = logging.getLogger(__name__)

TEST_SIZE=0.2
RANDOM_STATE=1234


class DiabetesDataHandler(DataHandler):
    """
    Data handler for Diabetes dataset to train a Multiclass Classification Model.
    TEST_SIZE is set to 0.2, and RANDOM_STATE is set to 1234.
    """
    def __init__(self, data_config=None):
        super().__init__()
        self.file_name = None
        if data_config is not None:
            if 'txt_file' in data_config: self.file_name = data_config['txt_file']

        # load the dataset
        data = self.load_dataset()
        # Separate Features and Targets
        X, y = self.preprocess(data)

        # Split Dataset
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    def get_data(self):
        """
        Get pre-processed diabetes training and testing data.

        :return: training data
        :rtype: `tuple`
        """
        return (self.x_train, self.x_test), (self.y_train, self.y_test)

    def load_dataset(self):
        """
        Loads the dataset from a given local path. \
        If no local path is provided, it will download the \
        diabetes dataset from UCI.

        :return: the local dataset
        :rtype: `pandas.core.frame.DataFrame`
        """
        if self.file_name is None:
            data = load_diabetes()
        else:
            try:
                logger.info('Loaded training data from ' + str(self.file_name))
                data = pd.read_csv(self.file_name, dtype='category')
            except Exception:
                raise IOError('Unable to load training data from path '
                              'provided in config file: ' + self.file_name)
        return data

    def preprocess(self, data):
        # Obtain Labels
        y = data['readmitted'].to_numpy()
        data = data.drop(['readmitted'], axis=1).to_numpy()

        # Perform Encoding Transformation
        enc = OrdinalEncoder()
        enc.fit(data)
        X = enc.transform(data)

        le = LabelEncoder()
        y = le.fit_transform(y)

        return X, y
