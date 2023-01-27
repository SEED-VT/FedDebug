"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""

from __future__ import print_function
import logging
import numpy as np
import pandas as pd

from ibmfl.util.datasets import load_multovf
from sklearn.model_selection import train_test_split
from ibmfl.data.data_handler import DataHandler
from ibmfl.data.pandas_data_handler import PandasDataHandler

logger = logging.getLogger(__name__)

TEST_SIZE = 0.2
RANDOM_STATE = 1234


class MultovfDataHandler(DataHandler):
    """
    Data handler for Multiclass Overfit dataset to train a Multiclass
    Classification Model. TEST_SIZE is set to 0.2, and RANDOM_STATE is set to 42.
    """

    def __init__(self, data_config=None):
        super().__init__()
        self.file_name = None
        if data_config is not None:
            if 'txt_file' in data_config:
                self.file_name = data_config['txt_file']

        # load dataset
        X, y = self.load_dataset()
        # preprocess
        X, y = self.preprocess_x_y(X, y)
        # split the dataset into training and testing
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=TEST_SIZE,
                             random_state=RANDOM_STATE)

    def get_data(self):
        """
        Obtains the generated datasets.

        :return: training data and testing data
        :rtype: `tuple`
        """
        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def load_dataset(self):
        """
        Loads the dataset from a given local path. \
        If no local data path is provided, it load the dataset from \
        `load_multovf()` defined in `util/datasets.py`.

        :return: the local dataset as in the format of (features, labels).
        :rtype: `tuple`
        """
        if self.file_name is None:
            X, y = load_multovf()
        else:
            try:
                logger.info('Loaded training data from ' + str(self.file_name))
                data = pd.read_csv(self.file_name, header=None).to_numpy()
                X, y = data[:, :-1], data[:, -1].astype('int')
            except Exception:
                raise IOError('Unable to load training data from path '
                              'provided in config file: ' + self.file_name)
        return X, y

    def preprocess_x_y(self, X, y):
        return X, y


class MultovfDTDataHandler(MultovfDataHandler, PandasDataHandler):
    """
    Data handler for Multiclass Overfit dataset to train a multi-class
    classification decision tree Model.
    TEST_SIZE is set to 0.2, and RANDOM_STATE is set to 42.
    """
    def __init__(self):
        super().__init__()
        self.traindata = None
        self.testdata = None
        self.preprocess()

    def get_data(self):
        """
        Obtains generated data and splits to test and train sets.

        :return: training data and testing data
        :rtype: `tuple`
        """
        return self.traindata, (self.x_test, self.y_test)

    def get_dataset_info(self):
        """
        Read multovf and extract data information

        :return: spec, a dictionary that contains list_of_features, \
        feature_values and list_of_labels.
        :rtype: `dict`
        """
        training_dataset, (_, _) = self.get_data()
        spec = {'list_of_features': list(range(training_dataset.shape[1] - 1))}

        feature_values = []
        for feature in range(training_dataset.shape[1]):
            if training_dataset.columns[feature] != 'class':
                new_feature = training_dataset[
                    training_dataset.columns[feature]].cat.categories
                feature_values.append(new_feature.tolist())
        spec['feature_values'] = feature_values

        list_of_labels = training_dataset['class'].cat.categories
        spec['list_of_labels'] = list_of_labels.tolist()

        return spec

    def preprocess(self):
        """
        Preprocesses the dataset into pandas dataframe format for \
        decision tree training.

        :return: training data and testing data
        :rtype: `tuple` of `pandas.core.frame.DataFrame`
        """
        # convert to pd.DataFrame and add column names
        self.y_train = self.y_train.reshape((len(self.y_train), 1))
        traindata = np.append(self.x_train, self.y_train, axis=1)
        traindata = traindata.astype('int')
        # self.traindata = pd.DataFrame(data=traindata, columns=[0, 'class'])
        self.traindata = pd.DataFrame(data=traindata, columns=[0, 1, 2, 'class'])
        self.traindata[0] = self.traindata[0].astype('category')
        self.traindata[1] = self.traindata[1].astype('category')
        self.traindata[2] = self.traindata[2].astype('category')
        self.traindata['class'] = self.traindata['class'].astype('category')

        self.y_test = self.y_test.reshape((len(self.y_test), 1))
        testdata = np.append(self.x_test, self.y_test, axis=1)
        testdata = testdata.astype('int')
        # self.testdata = pd.DataFrame(data=testdata, columns=[0, 'class'])
        self.testdata = pd.DataFrame(data=testdata, columns=[0, 1, 2, 'class'])
        self.testdata[0] = self.testdata[0].astype('category')
        self.testdata[1] = self.testdata[1].astype('category')
        self.testdata[2] = self.testdata[2].astype('category')
        self.x_test = self.testdata.drop(['class'], axis=1)
        self.y_test = self.testdata['class'].astype('category')
        self.y_test = self.y_test.values.tolist()


class MultovfKerasDataHandler(MultovfDataHandler):
    def preprocess_x_y(self, X, y):
        from keras.utils import to_categorical
        return X, to_categorical(y)
