"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging
import pandas as pd


from ibmfl.data.pandas_data_handler import PandasDataHandler
from ibmfl.util.datasets import load_ionosphere
import numpy as np

logger = logging.getLogger(__name__)


class IonosphereDataHandler(PandasDataHandler):
    """
    Data handler for ionosphere dataset.
    """
    def __init__(self, data_config=None):
        super().__init__()
        self.file_name = None
        self.is_training_split = False
        if data_config is not None:
            if 'txt_file' in data_config:
                self.file_name = data_config['txt_file']

        # load dataset
        self.dataset = self.load_dataset()
        self.indices = None
        # split the dataset for training and test
        self.initialize_dataset()

    def initialize_dataset(self):
        """
        Initialize dataset with training and test split.
        """
        if self.is_training_split:  # dataset was already split.
            # training dataset
            if 'class' in self.dataset[0].columns:
                self.y_train = self.dataset[0]['class'].replace({'b': 0, 'g': 1})
                self.x_train = self.dataset[0].drop(['class'], axis=1)
            else:
                self.y_train = None
                self.x_train = self.dataset[0]
            # test dataset
            if 'class' in self.dataset[1].columns:
                self.y_test = self.dataset[1]['class'].replace({'b': 0, 'g': 1})
                self.x_test = self.dataset[1].drop(['class'], axis=1)
            else:
                self.y_test = None
                self.x_test = self.dataset[1]
        else:
            if 'class' in self.dataset.columns:
                y = self.dataset['class'].replace({'b': 0, 'g': 1})
                x = self.dataset.drop(['class'], axis=1)
            else:
                y = None
                x = self.dataset

            split_idx = int(len(self.dataset) * 0.8)
            self.x_train = x.iloc[:split_idx, :]
            self.x_test = x.iloc[split_idx:, :]
            if 'class' in self.dataset.columns:
                self.y_train = y.iloc[:split_idx]
                self.y_test = y.iloc[split_idx:]
            else:
                self.y_train = None
                self.y_test = None

    def get_data(self):
        """
        Read ionosphere dataset from a given dir.

        :return: A dataset structure
        :rtype: `tuple` of `pandas.core.frame.DataFrame`
        """
        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def update_data(self, compacted_data, permutation, identifiers=None):
        """
        Update the training dataset with PER preprocess result
        """
        # align local training dataset using permutation and compacted dataset
        per_dataset = compacted_data.iloc[permutation]
        if 'class' in per_dataset:
            self.y_train = per_dataset['class']
            self.x_train = per_dataset.drop(['class'], axis=1)
        else:
            self.y_train = None
            self.x_train = per_dataset

        # remove identifiers if there exist
        if identifiers is not None:
            self.x_train = self.x_train.drop(identifiers, axis=1)

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
        return self.x_train.to_numpy(dtype=float)[indices_train]

    def shuffle_data(self, seed):
        """
        Create random permutation of indices for random mini-batch selection.

        :param seed: Random seed agreed upon by all parties.
        :type seed: int
        :return: None
        """
        np.random.seed(seed)
        self.indices = np.random.permutation(len(self.x_train))

    def get_dataset_info(self):
        """
        Read ionosphere.data.txt and extract data information

        :return: spec, a dictionary that contains list_of_features, \
        feature_values and list_of_labels.
        :rtype: `dict`
        """
        spec = {'list_of_features': list(range(self.dataset.shape[1]))}

        feature_values = []
        for feature in range(self.dataset.shape[1]):
            if self.dataset.columns[feature] != 'class':
                new_feature = self.dataset[self.dataset.columns[feature]].cat.categories
                feature_values.append(new_feature.tolist())
        spec['feature_values'] = feature_values

        list_of_labels = self.dataset['class'].cat.categories
        spec['list_of_labels'] = list_of_labels.tolist()

        return spec

    def load_dataset(self):
        """
        Loads the local dataset from a given local path. \
        If no local path is provided, it will download the original nursery \
        dataset from UCI.

        :return: raw dataset
        :rtype: `pandas.core.frame.DataFrame`
        """
        if self.file_name is None:
            dataset = load_ionosphere()
        else:
            try:
                logger.info('Loaded training data from {}'.format(self.file_name))
                if isinstance(self.file_name, dict):
                    # training and dataset is split already.
                    dataset_train = pd.read_csv(self.file_name['train'], dtype='category')
                    dataset_test = pd.read_csv(self.file_name['test'], dtype='category')
                    dataset = [dataset_train, dataset_test]
                    self.is_training_split = True
                else:
                    dataset = pd.read_csv(self.file_name, dtype='category')
            except Exception:
                raise IOError('Unable to load training data from path '
                              'provided in config file: ' + self.file_name)
        return dataset

    def get_labels(self):
        """
        Returns the labels of the dataset

        :return: training and test labels
        :rtype: `tuple` of `np.ndarray`
        """
        return (self.y_train.to_numpy(dtype=float).reshape(-1,1), self.y_test.to_numpy(dtype=float).reshape(-1,1))

    def get_labels_batch(self, batch_size, batch_num):
        """
        Returns the labels for specified mini-batch in training dataset

        :param batch_size: Size of the mini-batch
        :type batch_size: int
        :param batch_num: Mini-batch number to return
        :type batch_num: int
        """
        y_train = self.y_train.to_numpy(dtype=float).reshape(-1,1)
        indices_train = self.indices[int(batch_size*batch_num):int(batch_size*(batch_num+1))]
        return y_train[indices_train]
