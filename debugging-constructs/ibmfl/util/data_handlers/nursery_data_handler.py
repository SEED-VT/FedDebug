"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging
import pandas as pd


from ibmfl.data.pandas_data_handler import PandasDataHandler
from ibmfl.util.datasets import load_nursery

logger = logging.getLogger(__name__)


class NurseryDataHandler(PandasDataHandler):
    """
    Data handler for nursery dataset.
    """
    def __init__(self, data_config=None):
        super().__init__()
        self.file_name = None
        if data_config is not None:
            if 'txt_file' in data_config:
                self.file_name = data_config['txt_file']

        # load dataset
        self.training_dataset = self.load_dataset()
        # pre-process the data
        try:
            self.y_test = self.training_dataset['class'].values.tolist()
            self.x_test = self.training_dataset.drop(['class'], axis=1)
        except KeyError:
            logger.warning(
                'Column name missing. Adding default column names...')
            self.training_dataset.columns = ['1', '2', '3', '4', '5', '6',
                                             '7', '8', 'class']
            self.y_test = self.training_dataset['class'].values.tolist()
            self.x_test = self.training_dataset.drop(['class'], axis=1)

    def get_data(self):
        """
        Read nursery.data.txt from a given dir.

        :return: A dataset structure
        :rtype: `tuple` of `pandas.core.frame.DataFrame`
        """
        return self.training_dataset, (self.x_test, self.y_test)

    def get_dataset_info(self):
        """
        Read nursery.data.txt and extract data information

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

    def load_dataset(self):
        """
        Loads the local dataset from a given local path. \
        If no local path is provided, it will download the original nursery \
        dataset from UCI.

        :return: raw dataset
        :rtype: `pandas.core.frame.DataFrame`
        """
        if self.file_name is None:
            training_dataset = load_nursery()
        else:
            try:
                logger.info('Loaded training data from ' +
                            str(self.file_name))
                training_dataset = pd.read_csv(self.file_name,
                                               dtype='category')
            except Exception:
                raise IOError('Unable to load training data from path '
                              'provided in config file: ' + self.file_name)
        return training_dataset
