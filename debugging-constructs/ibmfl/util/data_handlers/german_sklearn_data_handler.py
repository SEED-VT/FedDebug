"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging

import numpy as np

from ibmfl.data.data_handler import DataHandler
from ibmfl.util.datasets import load_german
from ibmfl.data.data_util import get_reweighing_weights, get_hist_counts
from sklearn.model_selection import train_test_split
import pandas as pd

logger = logging.getLogger(__name__)

TEST_SIZE = 0.2
RANDOM_STATE = 42
SENSITIVE_ATTRIBUTE = 'sex'

class GermanSklearnDataHandler(DataHandler):
    """
    Data handler for German Credit dataset to train a Logistic Regression Classifier on scikit-learn.
    TEST_SIZE is set to 0.2, and RANDOM_STATE is set to 42.
    """
    def __init__(self, data_config=None):
        super().__init__()
        self.file_name = None
        if data_config is not None:
            if 'txt_file' in data_config:
                self.file_name = data_config['txt_file']
            if 'epsilon' in data_config:
                self.epsilon = data_config['epsilon']

        # load dataset
        training_dataset = self.load_dataset()

        # pre-process the data
        self.training_dataset = self.preprocess(training_dataset)
        x_0 = self.training_dataset.iloc[:, :-1]
        y_0 = self.training_dataset.iloc[:, -1]
        x = np.array(x_0)
        y = np.array(y_0)

        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    def get_data(self):
        """
        Gets pre-processed adult training and testing data.

        :return: training data
        :rtype: `tuple`
        """
        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def load_dataset(self):
        """
        Loads the training dataset from a given local path. \
        If no local path is provided, it will download the original german \
        dataset from UCI.

        :return: raw dataset
        :rtype: `pandas.core.frame.DataFrame`
        """
        if self.file_name is None:
            training_dataset = load_german()
        else:
            try:
                if str(self.file_name).endswith('german.data'):
                    #Raw data needs to be split into multiple columns, while party data is already split in generate_data.py
                    logger.info('Loaded training data from ' + str(self.file_name))
                    training_dataset = pd.read_csv(
                        self.file_name, sep = ' ', dtype='category', header=None)
                else:
                    logger.info('Loaded training data from ' + str(self.file_name))
                    training_dataset = pd.read_csv(
                        self.file_name, dtype='category')
            except Exception:
                raise IOError('Unable to load training data from path '
                              'provided in config file: ' +
                              self.file_name)
        return training_dataset

    def get_weight(self):
        """
        Gets pre-processed adult training and testing data, calculates weights for points
        weight = P-expected(sensitive_attribute & class)/P-observed(sensitive_attribute & class)

        :return: weights
        :rtype: `np.array`
        """
        cols = self.get_col_names()
        training_data, (_) = self.get_data()
        return get_reweighing_weights(training_data, SENSITIVE_ATTRIBUTE, cols)

    def get_hist(self):
        """
        Gets pre-processed adult training and testing data, calculates counts for sensitive attribute
        and label
        :return: weights
        :rtype: `np.array`
        """
        e = self.epsilon
        cols = self.get_col_names()
        training_data, (_) = self.get_data()
        return get_hist_counts(training_data, SENSITIVE_ATTRIBUTE, cols, e)


    @staticmethod
    def get_col_names():
        """
        Returns the names of the dataset columns

        :return: column names
        :rtype: `list`
        """
        cols = ['age', 'sex', 'credit_history = delay', 'credit_history = none/paid', 'credit_history = other',
                'savings = 500+', 'savings = <500', 'savings = other', 'employment = 1-4 years',
                'employment = 4+ years', 'employment = unemployed']

        return cols

    @staticmethod
    def get_sa():
        """
        Returns the sensitive attribute

        :return: sensitive attribute
        :rtype: `str`
        """

        return SENSITIVE_ATTRIBUTE

    def preprocess(self, training_data):
        """
        Performs the following preprocessing on adult training and testing data:
        * Drop following features: 'checking-status', 'duration-months', 'purpose', 'credit-amount',
                                    'installment-rate', 'other-debtors', 'residence', 'property', 'installment-plans',
                                    'housing', 'credits', 'job', 'number-liable', 'telephone', 'foreign'
        * Map 'age', 'sex' and 'class' values to 0/1
            * >= 25: 1, <25': 0
            * 'A91': 1, 'A93': 1, 'A94': 1, 'A92': 0, 'A95': 0
        * Split 'credit-history', 'savings' and 'employment' columns into multiple columns based on value

        :param training_data: Raw training data
        :type training_data: `pandas.core.frame.DataFrame
        :return: Preprocessed training data
        :rtype: `pandas.core.frame.DataFrame`
        """
        training_data.columns = ['checking-status', 'duration-months', 'credit-history', 'purpose', 'credit-amount',
                                 'savings', 'employment', 'installment-rate', 'personal-status', 'other-debtors',
                                 'residence', 'property', 'age', 'installment-plans', 'housing', 'credits', 'job',
                                 'number-liable', 'telephone', 'foreign', 'class']

        # filter out columns unused in training, and reorder columns
        training_dataset = training_data[['credit-history', 'savings', 'employment', 'personal-status', 'age',
                                          'class']]

        # map 'sex' and 'race' feature values based on sensitive attribute privileged/unprivileged groups
        training_dataset['sex'] = training_dataset['personal-status'].map({'A91': 1, 'A93': 1, 'A94': 1, 'A92': 0,
                                                                           'A95': 0})
        training_dataset['class'] = training_dataset['class'].map({'1': 1, '2': 0})

        def group_credit_hist(x):
            col = []
            for i in x:
                if i in ['A30', 'A31', 'A32']:
                    col.append('None/Paid')
                elif i == 'A33':
                    col.append('Delay')
                elif i == 'A34':
                    col.append('Other')
                else:
                    col.append('NA')
            return col

        def group_savings(x):
            col = []
            for i in x:
                if i in ['A61', 'A62']:
                    col.append('<500')
                elif i in ['A63', 'A64']:
                    col.append('500+')
                elif i == 'A65':
                    col.append('Unknown/None')
                else:
                    col.append('NA')
            return col

        def group_employ(x):
            col = []
            for i in x:
                if i == 'A71':
                    col.append('Unemployed')
                elif i in ['A72', 'A73']:
                    col.append('1-4 years')
                elif i in ['A74', 'A75']:
                    col.append('4+ years')
                else:
                    col.append('NA')
            return col

        def group_age(x):
            col = []
            for i in x:
                if int(i) >= 26:
                    col.append(1)
                if int(i) < 26:
                    col.append(0)
            return col

        training_dataset['credit-history'] = group_credit_hist(training_dataset['credit-history'])
        training_dataset['savings'] = group_savings(training_dataset['savings'])
        training_dataset['employment'] = group_employ(training_dataset['employment'])
        training_dataset['age'] = group_age(training_dataset['age'])

        new_cols = ['credit_history = delay', 'credit_history = none/paid', 'credit_history = other', 'savings = 500+',
                    'savings = <500', 'savings = other', 'employment = 1-4 years', 'employment = 4+ years',
                    'employment = unemployed']

        for i in new_cols:
            training_dataset[i] = 0

        for index, row in training_dataset.iterrows():
            if row['credit-history'] == "Delay":
                training_dataset.loc[index, 'credit_history = delay'] = 1
            elif row['credit-history'] == "None/Paid":
                training_dataset.loc[index, 'credit_history = none/paid'] = 1
            elif row['credit-history'] == "Other":
                training_dataset.loc[index, 'credit_history = other'] = 1

        for index, row in training_dataset.iterrows():
            if row['savings'] == "500+":
                training_dataset.loc[index, 'savings = 500+'] = 1
            elif row['savings'] == "<500":
                training_dataset.loc[index, 'savings = <500'] = 1
            elif row['savings'] == "Unknown/None":
                training_dataset.loc[index, 'savings = other'] = 1

        for index, row in training_dataset.iterrows():
            if row['employment'] == "4+ years":
                training_dataset.loc[index, 'employment = 4+ years'] = 1
            elif row['savings'] == "1-4 years":
                training_dataset.loc[index, 'employment = 1-4 years'] = 1
            elif row['savings'] == "Unemployed":
                training_dataset.loc[index, 'employment = unemployed'] = 1

        # move class column to be last column
        label = training_dataset['class']
        training_dataset.drop(['personal-status', 'class', 'credit-history', 'savings', 'employment'],
                               axis=1, inplace=True)
        training_dataset['class'] = label

        return training_dataset
