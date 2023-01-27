"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging

import numpy as np

from ibmfl.data.data_handler import DataHandler
from ibmfl.data.data_util import get_reweighing_weights, get_hist_counts
from ibmfl.util.datasets import load_compas
from sklearn.model_selection import train_test_split
import pandas as pd
import gzip

logger = logging.getLogger(__name__)

TEST_SIZE = 0.2
RANDOM_STATE = 1
SENSITIVE_ATTRIBUTE = 'sex'


class CompasSklearnDataHandler(DataHandler):
    """
    Data handler for Compas dataset to train a Logistic Regression Classifier on scikit-learn.
    TEST_SIZE is set to 0.2, and RANDOM_STATE is set to 1.
    """
    def __init__(self, data_config=None):
        super().__init__()
        self.file_name = None
        if data_config is not None:
            if 'txt_file' in data_config:
                self.file_name = data_config['txt_file']
            if 'epsilon' in data_config:
                self.epsilon = data_config['epsilon']

        # load local dataset
        training_dataset = self.load_dataset()

        # preprocess the dataset
        self.training_dataset = self.preprocess(training_dataset)
        x_0 = self.training_dataset.iloc[:, :-1]
        y_0 = self.training_dataset.iloc[:, -1]
        x = np.array(x_0)
        y = np.array(y_0)

        # split the dataset into training and testing
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
        If no local path is provided, it will download the original Compas \
        dataset from UCI.

        :return: raw dataset
        :rtype: `pandas.core.frame.DataFrame`
        """
        if self.file_name is None:
            training_dataset = load_compas()
        else:
            try:
                if str(self.file_name).endswith('compas-scores-two-years.csv'):
                    #Raw data needs to be unzipped
                    logger.info('Loaded training data from ' + str(self.file_name))
                    with open(str(self.file_name), 'rb') as fd:
                        compas_unzip = gzip.GzipFile(fileobj=fd)
                        training_dataset = pd.read_csv(compas_unzip)
                    training_dataset['class'] = training_dataset['two_year_recid']
                    training_dataset = training_dataset.drop('two_year_recid', axis=1)
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
        cols = ['sex', 'race', 'age_cat = 25 to 45', 'age_cat = Greater than 45',
                'age_cat = Less than 25', 'priors_count = 0', 'priors_count = 1 to 3', 'priors_count = More than 3', 'c_charge_degree = F',
                'c_charge_degree = M']

        return cols

    def get_sa(self):
        """
        Returns the sensitive attribute

        :return: sensitive attribute
        :rtype: `str`
        """

        return SENSITIVE_ATTRIBUTE

    def preprocess(self, training_dataset):
        """
        Performs the following preprocessing on adult training and testing data:
        * Map 'sex' values to 0/1 based on underprivileged/privileged groups
        * Filter out rows with values outside of specific ranges for 'days_b_screening_arrest', 'is_recid',
        'c_charge_degree', 'score_text', 'race'
        * Quantify length_of_stay from 'c_jail_out' and 'c_jail_in'
        * Quanitfy 'priors-count', 'length_'age_cat', 'score_text', 'age_cat'
        * Drop following features: 'id', 'name', 'first', 'last', 'compas_screening_date', 'dob',
            'age', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count', 'priors_count',
            'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number', 'c_offense_date',
            'c_arrest_date', 'c_days_from_compas', 'c_charge_desc', 'is_recid', 'r_case_number',
            'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc', 'r_jail_in',
            'r_jail_out', 'violent_recid', 'is_violent_recid', 'vr_case_number', 'vr_charge_degree',
            'vr_offense_date', 'vr_charge_desc', 'type_of_assessment', 'decile_score.1', 'screening_date',
            'v_type_of_assessment', 'v_decile_score', 'v_score_text', 'v_screening_date', 'in_custody',
            'out_custody', 'priors_count.1', 'start', 'end', 'event'
        * Split 'age-cat', 'priors_count' and 'c_charge_degree' columns into multiple columns based on value

        :param training_dataset: Raw training data
        :type training_dataset: `pandas.core.frame.DataFrame
        :return: Preprocessed training data
        :rtype: `pandas.core.frame.DataFrame`
        """
        # map 'sex' feature values based on sensitive attribute privileged/unprivileged groups
        training_dataset['sex'] = training_dataset['sex'].map({'Female': 1, 'Male': 0})
        training_dataset['days_b_screening_arrest'] = training_dataset['days_b_screening_arrest'].astype(float)

        ix = training_dataset['days_b_screening_arrest'] <= 30
        ix = (training_dataset['days_b_screening_arrest'] >= -30) & ix
        ix = (training_dataset['is_recid'] != -1) & ix
        ix = (training_dataset['c_charge_degree'] != "O") & ix
        ix = (training_dataset['score_text'] != 'N/A') & ix
        training_dataset = training_dataset.loc[ix, :]
        training_dataset['length_of_stay'] = (pd.to_datetime(training_dataset['c_jail_out']) -
                                              pd.to_datetime(training_dataset['c_jail_in'])).apply(
                                              lambda x: x.days)

        # filter out columns unused in training, and reorder columns
        training_dataset = training_dataset.loc[~training_dataset['race'].isin(
            ['Native American', 'Hispanic', 'Asian', 'Other']), :]
        training_dataset = training_dataset[['sex', 'race', 'age_cat', 'c_charge_degree',
                                             'score_text', 'priors_count', 'is_recid',
                                             'length_of_stay', 'class']]
        training_dataset['priors_count'] = training_dataset['priors_count'].astype(int)
        training_dataset['class'] = training_dataset['class'].map({'1': 0, '0': 1})
        # Quantize priors count between 0, 1-3, and >3
        def quantizePrior(x):
            col = []
            for i in x:
                if i <= 0:
                    col.append('0')
                elif 1 <= i <= 3:
                    col.append('1 to 3')
                else:
                    col.append('More than 3')
            return col

        # Quantize length of stay
        def quantizeLOS(x):
            col = []
            for i in x:
                if i <= 7:
                    col.append('<week')
                elif 8 <= i <= 93:
                    col.append('<3months')
                else:
                    col.append('>3 months')
            return col

        # Quantize length of stay
        def adjustAge(x):
            col = []
            for i in x:
                if i == '25 - 45':
                    col.append('25 to 45')
                else:
                    col.append(i)
            return col

        # Quantize score_text to MediumHigh
        def quantizeScore(x):
            col = []
            for i in x:
                if (i == 'High') | (i == 'Medium'):
                    col.append('MediumHigh')
                else:
                    col.append(i)
            return col

        # Map race to 0/1 based on unprivileged/privileged groups
        def group_race(x):
            col = []
            for i in x:
                if i == "Caucasian":
                    col.append(1.0)
                else:
                    col.append(0.0)
            return col

        training_dataset['priors_count'] = quantizePrior(training_dataset['priors_count'])
        training_dataset['length_of_stay'] = quantizeLOS(training_dataset['length_of_stay'])
        training_dataset['score_text'] = quantizeScore(training_dataset['score_text'])
        training_dataset['age_cat'] = adjustAge(training_dataset['age_cat'])
        training_dataset['race'] = group_race(training_dataset['race'])

        new_cols = ['age_cat = 25 to 45', 'age_cat = Greater than 45', 'age_cat = Less than 25', 'priors_count = 0',
                    'priors_count = 1 to 3', 'priors_count = More than 3', 'c_charge_degree = F', 'c_charge_degree = M']

        for i in new_cols:
            training_dataset[i] = 0

        for index, row in training_dataset.iterrows():
            if row['age_cat'] == '25 to 45':
                training_dataset.loc[index, 'age_cat = 25 to 45'] = 1
            elif row['age_cat'] == 'More than 45':
                training_dataset.loc[index, 'age_cat = Greater than 45'] = 1
            elif row['age_cat'] == 'Less than 45':
                training_dataset.loc[index, 'age_cat = Less than 25'] = 1

        for index, row in training_dataset.iterrows():
            if row['priors_count'] == '0':
                training_dataset.loc[index, 'priors_count = 0'] = 1
            elif row['priors_count'] == '1 to 3':
                training_dataset.loc[index, 'priors_count = 1 to 3'] = 1
            elif row['priors_count'] == 'More than 3':
                training_dataset.loc[index, 'priors_count = More than 3'] = 1

        for index, row in training_dataset.iterrows():
            if row['c_charge_degree'] == "F":
                training_dataset.loc[index, 'c_charge_degree = F'] = 1
            elif row['c_charge_degree'] == "M":
                training_dataset.loc[index, 'c_charge_degree = M'] = 1

        training_dataset = training_dataset.drop(
            ['age_cat', 'priors_count', 'c_charge_degree', 'is_recid', 'score_text', 'length_of_stay'], axis=1)

        label = training_dataset['class']
        training_dataset.drop(['class'], axis=1, inplace=True)
        training_dataset['class'] = label

        return training_dataset
