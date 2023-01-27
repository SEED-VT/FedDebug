"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging
import numpy as np
import pandas as pd
import tempfile
from ibmfl.util.data_handlers.compas_sklearn_data_handler import CompasSklearnDataHandler

logger = logging.getLogger(__name__)

SENSITIVE_ATTRIBUTE='sex'

class CompasPRDataHandler(CompasSklearnDataHandler):

    def get_data(self):
        """
        Returns pre-processed adult training and testing data.

        :return: training and testing data
        :rtype: `tuple`
        """
        columns = self.get_cols()
        sa = self.get_sa()
        training_data = (self.x_train, self.y_train)
        testing_data = (self.x_test, self.y_test)
        (self.x_train, self.y_train) = self.reformat(training_data, sa, columns)
        (self.x_test, self.y_test) = self.reformat(testing_data, sa, columns)

        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def _create_file_in_kamishima_format(self, df, class_attr,
                                         positive_class_val, sensitive_attrs,
                                         single_sensitive, privileged_vals):
        """
        Reformatting for the Kamishima code.

        :param df: dataframe
        :type df: `pandas.Dataframe`
        :param class_attr: class column name
        :type class_attr: `str`
        :param positive_class_val: value of positive label
        :type positive_class_val: `int`
        :param sensitive_attrs: sensitive attribute list
        :type sensitive_attrs: `lst`
        :param single_sensitive: sensitive attribute column
        :type single_sensitive: `pandas.Dataframe`
        :param privileged_vals: privilege class value
        :type privileged_vals: `list`
        :return: text file of array
        :rtype: `txtfile`
        """
        x = []
        for col in df:
            if col != class_attr and col not in sensitive_attrs:
                x.append(np.array(df[col].values, dtype=np.float64))
        x.append(np.array(single_sensitive.isin(privileged_vals),
                          dtype=np.float64))
        x.append(np.array(df[class_attr] == positive_class_val,
                          dtype=np.float64))

        fd, name = tempfile.mkstemp()
        np.savetxt(name, np.array(x).T)
        return name

    def reformat(self, dataset, sa, columns):
        """
        Convert array to Kamishima format, and reformat for fit_model().

        :param dataset: training dataset
        :type dataset: `np.array`
        :param sa: sensitive attribute
        :type sa: `str`
        :param columns: list of column names
        :type columns: `list`
        :return: tuple of transformed data
        :rtype: `tuple`
        """
        x = dataset[0]
        y = dataset[1]
        data = np.column_stack([x, y])
        train_df = pd.DataFrame(data=data, columns=columns)

        all_sensitive_attributes = [sa]
        sens_df = train_df[sa]
        train_name = self._create_file_in_kamishima_format(train_df, 'class', 1,
                                                           all_sensitive_attributes, sens_df, [1])

        D = np.loadtxt(train_name)
        y = np.array(D[:, -1])

        return (D[:, :-1], y)

    @staticmethod
    def get_cols():
        """
        Returns the names of the dataset columns

        :return: column names
        :rtype: `list`
        """
        cols = ['sex', 'race', 'age_cat = 25 to 45', 'age_cat = Greater than 45',
                'age_cat = Less than 25', 'priors_count = 0', 'priors_count = 1 to 3', 'priors_count = More than 3',
                'c_charge_degree = F',
                'c_charge_degree = M']
        cols.remove(SENSITIVE_ATTRIBUTE)
        cols = np.append(cols, SENSITIVE_ATTRIBUTE)
        return cols

    def preprocess(self, training_data):
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

        :param: training_dataset: Raw training data
        :type training_dataset: `pandas.core.frame.DataFrame
        :return: Preprocessed training data
        :rtype: `pandas.core.frame.DataFrame`
        """
        # map 'sex' feature values based on sensitive attribute privileged/unprivileged groups
        training_data['sex'] = training_data['sex'].map({'Female': 1, 'Male': 0})
        training_data['days_b_screening_arrest'] = training_data['days_b_screening_arrest'].astype('float')
        #training_dataset['is_recid'] = training_dataset['is_recid'].astype('int')

        ix = training_data['days_b_screening_arrest'] <= 30
        ix = (training_data['days_b_screening_arrest'] >= -30) & ix
        ix = (training_data['is_recid'] != -1) & ix
        ix = (training_data['c_charge_degree'] != "O") & ix
        ix = (training_data['score_text'] != 'N/A') & ix
        training_data = training_data.loc[ix, :]
        training_data['length_of_stay'] = (pd.to_datetime(training_data['c_jail_out']) - pd.to_datetime(training_data['c_jail_in'])).apply(
            lambda x: x.days)

        # filter out columns unused in training, and reorder columns
        training_data = training_data.loc[~training_data['race'].isin(['Native American', 'Hispanic', 'Asian', 'Other']), :]
        training_dataset = training_data[['sex', 'race', 'age_cat', 'c_charge_degree', 'score_text', 'priors_count', 'is_recid', 'length_of_stay', 'class']]
        training_dataset['priors_count'] = training_dataset['priors_count'].astype(int)
        training_dataset['class'] = training_dataset['class'].astype(int)

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

        def flip_class(x):
            col = []
            for i in x:
                if i == 1:
                    col.append(0)
                if i == 0:
                    col.append(1)
            return col

        training_dataset['priors_count'] = quantizePrior(training_dataset['priors_count'])
        training_dataset['length_of_stay'] = quantizeLOS(training_dataset['length_of_stay'])
        training_dataset['score_text'] = quantizeScore(training_dataset['score_text'])
        training_dataset['age_cat'] = adjustAge(training_dataset['age_cat'])
        training_dataset['race'] = group_race(training_dataset['race'])
        training_dataset['class'] = flip_class(training_dataset['class'])

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

        training_dataset = training_dataset.drop(['age_cat', 'priors_count', 'c_charge_degree', 'is_recid', 'score_text',
                      'length_of_stay'], axis=1)

        label = training_dataset['class']
        sa_col = training_dataset[SENSITIVE_ATTRIBUTE]
        training_dataset.drop([SENSITIVE_ATTRIBUTE, 'class'], axis=1, inplace=True)
        training_dataset[SENSITIVE_ATTRIBUTE] = sa_col
        training_dataset['class'] = label

        return training_dataset
