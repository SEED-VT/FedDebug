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
from ibmfl.util.data_handlers.adult_sklearn_data_handler import AdultSklearnDataHandler

logger = logging.getLogger(__name__)

SENSITIVE_ATTRIBUTE='sex'

class AdultPRDataHandler(AdultSklearnDataHandler):

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
        cols = ['race', 'sex', 'age1', 'age2', 'age3', 'age4', 'age5', 'age6',
                'age7', 'ed6less', 'ed6', 'ed7', 'ed8', 'ed9',
                'ed10', 'ed11', 'ed12', 'ed12more']
        cols.remove(SENSITIVE_ATTRIBUTE)
        cols = np.append(cols, SENSITIVE_ATTRIBUTE)
        cols = np.append(cols, 'class')
        return cols

    def preprocess(self, training_data):
        """
        Performs the following preprocessing on adult training and testing data:
        * Drop following features: 'workclass', 'fnlwgt', 'education', 'marital-status', 'occupation',
          'relationship', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
        * Map 'race', 'sex' and 'class' values to 0/1
            * ' White': 1, ' Amer-Indian-Eskimo': 0, ' Asian-Pac-Islander': 0, ' Black': 0, ' Other': 0
            * ' Male': 1, ' Female': 0
            * Further details in Kamiran, F. and Calders, T. Data preprocessing techniques for classification without discrimination
        * Split 'age' and 'education' columns into multiple columns based on value

        :param training_data: Raw training data
        :type training_data: `pandas.core.frame.DataFrame
        :return: Preprocessed training data
        :rtype: `pandas.core.frame.DataFrame`
        """
        if len(training_data.columns)==15:
            # drop 'fnlwgt' column
            training_data = training_data.drop(
                training_data.columns[2], axis='columns')

        training_data.columns = ['age', 'workclass', 'education', 'education-num', 'marital-status',
                                    'occupation',
                                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                                    'native-country',
                                    'class']

        # filter out columns unused in training, and reorder columns
        training_dataset = training_data[['race', 'sex', 'age', 'education-num', 'class']]

        # map 'sex' and 'race' feature values based on sensitive attribute privileged/unpriveleged groups
        training_dataset['sex'] = training_dataset['sex'].map({' Female': 0, ' Male': 1})
        training_dataset['race'] = training_dataset['race'].map(
            {' Asian-Pac-Islander': 0, ' Amer-Indian-Eskimo': 0, ' Other': 0, ' Black': 0, ' White': 1})

        # map 'class' values to 0/1 based on positive and negative classification
        training_dataset['class'] = training_dataset['class'].map({' <=50K': 0, ' >50K': 1})

        training_dataset['age'] = training_dataset['age'].astype(int)
        training_dataset['education-num'] = training_dataset['education-num'].astype(int)

        # split age column into category columns
        for i in range(8):
            if i != 0:
                training_dataset['age' + str(i)] = 0

        for index, row in training_dataset.iterrows():
            if row['age'] < 20:
                training_dataset.loc[index, 'age1'] = 1
            elif ((row['age'] < 30) & (row['age'] >= 20)):
                training_dataset.loc[index, 'age2'] = 1
            elif ((row['age'] < 40) & (row['age'] >= 30)):
                training_dataset.loc[index, 'age3'] = 1
            elif ((row['age'] < 50) & (row['age'] >= 40)):
                training_dataset.loc[index, 'age4'] = 1
            elif ((row['age'] < 60) & (row['age'] >= 50)):
                training_dataset.loc[index, 'age5'] = 1
            elif ((row['age'] < 70) & (row['age'] >= 60)):
                training_dataset.loc[index, 'age6'] = 1
            elif row['age'] >= 70:
                training_dataset.loc[index, 'age7'] = 1

        # split age column into multiple columns
        training_dataset['ed6less'] = 0
        for i in range(13):
            if i >= 6:
                training_dataset['ed' + str(i)] = 0
        training_dataset['ed12more'] = 0

        for index, row in training_dataset.iterrows():
            if row['education-num'] < 6:
                training_dataset.loc[index, 'ed6less'] = 1
            elif row['education-num'] == 6:
                training_dataset.loc[index, 'ed6'] = 1
            elif row['education-num'] == 7:
                training_dataset.loc[index, 'ed7'] = 1
            elif row['education-num'] == 8:
                training_dataset.loc[index, 'ed8'] = 1
            elif row['education-num'] == 9:
                training_dataset.loc[index, 'ed9'] = 1
            elif row['education-num'] == 10:
                training_dataset.loc[index, 'ed10'] = 1
            elif row['education-num'] == 11:
                training_dataset.loc[index, 'ed11'] = 1
            elif row['education-num'] == 12:
                training_dataset.loc[index, 'ed12'] = 1
            elif row['education-num'] > 12:
                training_dataset.loc[index, 'ed12more'] = 1

        training_dataset.drop(['age', 'education-num'], axis=1, inplace=True)

        # move class and sa column to be last columns
        label = training_dataset['class']
        sa_col = training_dataset[SENSITIVE_ATTRIBUTE]
        training_dataset.drop([SENSITIVE_ATTRIBUTE, 'class'], axis=1, inplace=True)
        training_dataset[SENSITIVE_ATTRIBUTE] = sa_col
        training_dataset['class'] = label

        return training_dataset
