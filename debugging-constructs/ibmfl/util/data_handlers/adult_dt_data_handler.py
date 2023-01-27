"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging
import pandas as pd


from ibmfl.data.pandas_data_handler import PandasDataHandler
from ibmfl.util.datasets import load_adult

logger = logging.getLogger(__name__)


class AdultDTDataHandler(PandasDataHandler):
    """
    Data handler for adult dataset.
    """
    def __init__(self, data_config=None):
        super().__init__()
        self.file_name = None
        # extract local data path if any
        if data_config is not None:
            if 'txt_file' in data_config:
                self.file_name = data_config['txt_file']

        # load dataset
        training_dataset = self.load_dataset()
        # pre-process the data
        self.training_dataset = self.preprocess(training_dataset)

        # testing dataset = whole dataset
        testing_dataset = load_adult()
        self.testing_dataset = self.preprocess(testing_dataset)

    def get_data(self):
        """
        Returns training and testing datasets.

        :return: A dataset structure
        :rtype: `pandas.core.frame.DataFrame`
        """
        self.y_test = self.testing_dataset['class'].values.tolist()
        self.x_test = self.testing_dataset.drop(['class'], axis=1)

        return self.training_dataset, (self.x_test, self.y_test)

    def get_dataset_info(self):
        """
        Read adult.data.txt and extract data information

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
        Loads the training dataset from a given local path. \
        If no local path is provided, it will download the original adult \
        dataset from UCI.

        :return: raw dataset
        :rtype: `pandas.core.frame.DataFrame`
        """
        if self.file_name is None:
            training_dataset = load_adult()
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

    def preprocess(self, training_dataset):
        """
        Performs the following pre-processing on the provided dataset:
        * Categorize `age` feature into 4 categories,
            i.e., [0, 18] -> 'teen', [18, 40] -> 'adult',
            [40, 80] -> 'old-adult', and [80, 99] -> 'elder'.
        * Categorize `workclass` feature into 3 categories,
            i.e., ["?", "Never-worked", "Private", "Without-pay"] -> 'others',
            ["Federal-gov", " Local-gov"] -> 'gov',
            and ->["Self-emp-inc", "Self-emp-not-inc"] -> 'self'.
        * Categorize `education` feature into 5 categories,
            i.e., ["10th"," 11th", " 12th", " 1st-4th", " 5th-6th", " 7th-8th",
            " 9th"] -> "non_college",
            [" Assoc-acdm", " Assoc-voc"] ->"assoc",
            [" Bachelors", " Some-college"] -> "college",
            [" Doctorate", " HS-grad", " Masters"] -> "grad",
            and [" Preschool", " Prof-school"] ->"others"
        * Categorize `education-num` feature into 3 categories:
            i.e., [0, 5] -> '<5', [5, 10] -> '5-10', and [10, 17] -> '>10'.
        * Categorize `capital-gain` feature into 5 categories,
            i.e., [-1, 1] -> 0, [1, 39999] -> 1,[39999, 49999] ->2,
            [49999, 79999] ->3, and [79999, 99999] ->4.
        * Categorize `capital-loss` feature into 6 categories,
            i.e., [-1, 1] -0, [1, 999] ->1, [999, 1999] ->2, [1999, 2999] ->3,
             [2999, 3999] ->4, [3999, 4499] -> 5.
        * Categorize `hours` feature into 3 categories,
            i.e., [0, 20] -> '<20', [20, 40] -> '20-40', [40, 100] -> '>40'.
        * Categorize `native-coutry` into 5 categories,
            i.e., [' ?',] -> 'others',
            [' Cambodia', ' China', ' Hong', ' India', ' Iran', ' Japan',
            ' Laos', ' Philippines', ' South', ' Taiwan', ' Thailand',
            ' Vietnam'] -> 'asia',
            [' Canada', ' Outlying-US(Guam-USVI-etc)', ' United-States'] -> 'north_america',
            [' Columbia', ' Cuba', ' Dominican-Republic', ' Ecuador',
            ' El-Salvador', ' Guatemala', ' Haiti', ' Honduras', ' Jamaica',
            ' Mexico', ' Nicaragua', ' Peru', ' Puerto-Rico',
            ' Trinadad&Tobago'] -> 'south_america',
            [' England', ' France', ' Germany', ' Greece',
            ' Holand-Netherlands', ' Hungary', ' Ireland', ' Italy', ' Poland',
            ' Portugal', ' Scotland', ' Yugoslavia'] -> 'europe'.
        * Training label column is renamed to `class`

        Function assumes dataset has column labels ['1', '2', '3'...'13', 'class'],
        added in the load_adult() function in datasets.py,
        feature `fnlwgt` was also dropped in load_adult().

        :param training_dataset: Raw training data
        :type training_dataset: `pandas.core.frame.DataFrame`
        :return: Preprocessed training data
        :rtype: `pandas.core.frame.DataFrame`
        """

        training_dataset['1'] = training_dataset['1'].astype(int)
        training_dataset['4'] = training_dataset['4'].astype(int)
        training_dataset['10'] = training_dataset['10'].astype(int)
        training_dataset['11'] = training_dataset['11'].astype(int)
        training_dataset['12'] = training_dataset['12'].astype(int)

        age = training_dataset['1']
        edu_num = training_dataset['4']
        cap_gain = training_dataset['10']
        cap_loss = training_dataset['11']
        hours = training_dataset['12']

        age_bins = [0, 18, 40, 80, 99]
        age_labels = ['teen', 'adult', 'old-adult', 'elder']
        edu_num_bins = [0, 5, 10, 17]
        edu_num_labels = ['<5', '5-10', '>10']
        cap_gain_bins = [-1, 1, 39999, 49999, 79999, 99999]
        cap_gain_labels = [0, 1, 2, 3, 4]
        cap_loss_bins = [-1, 1, 999, 1999, 2999, 3999, 4499]
        cap_loss_labels = [0, 1, 2, 3, 4, 5]
        hr_bins = [0, 20, 40, 100]
        hr_labels = ['<20', '20-40', '>40']

        training_dataset['1'] = pd.cut(age, bins=age_bins, labels=age_labels)
        training_dataset['4'] = pd.cut(edu_num, bins=edu_num_bins, labels=edu_num_labels)
        training_dataset['10'] = pd.cut(cap_gain, bins=cap_gain_bins, labels=cap_gain_labels)
        training_dataset['11'] = pd.cut(cap_loss, bins=cap_loss_bins, labels=cap_loss_labels)
        training_dataset['12'] = pd.cut(hours, bins=hr_bins, labels=hr_labels)

        training_dataset['2'] = training_dataset['2'].map({
            " ?": 'others',
            " Federal-gov": 'gov', " Local-gov": 'gov',
            " Never-worked":'others',
            " Private": 'others',
            " Self-emp-inc": 'self',
            " Self-emp-not-inc":'self', " State-gov": 'gov',
            " Without-pay": 'others'
        })
        training_dataset['3'] = training_dataset['3'].map({
            " 10th": 'non_college'," 11th": 'non_college',
            " 12th": 'non_college', " 1st-4th": 'non_college',
            " 5th-6th": 'non_college', " 7th-8th": 'non_college',
            " 9th": 'non_college',
            " Assoc-acdm": 'assoc' , " Assoc-voc": 'assoc',
            " Bachelors": 'college',
            " Doctorate": 'grad', " HS-grad": 'grad', " Masters": 'grad',
            " Preschool": 'others',
            " Prof-school": 'others',
            " Some-college": 'college'
        })
        training_dataset['13'] = training_dataset['13'].map({
            ' ?':'others',
            ' Cambodia': 'asia',
            ' Canada': 'north_america',
            ' China': 'asia',
            ' Columbia': 'south_america',
            ' Cuba': 'south_america',
            ' Dominican-Republic': 'south_america',
            ' Ecuador': 'south_america',
            ' El-Salvador': 'south_america',
            ' England': 'europe',
            ' France': 'europe',
            ' Germany':'europe',
            ' Greece':'europe',
            ' Guatemala': 'south_america',
            ' Haiti': 'south_america',
            ' Holand-Netherlands': 'europe',
            ' Honduras': 'south_america' ,
            ' Hong': 'asia',
            ' Hungary':'europe',
            ' India':'asia',
            ' Iran': 'asia',
            ' Ireland': 'europe',
            ' Italy': 'europe',
            ' Jamaica': 'south_america',
            ' Japan': 'asia',
            ' Laos': 'asia',
            ' Mexico': 'south_america',
            ' Nicaragua': 'south_america',
            ' Outlying-US(Guam-USVI-etc)': 'north_america',
            ' Peru': 'south_america',
            ' Philippines': 'asia',
            ' Poland': 'europe',
            ' Portugal': 'europe',
            ' Puerto-Rico': 'south_america',
            ' Scotland':'europe' ,
            ' South': 'asia',
            ' Taiwan': 'asia',
            ' Thailand': 'asia',
            ' Trinadad&Tobago': 'south_america',
            ' United-States': 'north_america',
            ' Vietnam': 'asia', ' Yugoslavia':'europe'
        }
        )

        training_dataset['1'] = training_dataset['1'].astype('category')
        training_dataset['2'] = training_dataset['2'].astype('category')
        training_dataset['3'] = training_dataset['3'].astype('category')
        training_dataset['4'] = training_dataset['4'].astype('category')
        training_dataset['10'] = training_dataset['10'].astype('category')
        training_dataset['11'] = training_dataset['11'].astype('category')
        training_dataset['12'] = training_dataset['12'].astype('category')
        training_dataset['13'] = training_dataset['13'].astype('category')

        return training_dataset
