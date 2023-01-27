"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging
import abc
import numpy

from ibmfl.data.data_util import get_min, get_max, get_mean,\
    get_var, get_std, get_quantile, get_normalizer, get_standardscaler, \
    get_minmaxscaler
from ibmfl.exceptions import FLException

logger = logging.getLogger(__name__)


class DataHandler(abc.ABC):
    """
    Base class to load and pre-process data.
    """

    def __init__(self, **kwargs):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.preprocessor = None
        self.testing_generator = None
        self.training_generator = None

    @abc.abstractmethod
    def get_data(self, **kwargs):
        """
        Access the local dataset and return the training and testing dataset
        as a tuple.

        :param kwargs:
        :return: `tuple`. (training_set, testing_set)
        """
        raise NotImplementedError


    def get_train_counts(self):
        """
        Returns the training sample size.

        :return: The training sample size
        :rtype: `int`
        """
        if self.x_train is not None and isinstance(self.x_train, numpy.ndarray):
            return self.x_train.shape[0]
        else:
            logger.info('The attribute x_train is None or is not of numpy.ndarray format!'
                'Trying to access training counts assuming data is loaded \
                as DataGenerator type.')
            try:
                counts = len(self.training_generator.filenames)
                assert isinstance(counts, int)
                return counts
            except:
                raise FLException("Error ocurred during accessing the training sample size.")


    def get_val_data(self, **kwargs):
        """
        Access the local dataset and return the validation dataset
        as a tuple.
        :param kwargs:
        :return: `tuple`. (validation_set)
        """
        pass

    def get_statistics_of_training_data(self, sample_data_schema,
                                        lst_stats_name, **kwargs):
        """
        Return the corresponding statistics, which is specified by the
        provided list of statistics names, of the local training dataset.

        :param sample_data_schema: Provided data with only feature values. \
        Assuming the dataset has shape (num_samples, num_features).
        :type sample_data_schema: `np.array`
        :param lst_stats_name: A list of statistics names, \
        all in lowercase form, for example, ['min'], ['mean', 'variance'], etc.
        :type lst_stats_name: `list` of `str`
        :param kwargs: Additional parameters to obtain the statistics.
        :type kwargs: `dict`
        :return: The requested statistics based on the local dataset.
        :rtype: `dict`
        """
        if sample_data_schema is None and self.x_train is None:
            raise FLException("No data is provided!")
        elif sample_data_schema is None and self.x_train is not None:
            sample_data_schema = self.x_train
            logger.warning('No dataset is provided, use the data schema from '
                           'training dataset(x_train) as the default one.')

        if type(lst_stats_name) is not list:
            raise FLException("list of requested statistics badly form. "
                              "It should of type list, but it is instead "
                              "of type " + str(type(lst_stats_name)))
        list_stats = {}
        if not isinstance(sample_data_schema, numpy.ndarray):
            raise FLException("Expecting the local dataset to be of type "
                              "numpy.ndarray, instead it is of type " +
                              str(type(sample_data_schema)))

        while len(lst_stats_name) != 0:
            tmp_stat_name = lst_stats_name.pop()
            if type(tmp_stat_name) is not str:
                logger.warning("Skipping the current requested statistics "
                               "name. It should be of type string, "
                               "but now it is instead of type" +
                               str(type(tmp_stat_name)))
            elif tmp_stat_name == 'min' or tmp_stat_name == 'minimum':
                list_stats[tmp_stat_name] = get_min(sample_data_schema)
            elif tmp_stat_name == 'max' or tmp_stat_name == 'maximum':
                list_stats[tmp_stat_name] = get_max(sample_data_schema)
            elif tmp_stat_name == 'mean':
                list_stats[tmp_stat_name] = get_mean(sample_data_schema)
            elif tmp_stat_name == 'var' or tmp_stat_name == 'variance':
                list_stats[tmp_stat_name] = get_var(sample_data_schema)
            elif tmp_stat_name == 'std' or \
                    tmp_stat_name == 'standard deviation':
                list_stats[tmp_stat_name] = get_std(sample_data_schema)
            elif tmp_stat_name == 'quantile':
                if kwargs and 'q' in kwargs:
                    list_stats[tmp_stat_name] = get_quantile(
                        sample_data_schema,
                        percentage=kwargs['q'])
                else:
                    raise FLException('Cannot compute quantile, '
                                      'missing quantile requirement.')
            else:
                logger.warning("Current required statistics "
                               "is not supported. Skipping...")

        return list_stats

    def get_preprocessor(self, sample_data_schema, preprocessor_name, **kwargs):
        """
        Set the data preprocessor of the data handler class as the requested
        type of preprocessor. The supported preprocessors
        include `normalizer`, `standardscaler` and `minmaxscaler`.
        All provided based on `sklearn.preprocessing` module.
        The preprocessor can be applied to perform the
        required preprocessing step for the party's local dataset
        via `transform` method.

        :param sample_data_schema: Provided data with only feature values to \
        initialize the preprocessor. \
        Assuming the dataset has shape (num_samples, num_features).
        :type sample_data_schema: `np.array`
        :param preprocessor_name: The requested preprocessor name in lowercase.
        :type preprocessor_name: `str`
        :param kwargs: Additional parameters to obtain the preprocessor.
        :type kwargs: `dict`
        :return: None
        """
        if sample_data_schema is None and self.x_train is None:
            raise FLException("No data is provided!")
        elif sample_data_schema is None and self.x_train is not None:
            sample_data_schema = self.x_train
            logger.warning('No dataset is provided, use the data schema from '
                           'training dataset(x_train) as the default one.')

        if type(preprocessor_name) is not str:
            raise FLException("Expecting the requested preprocessor "
                              "to be of type string, instead it is of type" +
                              str(type(preprocessor_name)))

        if not isinstance(sample_data_schema, numpy.ndarray):
            raise FLException("Expecting the local dataset to be of type "
                              "numpy.ndarray, instead it is of type " +
                              str(type(sample_data_schema)))

        if preprocessor_name == 'normalizer' or \
                preprocessor_name == 'normalization':
            if 'norm' in kwargs:
                self.preprocessor = get_normalizer(sample_data_schema,
                                                   norm=kwargs['norm'])
            else:
                self.preprocessor = get_normalizer(sample_data_schema)
        elif preprocessor_name == 'standardscaler' or \
                preprocessor_name == 'standardization':
            mean_val = None
            std = None
            if 'mean' in kwargs:
                mean_val = kwargs['mean']
            if 'scale' in kwargs:
                std = kwargs['scale']
            self.preprocessor = get_standardscaler(sample_data_schema,
                                                   mean_val=mean_val,
                                                   std=std)
        elif preprocessor_name == 'minmaxscaler':
            if 'feature_range' in kwargs:
                self.preprocessor = get_minmaxscaler(
                    sample_data_schema,
                    feature_range=kwargs['feature_range'])
            else:
                self.preprocessor = get_minmaxscaler(sample_data_schema)
        else:
            logger.warning("Required preprocessor is not supported. "
                           "Skipping...")
