"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Module providing utility functions helpful for preproccessing data
"""
import numpy as np
import logging

from ibmfl.exceptions import FLException

logger = logging.getLogger(__name__)

# TODO get dp stats


def get_min(data, **kwargs):
    """
    Assuming the dataset is loaded as type `np.array`, and has shape
     (num_samples, num_features).

    :param data: Provided dataset, assume each row is a data sample and
    each column is one feature.
    :type data: `np.ndarray`
    :param kwargs: Dictionary of differential privacy arguments \
    for computing the minimum value of each feature across all samples, \
    e.g., epsilon and delta, etc. \
    :type kwargs: `dict`
    :return: A vector of shape (1, num_features) stores the minimum value \
    of each feature across all samples.
    :rtype: `np.array` of `float`
    """
    try:
        min_vec = np.min(data, axis=0)
    except Exception as ex:
        raise FLException('Error occurred when calculating '
                          'the minimum value. ' + str(ex))
    return min_vec


def get_max(data, **kwargs):
    """
    Assuming the dataset is loaded as type `np.array`, and has shape
     (num_samples, num_features).

    :param data: Provided dataset, assume each row is a data sample and \
    each column is one feature.
    :type `np.ndarray`
    :param kwargs: Dictionary of differential privacy arguments \
    for computing the maximum value of each feature across all samples, \
    e.g., epsilon and delta, etc.
    :type kwargs: `dict`
    :return: A vector of shape (1, num_features) stores the maximum value \
    of each feature across all samples.
    :rtype: `np.array` of `float`
    """
    try:
        max_vec = np.max(data, axis=0)
    except Exception as ex:
        raise FLException('Error occurred when calculating '
                          'the maximum value. ' + str(ex))
    return max_vec


def get_mean(data, **kwargs):
    """
    Assuming the dataset is loaded as type `np.array`, and has shape
     (num_samples, num_features).

    :param data: Provided dataset, assume each row is a data sample and \
    each column is one feature.
    :type data: `np.ndarray`
    :param kwargs: Dictionary of differential privacy arguments \
    for computing the maximum value of each feature across all samples, \
    e.g., epsilon and delta, etc.
    :type kwargs: `dict`
    :return: A vector of shape (1, num_features) stores the maximum value \
    of each feature across all samples.
    :rtype: `np.array` of `float`
    """
    try:
        var_vec = np.var(data, axis=0)
    except Exception as ex:
        raise FLException('Error occurred when calculating '
                          'the mean value. ' + str(ex))
    return var_vec


def get_var(data, **kwargs):
    """
    Assuming the dataset is loaded as type `np.array`, and has shape
     (num_samples, num_features).

    :param data: Provided dataset, assume each row is a data sample and \
    each column is one feature.
    :type data: `np.ndarray`
    :param kwargs: Dictionary of differential privacy arguments \
    for computing the variance of each feature across all samples, \
    e.g., epsilon, etc.
    :type kwargs: `dict`
    :return: A vector of shape (1, num_features) stores the variance
    of each feature across all samples.
    :rtype: `np.array` of `float`
    """
    try:
        var_vec = np.var(data, axis=0)
    except Exception as ex:
        raise FLException('Error occurred when calculating '
                          'the variance. ' + str(ex))
    return var_vec


def get_std(data, **kwargs):
    """
    Assuming the dataset is loaded as type `np.array`, and has shape
     (num_samples, num_features).

    :param data: Provided dataset, assume each row is a data sample and \
    each column is one feature.
    :type data: `np.ndarray`
    :param kwargs: Dictionary of differential privacy arguments \
    for computing the standard deviation of each feature across all samples, \
    e.g., epsilon, etc.
    :type kwargs: `dict`
    :return: A vector of shape (1, num_features) stores the
    standard deviation of each feature across all samples.
    :rtype: `np.array` of `float`
    """
    try:
        std_vec = np.std(data, axis=0)
    except Exception as ex:
        raise FLException('Error occurred when calculating '
                          'the standard deviation. ' + str(ex))
    return std_vec


def get_quantile(data, percentage, **kwargs):
    """
    Assuming the dataset is loaded as type `np.array`, and has shape
     (num_samples, num_features).

    :param data: Provided dataset, assume each row is a data sample and \
    each column is one feature.
    :type data: `np.ndarray`
    :param percentage: Quantile or sequence of quantiles to compute, \
    which must be between 0 and 1 inclusive.
    :type percentage: `float` or `np.array` of `float`
    :param kwargs: Dictionary of differential privacy arguments \
    for computing the specified quantile of each feature across all samples, \
    e.g., epsilon, etc.
    :type kwargs: `dict`
    :return: A vector of shape (1, num_features) stores the
    standard deviation of each feature across all samples.
    :rtype: `np.array` of `float`
    """
    try:
        quantile_vec = np.quantile(data, q=percentage, axis=0)
    except Exception as ex:
        raise FLException('Error occurred when calculating '
                          'the quantile. ' + str(ex))
    return quantile_vec


def get_normalizer(data, norm='l2'):
    """
    Obtain the normalizer that perform the normalization preprocessing
    technique across all features via sklearn.preprocessing.normalizer API.
    A normalizer will scale a dataset w.r.t. features to unit norm.

    :param data: Provided dataset, assume each row is a data sample and \
    each column is one feature.
    :type data: `np.ndarray`
    :param norm: The norm to use to normalize each non zero sample.
    By default, norm is set to `l2`.
    :type norm: `str`
    :return: The normalizer preprocessor that can be applied to perform \
    normalizing preprocessing step for the party's local dataset \
    via `transform` method.
    :rtype: `sklearn.preprocessing.Normalizer`
    """
    from sklearn import preprocessing
    try:
        normalizer = preprocessing.Normalizer(norm=norm).fit(data)

        # test the normalizer
        normalizer.transform(data)
    except Exception as ex:
        raise FLException('Error occurred when obtaining '
                          'the normalizer. ' + str(ex))
    return normalizer


def get_standardscaler(data, mean_val=None, std=None):
    """
    Obtain the StandardScaler that perform the standardization preprocessing
    technique with provided mean and standard deviation values
    via sklearn.preprocessing.scaler API.
    A StandardScaler will standardize a dataset along any axis, and
    center to the mean and component wise scale to unit variance.

    :param data: Provided dataset, assume each row is a data sample and \
    each column is one feature.
    :type data: `np.ndarray`
    :param mean_val: (Optional) A vector of mean values one wants to scale \
    the dataset. \
    The vector should be of shape (1, num_features).
    :type mean_val: `np.ndarray`
    :param std: (Optional) A vector of standard deviation values \
    one wants to scale the dataset. \
    The vector should be of shape (1, num_features).
    :type std: `np.ndarray`
    :return: The standard scaler preprocessor that can be applied to perform \
    standardization preprocessing step for the party's local dataset \
    via `transform` method.
    :rtype: 'sklearn.preprocessing.StandardScaler'
    """
    from sklearn import preprocessing
    try:
        scaler = preprocessing.StandardScaler().fit(data)

        # set scaler with correct mean_val and std values
        if mean_val is not None:
            logger.info("Set mean_val value of the StandardScaler "
                        "as the provided mean...")
            scaler.mean_ = mean_val
        if std is not None:
            logger.info("Set standard deviation of the StandardScaler "
                        "as the provided standard deviation...")
            scaler.scale_ = std
    except Exception as ex:
        raise FLException('Error occurred when obtaining '
                          'the standardscaler. ' + str(ex))

    return scaler


def get_minmaxscaler(data, feature_range=(0, 1)):
    """
    Obtain a MinMaxScaler that perform the MinMaxScale preprocessing technique
     with provided feature range via sklearn.preprocessing.minmax_scale API.

    A MinMaxScaler will transforms features by scaling each feature to
    a given range.
    This estimator scales and translates each feature individually
    such that it is in the given range on the training set,
    i.e. between zero and one.

    :param data: Provided dataset, assume each row is a data sample and \
    each column is one feature.
    :type data: `np.ndarray`
    :param feature_range: Desired range of transformed data.
    :type feature_range: tuple (min, max), default=(0, 1)
    :return: The minmaxscaler preprocessor that can be applied to perform \
    minmax scaling preprocessing step for the party's local dataset \
    via `transform` method.
    :rtype: `sklearn.preprocessing.MinMaxScaler`
    """
    from sklearn import preprocessing
    try:
        scaler = preprocessing.MinMaxScaler(feature_range=feature_range).\
            fit(data)
    except Exception as ex:
        raise FLException('Error occurred when obtaining '
                          'the minmaxcaler. ' + str(ex))
    return scaler


def get_reweighing_weights(data, sensitive_attribute, columns):
    """
    Calculates reweighing weights for points, assuming:
    * privileged group has sensitive attribute value = 1, unprivileged group is 0
    * positive class has value = 1, negative class is 0
    weight = P_expected(sensitive_attribute & class)/P_observed(sensitive_attribute & class)

    :param data: Provided dataset, assume each row is a data sample and
    each column is one feature.
    :type data: `tuple`
    :param sensitive_attribute: Sensitive attribute
    :type sensitive_attribute: `str`
    :param columns: dataset column names
    :type columns: `list`
    :return: weights
    :rtype: `np.array`
    """
    # additional imports for reweighing algorithms
    import pandas as pd

    (features, labels) = data

    training_dataset = pd.DataFrame(data=features)
    class_values = labels.tolist()

    training_dataset.columns = columns
    training_dataset['class'] = class_values

    nrows = training_dataset.shape[0]

    priv = sum(training_dataset[sensitive_attribute])/nrows
    unpriv = nrows - sum(training_dataset[sensitive_attribute])/nrows
    pos = sum(training_dataset['class'])/nrows
    neg = nrows - sum(training_dataset['class'])/nrows

    tmp_unp_train_data = training_dataset[training_dataset[sensitive_attribute] == 0]
    tmp_p_train_data = training_dataset[training_dataset[sensitive_attribute] == 1]

    if len(tmp_unp_train_data) > 0:
        unpriv_neg = tmp_unp_train_data['class'].value_counts()[0]/nrows
        if sum(tmp_unp_train_data['class']) > 0:
            unpriv_pos = tmp_unp_train_data['class'].value_counts()[1]/nrows
        else:
            unpriv_pos = 0
    else:
        unpriv_neg = 0
    if len(tmp_p_train_data) > 0:
        priv_neg = tmp_p_train_data['class'].value_counts()[0]/nrows
        if sum(tmp_p_train_data['class']) > 0:
            priv_pos = tmp_p_train_data['class'].value_counts()[1]/nrows
        else:
            priv_pos = 0
    else:
        priv_pos = 0

    weight = []
    for index, row in training_dataset.iterrows():
        if row[sensitive_attribute] == 0 and row['class'] == 0:
            weight.append(unpriv * neg / unpriv_neg)
        elif row[sensitive_attribute] == 0 and row['class'] == 1:
            weight.append(unpriv * pos / unpriv_pos)
        elif row[sensitive_attribute] == 1 and row['class'] == 0:
            weight.append(priv * neg / priv_neg)
        elif row[sensitive_attribute] == 1 and row['class'] == 1:
            weight.append(priv * pos / priv_pos)

    return np.array(weight)

def get_hist_counts(data, sensitive_attribute, columns, eps):
    """
    Calculates noisy counts for reweighing with differential privacy (epsilon set in
    datahandler), assuming:
    * privileged group has sensitive attribute value = 1, unprivileged group is 0
    * positive class has value = 1, negative class is 0
    weight = P_expected(sensitive_attribute & class)/P_observed(sensitive_attribute & class)

    :param data: Provided dataset, assume each row is a data sample and
    each column is one feature.
    :type data: `tuple` of `nd.array`
    :param sensitive_attribute: Sensitive attribute
    :type sensitive_attribute: `str`
    :param columns: dataset column names
    :type columns: `list`
    :param eps: epsilon utilized for differential privacy
    :type eps: `float`
    :return: counts
    :rtype: `dict`
    """
    # additional imports for global reweighing with differential privacy
    import pandas as pd
    from diffprivlib import tools as dp

    (features, labels) = data
    counts = {}

    training_dataset = pd.DataFrame(data=features)
    class_values = labels.tolist()

    training_dataset.columns = columns
    training_dataset['class'] = class_values

    data_whole = training_dataset[[sensitive_attribute, 'class']]
    data_whole_0 = data_whole[data_whole[sensitive_attribute] == 0]
    data_whole_0_t = data_whole_0.T

    data_whole_0_t = np.array(data_whole_0_t)
    dp_hist_0, dp_bins_0 = dp.histogram(data_whole_0_t[1], bins=2, epsilon=eps)
    counts['unp_neg'] = dp_hist_0[0]
    counts['unp_pos'] = dp_hist_0[1]

    data_whole_1 = data_whole[data_whole[sensitive_attribute] == 1]
    data_whole_1_t = data_whole_1.T

    data_whole_1_t = np.array(data_whole_1_t)
    dp_hist_1, dp_bins_1 = dp.histogram(data_whole_1_t[1], bins=2, epsilon=eps)
    counts['p_neg'] = dp_hist_1[0]
    counts['p_pos'] = dp_hist_1[1]

    tmp_unp_train_data = training_dataset[training_dataset[sensitive_attribute] == 0]
    tmp_p_train_data = training_dataset[training_dataset[sensitive_attribute] == 1]
    unpriv_neg = tmp_unp_train_data['class'].value_counts()[0]
    unpriv_pos = tmp_unp_train_data['class'].value_counts()[1]
    priv_neg = tmp_p_train_data['class'].value_counts()[0]
    priv_pos = tmp_p_train_data['class'].value_counts()[1]

    counts['unpriv_neg'] = unpriv_neg
    counts['unpriv_pos'] = unpriv_pos
    counts['priv_neg'] = priv_neg
    counts['priv_pos'] = priv_pos

    return counts
