"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.metrics import confusion_matrix, f1_score, precision_score, \
    recall_score, classification_report
from sklearn.metrics import explained_variance_score, mean_squared_error, \
    mean_squared_log_error, median_absolute_error, median_absolute_error, \
    mean_absolute_error, r2_score
from sklearn.utils.multiclass import unique_labels

logger = logging.getLogger(__name__)


def _f1_precision_recall(y_true, y_pred):
    """Compute the f1, precision, recall based on the average method. \
        Uses sklearn util to compute this metric.
    :param y_true: 1d array-like, with ground truth (correct target values)
    :type y_true: `array`
    :param y_pred: 1d array-like, estimation returned by classifier.
    :type y_pred: `array`
    :param average: Required for multiclass/multilabel targets.
    :type average: `str`
    :return: f1, precision, recall or array of float
    :rtype: `float`, `float`, `float`
    """
    metrics = {}
    round_digits = 2
    try:
        metrics['f1'] = round(
            f1_score(y_true, y_pred, zero_division=0), round_digits)
        metrics['precision'] = round(
            precision_score(y_true, y_pred, zero_division=0), round_digits)
        metrics['recall'] = round(recall_score(
            y_true, y_pred, zero_division=0), round_digits)

    except ValueError as ve:
        logger.exception(ve)
        logger.info(
            "Error occurred while collecting binary classification metircs.")
        labels = unique_labels(y_true, y_pred)
        if(len(labels) > 2):
            logger.info(len(labels) + " labels given, expected number 2")

    except Exception as ex:
        logger.exception(ex)

    return metrics


def get_binary_classification_metrics(y_true, y_pred, eval_metrics=[]):
    """Compute and package different metrics for binary classification, \
    and return a dictionary with metrics
    :param y_true: 1d array-like, with ground truth (correct target values)
    :type y_true: `array`
    :param y_pred: 1d array-like, estimation returned by classifier.
    :type y_pred: `array`
    :param metrics: metrics requested by the user
    :type metrics: list of metrics which needs to be sent back.
    :return: dictionary with metrics
    :rtype: `dict`
    """
    metrics = {}
    round_digits = 2
    metrics = _f1_precision_recall(y_true, y_pred)
    try:
        labels = unique_labels(y_true)
        metrics['average precision'] = round(
            average_precision_score(y_true, y_pred), round_digits)
        if len(labels) > 1:
            metrics['roc auc'] = round(
                roc_auc_score(y_true, y_pred), round_digits)
            metrics['negative log loss'] = round(
                log_loss(y_true, y_pred), round_digits)
    except ValueError as ve:
        logger.exception(ve)
        logger.info(
            "Error occurred while collecting binary classification metircs.")
        labels = unique_labels(y_true, y_pred)
        if(len(labels) > 2):
            logger.error(len(labels) + " labels given, expected number 2")

    except Exception as ex:
        logger.exception(ex)

    # for metric in eval_metrics:
    # process each requested metric and send back the results.

    return metrics


def get_multi_label_classification_metrics(y_true, y_pred, eval_metrics=[]):
    """Compute and package different metrics for multi label classification, \
    and return a dictionary with metrics
    :param y_true: 1d array-like, with ground truth (correct target values)
    :type y_true: `array`
    :param y_pred: 1d array-like, estimation returned by classifier.
    :type y_pred: `array`
    :param metrics: metrics requested by the user
    :type metrics: list of metrics which needs to be sent back.
    :return: dictionary with metrics
    :rtype: `dict`
    """
    metrics = {}
    round_digits = 2
    multilabel_average_options = ['micro', 'macro', 'weighted']

    for avg in multilabel_average_options:

        try:
            metrics['f1 ' +
                    avg] = round(f1_score(y_true, y_pred, average=avg, zero_division=0), round_digits)
            metrics['precision ' +
                    avg] = round(precision_score(y_true, y_pred, average=avg, zero_division=0), round_digits)
            metrics['recall ' +
                    avg] = round(recall_score(y_true, y_pred, average=avg, zero_division=0), round_digits)
        except Exception as ex:
            logger.exception(ex)

    return metrics


def get_eval_metrics_for_classificaton(y_true, y_pred, eval_metrics={}):
    """Compute and package different metrics for classification problem, \
    and return a dictionary with metrics
    :param y_true: 1d array-like, with ground truth (correct target values)
    :type y_true: `array`
    :param y_pred: 1d array-like, estimation returned by classifier.
    :type y_pred: `array`
    :param eval_metrics: metrics requested by the user
    :type eval_metrics: list of metrics which needs to be sent back.
    :return: dictionary with metrics
    :rtype: `dict`
    """
    metrics = {}
    try:
        y_pred = (np.asarray(y_pred)).round()
        y_true = (np.asarray(y_true)).round()

        # reshape for tensorflow2.1 predictions
        if y_pred[0].shape != y_true[0].shape:  # FIXME
            logger.info("reshaping y_pred")
            y_pred = list(np.argmax(y_pred, axis=-1))

        # TODO: handle mix types labels {'multilabel-indicator', 'multiclass'}
        labels = unique_labels(y_true, y_pred)

        if len(labels) <= 2:
            metrics = get_binary_classification_metrics(
                y_true, y_pred, eval_metrics)
        else:
            metrics = get_multi_label_classification_metrics(
                y_true, y_pred, metrics)

    except Exception as ex:
        logger.exception(ex)
        logger.exception("Error occurred while evaluating metrics")

    return metrics


def get_eval_metrics_for_regression(y_true, y_pred, eval_metrics={}):
    """Compute and package different metrics for regression problem, \
    and return a dictionary with metrics
    :param y_true: 1d array-like, with ground truth (correct target values)
    :type y_true: `array`
    :param y_pred: 1d array-like, estimation returned by classifier.
    :type y_pred: `array`
    :param eval_metrics: metrics requested by the user
    :type eval_metrics: list of metrics which needs to be sent back.
    :return: dictionary with metrics
    :rtype: `dict`
    """

    metrics = {}

    try:
        metrics['Negative root mean squared error'] = round(mean_squared_error(
            y_true, y_pred, squared=False), 2)
        metrics['Negative mean absolute error'] = round(mean_absolute_error(
            y_true, y_pred), 2)
        metrics['Negative root mean squared log error'] = round(mean_squared_log_error(
            y_true, y_pred), 2)
        metrics['Explained Variance'] = round(
            explained_variance_score(y_true, y_pred), 2)
        metrics['Negative mean squared error'] = round(mean_squared_error(
            y_true, y_pred, squared=True), 2)
        metrics['Negative mean squared log error'] = round(np.sqrt(mean_squared_log_error(
            y_true, y_pred)), 2)
        metrics['Negative median absolute error'] = round(median_absolute_error(
            y_true, y_pred), 2)
        metrics['R2'] = round(r2_score(y_true, y_pred), 2)
    except Exception as ex:
        logger.error("Exception occurred while calculating metrics.")
        logger.error(ex)

    return metrics
