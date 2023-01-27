"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import numpy as np


def num_pos(y_test):
    """
    Gets number of positive labels in test set.

    :param y_test: Labels of test set
    :type nb_points: `np.array`
    :return: number of positive labels
    :rtype: `int`
    """
    if y_test == []:
        return 0
    else:
        return sum(y_test)


def num_true_pos(y_orig, y_pred):
    """
    Gets number of true positives between test set and model-predicted test set.

    :param y_orig: Labels of test set
    :type y_orig: `np.array`
    :param y_pred: Model-predicted labels of test set
    :type y_pred: `np.array`
    :return: number of true positives
    :rtype: `int`
    """
    a = y_orig == 1
    b = y_pred == 1
    c = np.logical_and(a, b)
    return np.sum(c)


def num_false_pos(y_orig, y_pred):
    """
    Gets number of false positives between test set and model-predicted test set.

    :param y_orig: Labels of test set
    :type y_orig: `np.array`
    :param y_pred: Model-predicted labels of test set
    :type y_pred: `np.array`
    :return: number of false positives
    :rtype: `int`
    """
    a = y_orig == 0
    b = y_pred == 1
    c = np.logical_and(a, b)
    return np.sum(c)


def num_true_neg(y_orig, y_pred):
    """
    Gets number of true negatives between test set and model-predicted test set.

    :param y_orig: Labels of test set
    :type y_orig: `np.array`
    :param y_pred: Model-predicted labels of test set
    :type y_pred: `np.array`
    :return: number of true negatives
    :rtype: `int`
    """
    a = y_orig == 0
    b = y_pred == 0
    c = np.logical_and(a, b)
    return np.sum(c)


def num_false_neg(y_orig, y_pred):
    """
    Gets number of false negatives between test set and model-predicted test set.

    :param y_orig: Labels of test set
    :type y_orig: `np.array`
    :param y_pred: Model-predicted labels of test set
    :type y_pred: `np.array`
    :return: number of false negatives
    :rtype: `int`
    """
    a = y_orig == 1
    b = y_pred == 0
    c = np.logical_and(a, b)
    return np.sum(c)


def tp_rate(TP, pos):
    """
    Gets true positive rate.

    :param TP: Number of true positives
    :type TP: `int`
    :param pos: Number of positive labels
    :type pos: `int`
    :return: true positive rate
    :rtype: `float`
    """
    if pos == 0:
        return 0
    else:
        return TP / pos


def tn_rate(TN, neg):
    """
    Gets true positive rate.

    :param: TP: Number of true negatives
    :type TN: `int`
    :param: pos: Number of negative labels
    :type neg: `int`
    :return: true negative rate
    :rtype: `float`
    """
    if neg == 0:
        return 0
    else:
        return TN / neg


def fp_rate(FP, neg):
    """
    Gets false positive rate.

    :param: FP: Number of false positives
    :type FP: `int`
    :param: neg: Number of negative labels
    :type neg: `int`
    :return: false positive rate
    :rtype: `float`
    """
    if neg == 0:
        return 0
    else:
        return FP / neg


def pp_value(TP, FP):
    """
    Gets positive predictive value, or precision.

    :param: TP: Number of true positives
    :type TP: `int`
    :param: FP:  Number of false positives
    :type FP: `int`
    :return: positive predictive value
    :rtype: `float`
    """
    if TP == 0 and FP == 0:
        return 0
    else:
        return TP / (TP + FP)


def fav_rate(y_pred_group):
    """
    Gets rate of favorable outcome.

    :param y_pred_group: Model-predicted labels of test set for privileged/unprivileged group
    :type y_pred_group: `np.array`
    :return: rate of favorable outcome
    :rtype: `float`
    """
    if y_pred_group == []:
        return 0
    else:
        return num_pos(y_pred_group) / y_pred_group.shape[0]


def stat_parity_diff(fav_rate_unpriv, fav_rate_priv):
    """
    Gets statistical parity difference between the unprivileged and privileged groups.

    :param fav_rate_unpriv: rate of favorable outcome for unprivileged group
    :type fav_rate_unpriv: `float`
    :param fav_rate_priv: rate of favorable outcome for privileged group
    :type fav_rate_priv: `float`
    :return: statistical parity difference
    :rtype: `float`
    """
    return fav_rate_unpriv - fav_rate_priv


def equal_opp_diff(TPR_unpriv, TPR_priv):
    """
     Gets equal opportunity difference between the unprivileged and privileged groups.

    :param: TPR_unpriv: true positive rate for unprivileged group
    :type TPR_unpriv: `float`
    :param: TPR_priv: true positive rate for privileged group
    :type TPR_priv: `float`
    :return: equal opportunity difference
    :rtype: `float`
    """
    return TPR_unpriv - TPR_priv


def avg_odds(FPR_unpriv, FPR_priv, TPR_unpriv, TPR_priv):
    """
    Gets average odds between the unprivileged and privileged groups.

    :param: FPR_unpriv: false positive rate for unprivileged group
    :type FPR_unpriv: `float`
    :param: FPR_priv: false positive rate for privileged group
    :type FPR_priv: `float`
    :param: TPR_unpriv: true positive rate for unprivileged group
    :type TPR_unpriv: `float`
    :param: TPR_priv: true positive rate for privileged group
    :type TPR_priv: `float`
    :return: average odds
    :rtype: `float`
    """
    return ((FPR_unpriv - FPR_priv) + (TPR_unpriv - TPR_priv)) / 2


def disparate_impact(fav_rate_unpriv, fav_rate_priv):
    """
    Gets disparate impact between the unprivileged and privileged groups.

    :param fav_rate_unpriv: rate of favorable outcome for unprivileged group
    :type fav_rate_priv: `float`
    :param fav_rate_priv: rate of favorable outcome for privileged group
    :type fav_rate_unpriv: `float`
    :return: disparate impact
    :rtype: `float`
    """
    if fav_rate_priv == 0:
        return 0
    else:
        return fav_rate_unpriv / fav_rate_priv
