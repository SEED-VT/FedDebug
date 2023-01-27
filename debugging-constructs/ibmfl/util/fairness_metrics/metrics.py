"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import numpy as np
import pandas as pd
from ibmfl.util.fairness_metrics.confusionmatrix import num_true_pos, num_false_pos, \
    num_pos, fav_rate, tp_rate, fp_rate
from ibmfl.util.fairness_metrics.confusionmatrix import stat_parity_diff, equal_opp_diff, avg_odds, disparate_impact
from sklearn.metrics import f1_score
from gensim.matutils import hellinger


def priv_unpriv_sets(training_data, y_test, y_pred, sensitive_attribute, cols):
    """
    Splits y_test and y_pred into two arrays each, depending on whether the sample associated
    with the label was privileged or unprivileged, with respect to the sensitive attribute.

    :param training_data: Feature set
    :type training_data: `np.array`
    :param y_test: Test set labels
    :type y_test: `np.array`
    :param y_pred: Test set predictions
    :type y_pred: `np.array`
    :param sensitive_attribute:
    :type sensitive_attribute: `str`
    :param cols: Feature set column names
    :type cols: `list`
    :return: privileged and unprivileged y label groups
    :rtype: `float`
    """
    training_data = pd.DataFrame(data=training_data)
    training_data.columns = cols

    p_set = training_data.loc[training_data[sensitive_attribute] == 1]
    unp_set = training_data.loc[training_data[sensitive_attribute] == 0]

    a = p_set.index.tolist()
    b = unp_set.index.tolist()

    return y_test[a], y_test[b], y_pred[a], y_pred[b]


def get_fairness_metrics(x_test, y_test, y_test_pred, SENSITIVE_ATTRIBUTE, cols):
    """
    Calculates middle terms for fairness metrics.

    :param x_test: Test feature set
    :type x_test: `np.array`
    :param y_test: Test set labels
    :type y_test: `np.array`
    :param y_test_pred: Test set predictions
    :type y_test_pred: `np.array`
    :param SENSITIVE_ATTRIBUTE:
    :type SENSITIVE_ATTRIBUTE: `str`
    :param cols: Feature set column namess
    :type cols: `list`
    :return: fairness metric variables
    :rtype: `float`
    """
    y_test_priv_set, y_test_unpriv_set, y_pred_priv_set, y_pred_unpriv_set = \
    priv_unpriv_sets(x_test, y_test, y_test_pred, SENSITIVE_ATTRIBUTE, cols)

    pos_unpriv_set = num_pos(y_test_unpriv_set)
    neg_unpriv_set = y_test_unpriv_set.shape[0] - pos_unpriv_set
    pos_priv_set = num_pos(y_test_priv_set)
    neg_priv_set = y_test_priv_set.shape[0] - pos_priv_set

    TP_unpriv = num_true_pos(y_test_unpriv_set, y_pred_unpriv_set)
    TP_priv = num_true_pos(y_test_priv_set, y_pred_priv_set)
    FP_unpriv = num_false_pos(y_test_unpriv_set, y_pred_unpriv_set)
    FP_priv = num_false_pos(y_test_priv_set, y_pred_priv_set)

    TPR_unpriv = tp_rate(TP_unpriv, pos_unpriv_set)
    TPR_priv = tp_rate(TP_priv, pos_priv_set)
    FPR_unpriv = fp_rate(FP_unpriv, neg_unpriv_set)
    FPR_priv = fp_rate(FP_priv, neg_priv_set)

    fav_rate_unpriv = fav_rate(y_pred_unpriv_set)
    fav_rate_priv = fav_rate(y_pred_priv_set)

    return fav_rate_unpriv, fav_rate_priv, TPR_unpriv, TPR_priv, FPR_unpriv, FPR_priv


def fairness_report(x_test, y_test, y_pred, sensitive_attribute, cols):
    """
    Gets fairness report, with F1 score, statistical parity difference, equal opportunity odds,
    average odds difference and disparate impact.

    :param x_test: Test feature set
    :type x_test: `np.array`
    :param y_test: Test set labels
    :type y_test: `np.array`
    :param y_pred: Test set predictions
    :type y__pred: `np.array`
    :param sensitive_attribute:
    :type sensitive_attribute: `str`
    :param cols: Feature set column namess
    :type cols: `list`
    :return: report
    :rtype: `dict`
    """

    fav_rate_unpriv, fav_rate_priv, TPR_unpriv, TPR_priv, FPR_unpriv, FPR_priv = get_fairness_metrics(
        x_test, y_test, y_pred, sensitive_attribute, cols)

    f1 = f1_score(y_test, y_pred)
    spd = stat_parity_diff(fav_rate_unpriv, fav_rate_priv)
    eod = equal_opp_diff(TPR_unpriv, TPR_priv)
    ao = avg_odds(FPR_unpriv, FPR_priv, TPR_unpriv, TPR_priv)
    di = disparate_impact(fav_rate_unpriv, fav_rate_priv)

    report = {'F1': f1, 'Statistical Parity Difference:': spd, 'Equal Opportunity Difference': eod,
              'Average Odds Difference:': ao, 'Disparate Impact:': di}

    return report


def uei(y_train, y_train_pred):
    """
    Gets UEI index between training set labels and training set predictions.

    :param y_train:
    :type y_train: `np.array`
    :param y_train_pred:
    :type y_train_pred: `np.array`
    :return: UEI index
    :rtype: `float`
    """

    y_train_norm = y_train / np.sum(y_train)
    if np.sum(y_train_pred) > 0:
        y_train_pred_norm = y_train_pred / np.sum(y_train_pred)
    else:
        y_train_pred_norm = y_train_pred

    return hellinger(y_train_norm, y_train_pred_norm)
