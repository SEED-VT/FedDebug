"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
from __future__ import print_function
import numpy as np
from scipy.optimize import linear_sum_assignment


def build_init(hungarian_weights, assignments, j):
    """
    Create local network weights from the matched global network weights

    :param hungarian_weights: global network weights
    :type hungarian_weights: `numpy.ndarray`
    :param assignments: assignment matrix mapping local to global neurons
    :type  assignments: `numpy.ndarray`
    :param j: network index for which updated local weights are required
    :type j: `int`
    :return: local network weights.
    :rtype: `list` of `list`
    """
    batch_init = []
    num_assignments = len(assignments)

    if len(hungarian_weights) == 4:
        batch_init.append(hungarian_weights[0][:, assignments[0][j]])
        batch_init.append(hungarian_weights[1][assignments[0][j]])
        batch_init.append(hungarian_weights[2][assignments[0][j]])
        batch_init.append(hungarian_weights[3])
        return batch_init

    for c in range(num_assignments):
        if c == 0:
            batch_init.append(hungarian_weights[c][:, assignments[c][j]])
            batch_init.append(hungarian_weights[c + 1][assignments[c][j]])
        else:
            batch_init.append(hungarian_weights[2 * c]
                              [assignments[c - 1][j]]
                              [:, assignments[c][j]])
            batch_init.append(hungarian_weights[2 * c + 1]
                              [assignments[c][j]])
            if c == num_assignments - 1:
                batch_init.append(hungarian_weights[2 * c + 2]
                                  [assignments[c][j]])
                batch_init.append(hungarian_weights[-1])
                return batch_init


def row_param_cost(global_weights, weights_j_l, global_sigmas, sigma_inv_j):
    """
    Computes the cost defined in equation 6 of PFNM paper
    :param global_weights:
    :type global_weights: `numpy.ndarray`
    :param weights_j_l:
    :type weights_j_l: `numpy.ndarray`
    :param global_sigmas:
    :type global_sigmas: `numpy.ndarray`
    :param sigma_inv_j:
    :type sigma_inv_j: `numpy.ndarray`
    :return Cost value
    :rtype: `numpy.ndarray`
    """

    match_norms = ((weights_j_l + global_weights) ** 2 /
                   (sigma_inv_j + global_sigmas)).sum(axis=1) - \
                  (global_weights ** 2 / global_sigmas).sum(axis=1)
    return match_norms


def compute_cost(global_weights, weights_j, global_sigmas, sigma_inv_j,
                 prior_mean_norm, prior_inv_sigma, popularity_counts, gamma, J):
    """
    Computes full cost to be used by Hungarian algorithm. Refer Equation (8) in the PFNM paper
    :param global_weights: weight matrix
    :type global_weights: `numpy.ndarray`
    :param weights_j: Jth network weights
    :type weights_j: `numpy.ndarray`
    :param global_sigmas: global sigma values
    :type global_sigmas: `numpy.ndarray`
    :param sigma_inv_j: Inverse sigma values for jth network
    :type sigma_inv_j: `numpy.ndarray`
    :param prior_mean_norm: Prior mean values
    :type prior_mean_norm: `numpy.ndarray`
    :param prior_inv_sigma: Prior inverse sigma  values
    :type prior_inv_sigma: `numpy.ndarray`
    :param popularity_counts: popularity count values
    :type popularity_counts: `list`
    :param gamma: Gamma value
    :type gamma: `float`
    :param J: number of clients
    :type J: `int`
    :return: Cost value
    :rtype: `numpy.ndarray`
    """

    Lj = weights_j.shape[0]
    counts = np.minimum(np.array(popularity_counts), 10)
    param_cost = np.array(
        [
            row_param_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j)
            for l in range(Lj)
        ]
    )
    param_cost += np.log(counts / (J - counts))

    # Nonparametric cost
    L = global_weights.shape[0]
    max_added = min(Lj, max(700 - L, 1))

    nonparam_cost = np.outer((((weights_j + prior_mean_norm) ** 2 /
                               (prior_inv_sigma + sigma_inv_j)).sum(axis=1)
                              - (prior_mean_norm ** 2 /
                                 prior_inv_sigma).sum()),
                             np.ones(max_added))
    cost_pois = 2 * np.log(np.arange(1, max_added + 1))
    nonparam_cost -= cost_pois
    nonparam_cost += 2 * np.log(gamma / J)

    full_cost = np.hstack((param_cost, nonparam_cost))

    return full_cost


def matching_upd_j(weights_j, global_weights, sigma_inv_j, global_sigmas,
                   prior_mean_norm, prior_inv_sigma, popularity_counts, gamma, J):
    """
    Computes cost [Equation 8] and solves the linear assignment problem
    using hungarian algo
    :param weights_j: Jth network weights
    :type weights_j: `numpy.ndarray`
    :param global_weights: global network weights
    :type global_weights: `numpy.ndarray`
    :param sigma_inv_j: Inverse sigma values for jth network
    :type sigma_inv_j: `numpy.ndarray`
    :param global_sigmas: Global sigma values
    :type global_sigmas: `numpy.ndarray`
    :param prior_mean_norm: Prior mean values
    :type prior_mean_norm: `numpy.ndarray`
    :param prior_inv_sigma: Prior inverse sigma values
    :type prior_inv_sigma: `numpy.ndarray`
    :param popularity_counts: Popularity counts
    :type popularity_counts: `list`
    :param gamma: Gamma values
    :type gamma: `float`
    :param J: number of clients
    :type J: `int`
    :return a tuple of global weights, sigma values, popularity counts, and assignment values
    :rtype: `tuple`
    """

    L = global_weights.shape[0]

    full_cost = compute_cost(global_weights, weights_j, global_sigmas, sigma_inv_j,
                             prior_mean_norm, prior_inv_sigma, popularity_counts, gamma, J)

    row_ind, col_ind = linear_sum_assignment(-full_cost)

    assignment_j = []
    new_L = L

    for l, i in zip(row_ind, col_ind):
        if i < L:
            popularity_counts[i] += 1
            assignment_j.append(i)
            global_weights[i] += weights_j[l]
            global_sigmas[i] += sigma_inv_j
        else:  # new neuron
            popularity_counts += [1]
            assignment_j.append(new_L)
            new_L += 1
            global_weights = np.vstack((global_weights,
                                        prior_mean_norm + weights_j[l]))
            global_sigmas = np.vstack((global_sigmas,
                                       prior_inv_sigma + sigma_inv_j))

    return global_weights, global_sigmas, popularity_counts, assignment_j


def patch_weights(w_j, L_next, assignment_j_c):
    """
    Patch weights for different layers of the network together into a
    single array of weights
    :param w_j: Jth network weights
    :type w_j: `numpy.ndarray`
    :param L_next: next layer dimensions
    :type L_next: `int`
    :param assignment_j_c: assignment matrix
    :type assignment_j_c: `numpy.ndarray`
    :return: patched weight matrix
    :rtype: `numpy.ndarray`
    """
    if assignment_j_c is None:
        return w_j
    new_w_j = np.zeros((w_j.shape[0], L_next))
    new_w_j[:, assignment_j_c] = w_j
    return new_w_j


def process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0):
    """
    Process the weights of the final classification layer. Since the final
    layer weights are directly associated with classes, these can be
    merged directly without finding an assignment matrix that's required
    for hidden layers
    :param batch_weights: weight matrix
    :type batch_weights: `numpy.ndarray`
    :param last_layer_const: last layer scaling factor
    :type last_layer_const: `list`
    :param sigma: sigma value
    :type sigma: `float`
    :param sigma0: sigma0 value
    :type sigma0: `float`
    :return: a tuple of new softmax bias values and sigma inverse values
    :rtype: `tuple`
    """

    J = len(batch_weights)
    sigma_bias = sigma
    sigma0_bias = sigma0
    mu0_bias = 0.1
    softmax_bias = [batch_weights[j][-1] for j in range(J)]
    softmax_inv_sigma = [s / sigma_bias for s in last_layer_const]
    softmax_bias = sum(
        [b * s for b, s in zip(softmax_bias, softmax_inv_sigma)]
    ) + mu0_bias / sigma0_bias
    softmax_inv_sigma = 1 / sigma0_bias + sum(softmax_inv_sigma)
    return softmax_bias, softmax_inv_sigma


def init_from_assignments(batch_weights_norm, sigma_inv_layer, prior_mean_norm,
                          sigma_inv_prior, assignment):
    """
    Initialize a global network using the assignment matrix of
    local networks found by hungarian algorithm
    :param batch_weights_norm: network weights
    :type batch_weights_norm: `numpy.ndarray`
    :param sigma_inv_layer: inverse sigma values for layer
    :type sigma_inv_layer: `list`
    :param prior_mean_norm: prior mean values
    :type prior_mean_norm: `numpy.ndarray`
    :param sigma_inv_prior: prior sigma inverse values
    :type sigma_inv_prior: `numpy.ndarray`
    :param assignment: assignment matrix
    :type assignment: `numpy.ndarray`
    :return: a tuple of popularity counts, initialized weights, and global sigma
    :rtype: `tuple`
    """

    L = int(max([max(a_j) for a_j in assignment])) + 1
    popularity_counts = [0] * L

    global_weights = np.outer(np.ones(L), prior_mean_norm)
    global_sigmas = np.outer(np.ones(L), sigma_inv_prior)

    for j, a_j in enumerate(assignment):
        for l, i in enumerate(a_j):
            popularity_counts[i] += 1
            global_weights[i] += batch_weights_norm[j][l]
            global_sigmas[i] += sigma_inv_layer[j]

    return popularity_counts, global_weights, global_sigmas


