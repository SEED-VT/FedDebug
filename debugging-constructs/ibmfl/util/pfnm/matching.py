"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
from __future__ import print_function
import logging
import numpy as np

from ibmfl.util.pfnm import core as pfnm_core


logger = logging.getLogger(__name__)


def match_network(batch_weights, batch_frequencies, sigma_layers,
                  sigma0_layers, gamma_layers, iters, assignments_old=None):
    """
    Main function to match fully-connected network weights

    :param batch_weights: a list of network weights. Each networks weights
    are a numpy array
    :type batch_weights: `list` of `np.array`
    :param batch_frequencies: a list of list with each outer list the
    frequencies of a network and the inner list is the frequency of each
    class for that network
    :type `list` of `list`
    :param sigma_layers: list of sigma value for each layer.
        Specify sigma as float, for a common param value across layers
    :type `list` or `float`
    :param sigma0_layers: list of sigma0 for each layer.
        Specify sigma0 as float for a common param value across layers
    :type `list` or `float`
    :param gamma_layers: list of gamma for each layer.
        Specify gamma as float for a common param value across layers
    :type `list` or `float`
    :param iters: Number of iterations of matching to be performed
    :type `int`
    :param assignments_old: Previous values of neuron assignments.
        Useful for iterative matching
    :type `list` of `list`

    :return: (global_weights, assignments), where
        global_weights is the matched global network weights
            Note: the global weights might not have the same number of
            neurons as the previous global network or any of the local
            network. The global network is expected to expand with matching
        assignments is the local to global neuron assignment matrix
    :rtype: `tuple` where all elements are `list` of `list`
    """

    n_layers = int(len(batch_weights[0]) / 2)
    j_client_cnts = len(batch_weights)
    d_datadim = batch_weights[0][0].shape[0]
    k_hiddendim = batch_weights[0][-1].shape[0]

    if assignments_old is None:
        assignments_old = (n_layers - 1) * [None]
    if not isinstance(sigma_layers, list):
        sigma_layers = (n_layers - 1) * [sigma_layers]
    if not isinstance(sigma0_layers, list):
        sigma0_layers = (n_layers - 1) * [sigma0_layers]
    if not isinstance(gamma_layers, list):
        gamma_layers = (n_layers - 1) * [gamma_layers]

    sigma_bias_layers = sigma_layers
    sigma0_bias_layers = sigma0_layers
    mu0 = 0.
    mu0_bias = 0.1
    assignment_c = [None for _ in range(j_client_cnts)]
    l_next = None
    assignment_all = []

    if batch_frequencies is None:
        last_layer_const = [np.ones(k_hiddendim) for _ in range(j_client_cnts)]
    else:
        eps = 1e-6
        total_freq_raw = sum(batch_frequencies)
        total_freq = [x if x != 0 else eps for x in total_freq_raw]
        last_layer_const = [f / total_freq for f in batch_frequencies]

    # Group descent for layer
    for c in range(1, n_layers)[::-1]:
        sigma = sigma_layers[c - 1]
        sigma_bias = sigma_bias_layers[c - 1]
        gamma = gamma_layers[c - 1]
        sigma0 = sigma0_layers[c - 1]
        sigma0_bias = sigma0_bias_layers[c - 1]
        if c == (n_layers - 1) and n_layers > 2:
            weights_bias = [
                np.hstack((
                    batch_weights[j][c * 2 - 1].reshape(-1, 1),
                    batch_weights[j][c * 2]
                ))
                for j in range(j_client_cnts)
            ]
            sigma_inv_prior = np.array(
                [1 / sigma0_bias] +
                (weights_bias[0].shape[1] - 1) * [1 / sigma0]
            )
            mean_prior = np.array(
                [mu0_bias] + (weights_bias[0].shape[1] - 1) * [mu0])
            sigma_inv_layer = [
                np.array([1 / sigma_bias] +
                         [y / sigma for y in last_layer_const[j]]
                         )
                for j in range(j_client_cnts)
            ]
        elif c > 1:
            weights_bias = [
                np.hstack(
                    (batch_weights[j][c * 2 - 1].reshape(-1, 1),
                     pfnm_core.patch_weights(batch_weights[j][c * 2], l_next, assignment_c[j])))
                for j in range(j_client_cnts)
            ]
            sigma_inv_prior = np.array(
                [1 / sigma0_bias] +
                (weights_bias[0].shape[1] - 1) * [1 / sigma0]
            )
            mean_prior = np.array(
                [mu0_bias] + (weights_bias[0].shape[1] - 1) * [mu0])
            sigma_inv_layer = [
                np.array(
                    [1 / sigma_bias] +
                    (weights_bias[j].shape[1] - 1) * [1 / sigma]
                ) for j in range(j_client_cnts)
            ]
        else:
            weights_bias = [
                np.hstack((
                    batch_weights[j][0].T,
                    batch_weights[j][c * 2 - 1].reshape(-1, 1),
                    pfnm_core.patch_weights(batch_weights[j][c * 2], l_next, assignment_c[j])
                ))
                for j in range(j_client_cnts)
            ]
            sigma_inv_prior = np.array(
                d_datadim * [1 / sigma0] +
                [1 / sigma0_bias] +
                (weights_bias[0].shape[1] - 1 - d_datadim) * [1 / sigma0]
            )
            mean_prior = np.array(
                d_datadim * [mu0] +
                [mu0_bias] +
                (weights_bias[0].shape[1] - 1 - d_datadim) * [mu0]
            )
            if n_layers == 2:
                sigma_inv_layer = [
                    np.array(
                        d_datadim * [1 / sigma] +
                        [1 / sigma_bias] +
                        [y / sigma for y in last_layer_const[j]]
                    )
                    for j in range(j_client_cnts)]
            else:
                sigma_inv_layer = [
                    np.array(
                        d_datadim * [1 / sigma] +
                        [1 / sigma_bias] +
                        (weights_bias[j].shape[1] - 1 - d_datadim) *
                        [1 / sigma])
                    for j in range(j_client_cnts)
                ]

        assignment_c, global_weights_c, global_sigmas_c = \
            match_layer(weights_bias, sigma_inv_layer, mean_prior, sigma_inv_prior,
                        gamma, iters, assignment=assignments_old[c - 1])
        l_next = global_weights_c.shape[0]
        assignment_all = [assignment_c] + assignment_all

        if c == (n_layers - 1) and n_layers > 2:
            softmax_bias, softmax_inv_sigma = \
                pfnm_core.process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0)
            global_weights_out = [global_weights_c[:, 0],
                                  global_weights_c[:, 1:], softmax_bias]
            global_inv_sigmas_out = [global_sigmas_c[:, 0],
                                     global_sigmas_c[:, 1:],
                                     softmax_inv_sigma]
        elif c > 1:
            global_weights_out = [global_weights_c[:, 0],
                                  global_weights_c[:, 1:]] + \
                                 global_weights_out
            global_inv_sigmas_out = [global_sigmas_c[:, 0],
                                     global_sigmas_c[:, 1:]] + global_inv_sigmas_out
        else:
            if n_layers == 2:
                softmax_bias, softmax_inv_sigma = \
                    pfnm_core.process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0)
                global_weights_out = [softmax_bias]
                global_inv_sigmas_out = [softmax_inv_sigma]
            global_weights_out = [global_weights_c[:, :d_datadim].T,
                                  global_weights_c[:, d_datadim],
                                  global_weights_c[:, (d_datadim + 1):]
                                  ] + global_weights_out
            global_inv_sigmas_out = [global_sigmas_c[:, :d_datadim].T,
                                     global_sigmas_c[:, d_datadim],
                                     global_sigmas_c[:, (d_datadim + 1):]
                                     ] + global_inv_sigmas_out

    map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]

    return map_out, assignment_all


def match_layer(weights_bias, sigma_inv_layer, mean_prior, sigma_inv_prior, gamma, iters, assignment=None):
    """
    Implements single-layer neural matching. See algorithm (1) in the PFNM paper
    :param weights_bias: weights and bias of the layer
    :type weights_bias: `numpy.ndarray`
    :param sigma_inv_layer: Inverse  Sigma values for the layer
    :type sigma_inv_layer: `list`
    :param mean_prior: Prior mean values
    :type mean_prior: `numpy.ndarray`
    :param sigma_inv_prior: Prior inverse sigma values
    :type sigma_inv_prior: `numpy.ndarray`
    :param gamma: Gamma values for the matching
    :type gamma: `float`
    :param iters: Number of iterations to match
    :type iters: `int`
    :param assignment: Previous assignment matrix
    :type assignment: `numpy.ndarray`
    :return: a tuple of updated assignment matrix, matched weights, and matched sigma values
    :rtype: `tuple`
    """

    J = len(weights_bias)
    group_order = sorted(range(J), key=lambda x: -weights_bias[x].shape[0])
    batch_weights_norm = [w * s for w, s in zip(weights_bias, sigma_inv_layer)]
    prior_mean_norm = mean_prior * sigma_inv_prior

    if assignment is None:
        global_weights = prior_mean_norm + batch_weights_norm[group_order[0]]
        global_sigmas = np.outer(
            np.ones(global_weights.shape[0]),
            sigma_inv_prior + sigma_inv_layer[group_order[0]]
        )

        popularity_counts = [1] * global_weights.shape[0]

        assignment = [[] for _ in range(J)]

        assignment[group_order[0]] = list(range(global_weights.shape[0]))

        # Initialize
        for j in group_order[1:]:
            global_weights, global_sigmas, popularity_counts, assignment_j = pfnm_core.matching_upd_j(
                batch_weights_norm[j], global_weights, sigma_inv_layer[j], global_sigmas,
                prior_mean_norm, sigma_inv_prior, popularity_counts, gamma, J)
            assignment[j] = assignment_j
    else:
        popularity_counts, global_weights, global_sigmas = \
            pfnm_core.init_from_assignments(batch_weights_norm, sigma_inv_layer, mean_prior,
                                            sigma_inv_prior, assignment)

    # Iterate over groups
    for _ in range(iters):
        random_order = np.random.permutation(J)
        for j in random_order:  # random_order:
            to_delete = []
            # Remove j
            Lj = len(assignment[j])
            for l, i in sorted(zip(range(Lj), assignment[j]), key=lambda x: -x[1]):
                popularity_counts[i] -= 1
                if popularity_counts[i] == 0:
                    del popularity_counts[i]
                    to_delete.append(i)
                    for j_clean in range(J):
                        for idx, l_ind in enumerate(assignment[j_clean]):
                            if i < l_ind and j_clean != j:
                                assignment[j_clean][idx] -= 1
                            elif i == l_ind and j_clean != j:
                                logger.warning('Weird unmatching detected')
                else:
                    global_weights[i] = global_weights[i] - \
                                        batch_weights_norm[j][l]
                    global_sigmas[i] -= sigma_inv_layer[j]

            global_weights = np.delete(global_weights, to_delete, axis=0)
            global_sigmas = np.delete(global_sigmas, to_delete, axis=0)

            # Match j
            global_weights, global_sigmas, popularity_counts, assignment_j = pfnm_core.matching_upd_j(
                batch_weights_norm[j], global_weights, sigma_inv_layer[j], global_sigmas,
                prior_mean_norm, sigma_inv_prior, popularity_counts, gamma, J)
            assignment[j] = assignment_j

    return assignment, global_weights, global_sigmas
