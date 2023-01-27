"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Module to where fusion algorithms are implemented.
"""
import logging
import numpy as np
from scipy.optimize import linear_sum_assignment

from ibmfl.model.model_update import ModelUpdate
from ibmfl.aggregator.fusion.fusion_handler import FusionHandler
from ibmfl.exceptions import GlobalTrainingException

logger = logging.getLogger(__name__)


class SPAHMFusionHandler(FusionHandler):
    """
    Class for SPAHM aggregation of exponential family models.
    The method is described here: https://arxiv.org/abs/1911.00218

    This method supports any model of the exponential family class
    """

    def __init__(self, hyperparams, protocol_handler,
                 fl_model=None, data_handler=None, **kwargs):
        """
        Initializes an SPAHMFusionHandler object with provided fl_model,
        data_handler and hyperparams.

        :param hyperparams: Hyperparameters used for SPAHM training. \
                The five hyperparameters used are: \
                1. sigma: `float` (default 1.0) Determines how far the local \
                model neurons are allowed from the global model. A bigger value \
                results in more matching and hence a smaller global model. \
                2. sigma0: `float` (default 1.0) Defines the standard-deviation \
                of the global network neurons. Acts as a regularizer. \
                3. gamma: `float` (default 1.0) Indian Buffet Process parameter \
                controlling the expected number of features present in each \
                observation. \
                4. iters: `int` (default 10) How many iterations of randomly \
                initialized matching-unmatching procedure is to be performed \
                before finalizing the solution \
                5. optimize_hyperparams: `bool` (default: True) Should SPAHM \
                optimize the provided hyperparameters or not?
        :type hyperparams: `dict`
        :param protocol_handler: Protocol handler used for handling learning \
        algorithm's request for communication.
        :type protocol_handler: `ProtoHandler`
        :param fl_model: model to be trained
        :type fl_model: `model.FLModel`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param kwargs: Additional arguments to initialize a fusion handler.
        :type kwargs: `dict
        """
        super().__init__(hyperparams, protocol_handler,
                         data_handler, fl_model, **kwargs)

        self.name = "SPAHM"
        self.params_global = hyperparams.get('global', {})
        self.params_local = hyperparams.get('local', None)

        if self.perc_quorum != 1.:
             raise GlobalTrainingException('Quorum specified is less than required value of 1') 

        self.rounds = self.params_global.get('rounds', 1)
        self.termination_accuracy = \
            self.params_global.get('termination_accuracy', float("inf"))
        self.sigma = self.params_global.get('sigma', 1.0)
        self.sigma0 = self.params_global.get('sigma0', 1.0)
        self.gamma = self.params_global.get('gamma', 1.0)
        self.iters = self.params_global.get('iters', 10)
        self.optimize_hyperparams = \
            self.params_global.get('optimize_hyperparams', True)

        self._local_weights = None
        self.curr_round = 0
        self.score = -1
        if fl_model is None:
            self.model_update = None
        else:
            self.model_update = fl_model.get_model_update()

    def start_global_training(self):
        """
        Starts global federated learning training process.
        """
        self.curr_round = 0

        while not self.reach_termination_criteria(self.curr_round):

            lst_parties = self.get_registered_parties()
            lst_payload = self.__prepare_payload__(lst_parties)

            lst_replies = self.query_parties(lst_payload, lst_parties)

            # Collect all model updates for fusion:
            global_weights = self.fusion_collected_responses(lst_replies)
            self.model_update = ModelUpdate(weights=global_weights)

            # Update model if we are maintaining one
            if self.fl_model is not None:
                self.fl_model.update_model(self.model_update)

            self.curr_round += 1
            self.save_current_state()

    def __prepare_payload__(self, lst_parties):
        """
        Prepares payload for each individual local model

        :return: payload for each client
        :rtype: `list`
        """

        lst_payload = []

        # Create custom payload for each local model
        for i in range(len(lst_parties)):
            party_model_update = ModelUpdate(weights=self._local_weights[i]) \
                if self._local_weights is not None else self.model_update

            payload = {
                'hyperparams': {'local': self.params_local},
                'model_update': party_model_update
            }
            lst_payload.append(payload)

        return lst_payload

    def reach_termination_criteria(self, curr_round):
        """
        Returns True when termination criteria has been reached, otherwise
        returns False.
        Termination criteria is reached when the number of rounds run reaches
        the one provided as global rounds hyperparameter.
        If a `DataHandler` has been provided and a targeted accuracy has been
        given in the list of hyperparameters, early termination is verified.

        :param curr_round: Number of global rounds that already run
        :type curr_round: `int`
        :return: boolean
        :rtype: `boolean`
        """
          
        if curr_round >= self.rounds:
            logger.info('Reached maximum global rounds. Finish training :) ')
            return True

        return self.terminate_with_metrics(curr_round)

    def get_global_model(self):
        """
        Returns last model_update

        :return: model_update
        :rtype: `ModelUpdate`
        """
        return self.model_update

    def fusion_collected_responses(self, lst_model_updates):
        """
        Receives a list of model updates, where a model update is of the type
        `ModelUpdate`, using the weighs included in each model_update, it
        finds the mean.
        The averaged weights is stored in self.model_update
        as type `ModelUpdate`.

        :param lst_model_updates: List of model updates of type `ModelUpdate` to be averaged.
        :type lst_model_updates: `list`
        :return: None
        """

        n_models = len(lst_model_updates)
        weights = [
            np.array(update.get('weights')) for update in lst_model_updates
        ]

        if n_models == 0:
            raise GlobalTrainingException('No weights available for SPAHM')

        for weight in weights:
            if None in weight or weight is None:
                return

        if n_models == 1:
            global_weights = weights[0]
            self._local_weights = weights
        else:
            global_weights, _, _, assignment = self.match_local_atoms(
                local_atoms=weights, sigma=self.sigma, sigma0=self.sigma0,
                gamma=self.gamma, it=self.iters,
                optimize_hyper=self.optimize_hyperparams)

            self._local_weights = [
                self.__build_init__(global_weights, assignment, j)
                for j in range(n_models)
            ]

        global_weights = np.array(global_weights).tolist()
        return global_weights

    @staticmethod
    def compute_cost(global_atoms, atoms_j, sigma, sigma0, mu0,
                     popularity_counts, gamma, J):
        """
        Computes the full cost to be used by Hungarian algorithm.
        Refer equation (9) in the paper
        """

        Lj = atoms_j.shape[0]
        counts = np.array(popularity_counts)
        sigma_ratio = sigma0 / sigma
        denum_match = np.outer(counts + 1, sigma0) + sigma
        param_cost = []
        for l in range(Lj):
            cost_match = ((sigma_ratio * (atoms_j[l] + global_atoms) ** 2 +
                           2 * mu0 * (atoms_j[l] + global_atoms)
                           ) / denum_match).sum(axis=1)
            param_cost.append(cost_match)

        denum_no_match = np.outer(counts, sigma0) + sigma
        cost_no_match = (
            (sigma_ratio * global_atoms ** 2 + 2 * mu0 * global_atoms) /
            denum_no_match).sum(axis=1)

        sigma_cost = (np.log(denum_no_match) - np.log(denum_match)).sum(axis=1)
        mu_cost = (
            np.outer(counts, mu0 ** 2) / denum_no_match -
            np.outer(counts + 1, mu0 ** 2) / denum_match
        ).sum(axis=1)
        counts = np.minimum(counts, 10)  # truncation of prior counts influence
        param_cost = (
            np.array(param_cost) - cost_no_match + sigma_cost +
            mu_cost + 2 * np.log(counts / (J - counts))
        )

        # Nonparametric cost
        L = global_atoms.shape[0]
        max_added = min(Lj, max(700 - L, 1))
        #    max_added = Lj
        nonparam_cost = (
            (sigma_ratio * atoms_j ** 2 + 2 * mu0 * atoms_j - mu0 ** 2) /
            (sigma0 + sigma)).sum(axis=1)
        nonparam_cost = np.outer(nonparam_cost, np.ones(max_added))
        cost_pois = 2 * np.log(np.arange(1, max_added + 1))
        nonparam_cost -= cost_pois
        nonparam_cost += 2 * np.log(gamma / J)

        # sigma penalty
        nonparam_cost += np.log(sigma).sum() - np.log(sigma0 + sigma).sum()

        full_cost = np.hstack((param_cost, nonparam_cost))
        return full_cost

    def matching_upd_j(self, atoms_j, global_atoms, global_atoms_squared,
                       sigma, sigma0, mu0, popularity_counts, gamma, J):
        """
        Computes cost [Equation 9] and solves the linear assignment problem
        using hungarian algorithm
        """

        L = global_atoms.shape[0]

        full_cost = self.compute_cost(global_atoms, atoms_j, sigma, sigma0,
                                      mu0, popularity_counts, gamma, J)

        row_ind, col_ind = linear_sum_assignment(-full_cost)
        assignment_j = []
        new_L = L

        for l, i in zip(row_ind, col_ind):
            if i < L:
                popularity_counts[i] += 1
                assignment_j.append(i)
                global_atoms[i] += atoms_j[l]
                global_atoms_squared[i] += atoms_j[l] ** 2
            else:  # new neuron
                popularity_counts += [1]
                assignment_j.append(new_L)
                new_L += 1
                global_atoms = np.vstack((global_atoms, atoms_j[l]))
                global_atoms_squared = np.vstack(
                    (global_atoms_squared, atoms_j[l] ** 2)
                )

        return (
            global_atoms, global_atoms_squared,
            popularity_counts, assignment_j
        )

    @staticmethod
    def objective(global_atoms, popularity_counts, sigma, sigma0, mu0):
        """ Computes the full objective function """

        popularity_counts = np.copy(popularity_counts)
        obj_denum = np.outer(popularity_counts, sigma0) + sigma
        obj_num = ((sigma0 / sigma) * global_atoms ** 2 +
                   2 * mu0 * global_atoms -
                   np.outer(popularity_counts, mu0 ** 2))
        obj = (obj_num / obj_denum - np.log(obj_denum)).sum()
        return obj

    @staticmethod
    def hyperparameters(global_atoms, global_atoms_squared, popularity_counts):
        """
        Estimates the hyperparameters mu0, sigma, and sigma0
        """

        popularity_counts = np.copy(popularity_counts)
        mean_atoms = global_atoms / popularity_counts.reshape(-1, 1)
        mu0 = mean_atoms.mean(axis=0)
        sigma = (
            global_atoms_squared -
            (global_atoms ** 2) / popularity_counts.reshape(-1, 1)
        )
        sigma = sigma.sum(axis=0) / (
            popularity_counts.sum() - len(popularity_counts))
        sigma = np.maximum(sigma, 1e-10)
        sigma0 = ((mean_atoms - mu0) ** 2).mean(axis=0)
        sigma0 = (sigma0 - sigma * ((1 / popularity_counts).sum()) /
                  len(popularity_counts))
        sigma0 = np.maximum(sigma0, 1e-10)
        return mu0, sigma, sigma0

    def match_local_atoms(self, local_atoms, sigma, sigma0,
                          gamma, it, optimize_hyper=True):
        """
        Estimates the global atoms given the local atoms along with the
        hyperparameters.
        """

        J = len(local_atoms)
        D = local_atoms[0].shape[1]
        group_order = sorted(range(J), key=lambda x: -local_atoms[x].shape[0])

        sigma = np.ones(D) * sigma
        sigma0 = np.ones(D) * sigma0
        total_atoms = sum([atoms_j.shape[0] for atoms_j in local_atoms])
        mu0 = sum(
            [atoms_j.sum(axis=0) for atoms_j in local_atoms]
        ) / total_atoms
        logger.info(f'SPAHM: Init mu0 estimate mean is {mu0.mean()}')

        global_atoms = np.copy(local_atoms[group_order[0]])
        global_atoms_squared = np.copy(local_atoms[group_order[0]] ** 2)

        popularity_counts = [1] * global_atoms.shape[0]

        assignment = [[] for _ in range(J)]

        assignment[group_order[0]] = list(range(global_atoms.shape[0]))

        # Initialize
        for j in group_order[1:]:
            (
                global_atoms, global_atoms_squared,
                popularity_counts, assignment_j) = self.matching_upd_j(
                    local_atoms[j], global_atoms, global_atoms_squared, sigma,
                    sigma0, mu0, popularity_counts, gamma, J)
            assignment[j] = assignment_j

        if optimize_hyper:
            mu0, sigma, sigma0 = self.hyperparameters(global_atoms,
                                                      global_atoms_squared,
                                                      popularity_counts)
            logger.info(f'SPAHM: Init Sigma mean estimate is {sigma.mean()}; '
                        f'sigma0 is {sigma0.mean()}; '
                        f'mu0 is {mu0.mean()}')

        logger.info('Init objective (without prior) is %f; '
                    'number of global atoms is %d' %
                    (self.objective(global_atoms, popularity_counts, sigma,
                                    sigma0, mu0), global_atoms.shape[0]))

        # Iterate over groups
        for iteration in range(it):
            random_order = np.random.permutation(J)
            for j in random_order:  # random_order:
                to_delete = []
                # Remove j
                Lj = len(assignment[j])
                for l, i in sorted(
                        zip(range(Lj), assignment[j]), key=lambda x: -x[1]):

                    popularity_counts[i] -= 1
                    if popularity_counts[i] == 0:
                        del popularity_counts[i]
                        to_delete.append(i)
                        for j_clean in range(J):
                            for idx, l_ind in enumerate(assignment[j_clean]):
                                if i < l_ind and j_clean != j:
                                    assignment[j_clean][idx] -= 1
                                elif i == l_ind and j_clean != j:
                                    logger.warning('SPAHM : weird unmatching')
                    else:
                        global_atoms[i] = global_atoms[i] - local_atoms[j][l]
                        global_atoms_squared[i] = global_atoms_squared[i] - \
                            local_atoms[j][l] ** 2

                global_atoms = np.delete(global_atoms, to_delete, axis=0)
                global_atoms_squared = np.delete(global_atoms_squared,
                                                 to_delete, axis=0)

                # Match j
                (
                    global_atoms, global_atoms_squared,
                    popularity_counts, assignment_j) = self.matching_upd_j(
                        local_atoms[j], global_atoms, global_atoms_squared,
                        sigma, sigma0, mu0, popularity_counts, gamma, J)

                assignment[j] = assignment_j

            if optimize_hyper:
                mu0, sigma, sigma0 = self.hyperparameters(global_atoms,
                                                          global_atoms_squared,
                                                          popularity_counts)
                logger.info(f'Sigma mean estimate is {sigma.mean()}; '
                            f'sigma0 is {sigma0.mean()}; '
                            f'mu0 is {mu0.mean()}')

            logger.info('Matching iteration %d' % iteration)
            logger.info('Objective (without prior) at iteration %d is %f; '
                        'number of global atoms is %d' %
                        (iteration,
                         self.objective(global_atoms, popularity_counts,
                                        sigma, sigma0, mu0),
                         global_atoms.shape[0]))

        logger.info(f'Number of global atoms is {global_atoms.shape[0]}, '
                    f'gamma {gamma}')

        map_out = (mu0 * sigma + global_atoms * sigma0) / (
            np.outer(popularity_counts, sigma0) + sigma)
        return map_out, popularity_counts, (
            mu0.mean(), sigma.mean(), sigma0.mean()), assignment

    @staticmethod
    def __build_init__(hungarian_weights, assignments, j):
        """
        Create local weights from the matched global weights

        :param hungarian_weights: Global network weights
        :param assignments: Assignment matrix mapping local to global neurons
        :param j: Network index for which updated local weights are required

        :return: local network weights
        :rtype: list of list
        """

        global_to_local_asgnmt = assignments[j]
        local_params = [hungarian_weights[k] for k in global_to_local_asgnmt]
        return local_params

    def get_current_metrics(self):
        """Returns metrics pertaining to current state of fusion handler

        :return: metrics
        :rtype: `dict`
        """
        fh_metrics = dict()
        fh_metrics['rounds'] = self.rounds
        fh_metrics['curr_round'] = self.curr_round
        fh_metrics['score'] = self.score 
        #fh_metrics['model_update'] = self.model_update
        return fh_metrics
