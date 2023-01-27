"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops


class PerturbedGradientDescent(OptimizerV2):
    """
    Implementation of Perturbed Gradient Descent i.e. FedProx optimizer \
    FedProx Optimizer tackles the systems and statistical \
    heterogeneity in federated networks. \
    It allows for variable amounts of work to be performed locally \
    across devices, and relies on a proximal term to help stabilize the method. \
    Paper link : https://arxiv.org/pdf/1812.06127.pdf
    """

    def __init__(self, learning_rate=0.01, mu=0.01, name="PGD", **kwargs):
        """
        Initialize hyperparams of optimizer
        :param learning_rate: learning rate
        :param mu: proximal term
        :param name: name of optimizer
        :param kwargs: keyword arguments
        """
        super().__init__(name, **kwargs)

        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("prox_mu", mu)

        self._lr_t = None
        self._mu_t = None

    def _prepare(self, var_list):
        """
        Tensor versions of the constructor arguments
        :param var_list: additional variables
        """
        self._lr_t = ops.convert_to_tensor(self._get_hyper('learning_rate'), name="learning_rate")
        self._mu_t = ops.convert_to_tensor(self._get_hyper('prox_mu'), name="prox_mu")

    def _create_slots(self, var_list):
        """
        Create slots for the global solution
        :param var_list: additional variables
        """
        for v in var_list:
            self.add_slot(v, "vstar")

    def _resource_apply_dense(self, grad, var):
        """
        Update variable given gradient tensor is dense
        :param grad: tensor representing the gradient
        :param var: tensor of dtype resource which points to the variable \
         to be updated
        :return: An Operation which updates the value of the variable
        """
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")

        var_update = state_ops.assign_sub(var, lr_t * (grad + mu_t * (var - vstar)))

        return control_flow_ops.group(*[var_update, ])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        """
        Helper function to update variable given gradient tensor is sparse
        :param grad: tensor representing the gradient
        :param var: tensor of dtype resource which points to the variable \
         to be updated
        :param indices: a tensor of integral type representing the indices \
        for which the gradient is nonzero. Indices may be repeated.
        :param scatter_add: adds sparse updates to the variable
        :return: An operation which updates the value of the variable
        """
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")

        v_diff = state_ops.assign(vstar, mu_t * (var - vstar), use_locking=self._use_locking)

        with ops.control_dependencies([v_diff]):  # run v_diff operation before scatter_add
            scaled_grad = scatter_add(vstar, indices, grad)
        var_update = state_ops.assign_sub(var, lr_t * scaled_grad)

        return control_flow_ops.group(*[var_update, ])

    def _resource_apply_sparse(self, grad, var):
        """
         Update variable given gradient tensor is sparse
        :param grad: tensor representing the gradient
        :param var: tensor of dtype resource which points to the variable \
         to be updated
        :return: An operation which updates the value of the variable
        """
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(x, i, v))

    def get_config(self):
        """
        Optimizer config
        :return: serialization of the optimizer, include all hyper parameters
        """
        base_config = super(PerturbedGradientDescent, self).get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "prox_mu": self._serialize_hyperparameter("prox_mu")
        }
