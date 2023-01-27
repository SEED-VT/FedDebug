"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging

import numpy as np

from ibmfl.data.data_handler import DataHandler
from ibmfl.util.datasets import load_simulated_federated_clustering

logger = logging.getLogger(__name__)


class FederatedClusteringDataHandler(DataHandler):
    """
    Data handler for a simulated federated clustering dataset.
    """
    def __init__(self, data_config=None):
        super().__init__()
        self.file_name = None
        if data_config is not None:
            if 'npz_file' in data_config:
                self.file_name = data_config['npz_file']
        (self.x_train, _), (self.x_test, _) = self.load_dataset()

    def get_data(self):
        """
        Returns the simulated datasets.

        :return: (x_train, None), (x_test, None)
        :rtype: `tuple`
        """
        return (self.x_train, None), (self.x_test, None)

    def load_dataset(self, **kwargs):
        """
        Loads the simulated federated clustering dataset as described in the
        paper [https://arxiv.org/abs/1911.00218].

        :param \**kwargs
            See below
        :Keyword Arguments:
            * *L* (``int``) -- Number of true global centroids, default 100
            * *J* (``int``) -- Number of clients, default 10
            * *D* (``int``) -- Data dimension, default 50
            * *M* (``int``) -- Data points per group, default 500
            * *mu0* (``float``) -- Global mean, default 0.0
            * *global_sd* (``float``) -- Global standard deviation, default `np.sqrt(L)`
            * *local_sd* (``float``) -- Local standard deviation, default 0.1

        :return: (x_train, None), (x_test, None)
        :rtype: `tuple`
        """

        if self.file_name is None:
            # When no file name is provided, simulate a federated clustering
            # dataset for 1 client. Because the `load_simulated_federated_clustering`
            # method generates samples for J clients, we explicitly
            # set the value of J to 1.
            kwargs['J'] = 1
            data = load_simulated_federated_clustering(**kwargs)[0]

            # Duplicates simulated data to training and testing datasets
            x_train = data
            x_test = data
        else:
            try:
                logger.info('Loaded training data from ' + str(self.file_name))
                data_pickle = np.load(self.file_name, allow_pickle=True)
                x_train = data_pickle['x_train']
                x_test = data_pickle['x_test']
            except Exception:
                raise IOError('Unable to load training data from path '
                              'provided in config file: ' +
                              self.file_name)
        return (x_train, None), (x_test, None)
