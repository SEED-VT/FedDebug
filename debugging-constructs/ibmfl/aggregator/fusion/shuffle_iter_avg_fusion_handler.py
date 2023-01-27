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
import sys
import random

from datetime import datetime
from ibmfl.model.model_update import ModelUpdate
from ibmfl.aggregator.fusion.iter_avg_fusion_handler import IterAvgFusionHandler

logger = logging.getLogger(__name__)


class ShuffleIterAvgFusionHandler(IterAvgFusionHandler):
    """
    Class for shuffle iterative averaging based fusion algorithms.
    Implements the shuffle aggregation algorithm presented in Section 4.3: 
    https://arxiv.org/pdf/2105.09400.pdf
    """

    def __init__(self, hyperparams,
                 protocol_handler,
                 data_handler=None,
                 fl_model=None,
                 **kwargs):
        """
        Initializes an IterAvgFusionHandler object with provided information,
        such as protocol handler and hyperparams.
        """
        super().__init__(hyperparams,
                         protocol_handler,
                         None,
                         None,
                         **kwargs)

        self.name = "Shuffle-Iterative-Weight-Average"
        self.trainingid = 0

        # Initialize random number generator for seed
        random.seed(datetime.now())

    def start_global_training(self):
        """
        Starts an iterative global federated learning training process.
        """
        self.curr_round = 0
        while not self.reach_termination_criteria(self.curr_round):
            # construct ModelUpdate
            if self.current_model_weights:
                model_update = ModelUpdate(weights=self.current_model_weights)
            else:
                model_update = None

            # aggseed is generated randomly for every training round
            self.trainingid = random.randrange(sys.maxsize)
            payload = {'hyperparams': {'local': self.params_local},
                       'model_update': model_update,
                       'aggseed': self.trainingid}

            logger.info('Model update' + str(model_update))

            # query all available parties
            lst_replies = self.query_all_parties(payload)

            self.update_weights(lst_replies)

            # Update model if we are maintaining one
            if self.fl_model is not None:
                self.fl_model.update_model(
                    ModelUpdate(weights=self.current_model_weights))

            self.curr_round += 1
            self.save_current_state()
