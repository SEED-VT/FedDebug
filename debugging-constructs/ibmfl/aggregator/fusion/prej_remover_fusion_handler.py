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

from ibmfl.model.model_update import ModelUpdate
from ibmfl.aggregator.fusion.iter_avg_fusion_handler import IterAvgFusionHandler

logger = logging.getLogger(__name__)

class PrejudiceRemoverFusionHandler(IterAvgFusionHandler):

    def fusion_collected_responses(self, lst_model_updates, key='weights'):
        """
        Receives a list of model updates, where a model update is of the type
        `ModelUpdate`, using the values (indicating by the key)
        included in each model_update, it finds the mean.

        :param lst_model_updates: List of model updates of type `ModelUpdate` \
        to be averaged.
        :type lst_model_updates:  `list`
        :param key: A key indicating what values the method will aggregate over.
        :type key: `str`
        :return: results after aggregation
        :rtype: `list`
        """
        v = []
        for update in lst_model_updates:
            a = update.get(key)
            #Checks if LRwPRType4() appends 'None' to updates
            if a[len(a)-1] == None:
                v.append(np.array(a[:-1]))
            else:
                v.append(np.array(a))

        results = np.mean(np.array(v), axis=0)

        return results.tolist()
