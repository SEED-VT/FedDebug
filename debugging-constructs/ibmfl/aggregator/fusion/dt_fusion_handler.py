"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import numpy as np
import logging

from ibmfl.model.dt_fl_model import DTFLModel
from ibmfl.model.model_update import ModelUpdate
from ibmfl.aggregator.fusion.fusion_handler import FusionHandler
from ibmfl.exceptions import HyperparamsException

logger = logging.getLogger(__name__)


class ID3FusionHandler(FusionHandler):
    """
    Class for training decision tree type model in aggregator side
    """

    def __init__(self,
                 hyperparams,
                 proto_handler,
                 data_handler,
                 fl_model=None,
                 **kwargs):
        """
        Initializes an DecisionTreeFusionHandler object with provided
        hyperparams, data_handler and fl_model.

        :param hyperparams: Hyperparameters used for training
        :type hyperparams: `dict`
        :param proto_handler: Proto_handler that will be used to send message
        :type proto_handler: `ProtoHandler`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param fl_model: (optional) model to be trained
        :type fl_model: `model.FLModel`
        :param kwargs: Additional arguments to initialize a fusion handler.
        :type kwargs: `Dict`
        """
        if fl_model is None:
            spec = data_handler.get_dataset_info()
            fl_model = DTFLModel(None, spec)
        super().__init__(hyperparams, proto_handler, data_handler, fl_model,
                         **kwargs)
        self.name = "ID3DecisionTreeFusion"
        try:
            if hyperparams['global'] is not None and \
                    'max_depth' in hyperparams['global']:
                self.max_depth = hyperparams['global']['max_depth']
            else:
                self.max_depth = 3
                logger.info('No maximum depth of the tree was provided, '
                            'max_depth is set to the default value ' +
                            str(self.max_depth))
        except Exception as e:
            logger.exception(str(e))
            raise HyperparamsException('Global hyperparameters are badly formed. '+str(e))

    def reach_termination_criteria(self, root=None):
        """
        Return True when termination criteria has been reached, otherwise
        returns False.
        Termination criteria is reached when the tree grows to its leaves and
        there is nothing to be split.

        :return: boolean
        :rtype: 'boolean'
        """
        if root is not None and root['leaf']:
            return True

        return False

    def build_branch(self, node, current_list_of_features=None,
                     current_feature_values=None, splits=[]):
        """
        Create a decision tree branch on a given node.

        :param node: A given node to start building the tree
        :type node: `dict`
        :param current_list_of_features: (Optional) A list stores current \
        list of features that waiting to be split.
        :type current_list_of_features: `list`
        :param current_feature_values: (Optional) A list stores the \
        corresponding feature value range.
        :type current_feature_values: `list`
        :param splits: A list containing the tree split information, \
        e.g. {[feature, feature_value]}
        :type splits: `list`
        :return: None
        """
        if self.reach_termination_criteria(node):
            logger.info('Reach leaf.')
            return

        if current_list_of_features is None:
            current_list_of_features = self.fl_model.list_of_features[:]
        if current_feature_values is None:
            current_feature_values = self.fl_model.feature_values[:]

        split_value = node['split']
        split_index = current_list_of_features.index(split_value)

        current_list_of_features.remove(
            current_list_of_features[split_index])
        logger.info('Deleting feature ' + str(split_index) +
                    ' from list of features')

        remove_feature_values = current_feature_values[split_index]
        current_feature_values = \
            current_feature_values[0:split_index] + \
            current_feature_values[split_index + 1:]
        logger.info('Deleting feature value ' + str(remove_feature_values)
                    + ' from feature value list')

        for feature_value in remove_feature_values:
            curr_splits = splits[:]
            curr_splits.append([split_value, feature_value])

            self.fl_model.update_model(
                new_list_of_features=current_list_of_features[:],
                new_feature_values=current_feature_values[:])
            node[feature_value] = self.build_node(curr_splits)
            self.build_branch(node[feature_value],
                              current_list_of_features[:],
                              current_feature_values[:],
                              splits=curr_splits)

    def build_node(self, splits=[]):
        """
        Create a tree node based on parties information, splits and max_depth requirement.

        :param splits: A list containing the tree split information, e.g. {[feature_index, feature_value]}
        :type splits: `list`
        :return: A decision tree node
        :rtype: `dict`
        """
        model = self.fl_model
        if len(model.feature_values) == 0 or len(splits) >= self.max_depth:

            fit_params = {'split': splits,
                          'list_of_labels': model.list_of_labels
                          }
            lst_model_updates = self.query_all_parties(fit_params)
            model_updates = self.fusion_collected_responses(lst_model_updates)
            label_counts = model_updates.get("counts_info")
            return {'leaf': True,
                    'counts': label_counts,
                    'outcome': model.list_of_labels[
                        label_counts.index(max(label_counts))],
                    'split': None}

        fit_params = {'split': splits[:],
                      'list_of_labels': model.list_of_labels,
                      'feature_values': model.feature_values,
                      'list_of_features': model.list_of_features
                      }
        lst_model_updates = self.query_all_parties(fit_params)
        model_updates = self.fusion_collected_responses(lst_model_updates)

        scores = []
        all_label_counts = np.array(model_updates.get("counts_info"))
        all_label_counts = np.transpose(
            np.reshape(all_label_counts, [-1, len(model.list_of_labels)]))
        all_counts = np.sum(all_label_counts, axis=0)
        all_scores = all_label_counts * np.log2(
            np.divide(all_label_counts, all_counts,
                      out=np.zeros_like(all_label_counts, dtype=float),
                      where=all_counts != 0),
            out=np.zeros_like(all_label_counts, dtype=float),
            where=all_label_counts != 0)
        score_per_feature_value = np.sum(all_scores, axis=0)

        for feature_value in model.feature_values:
            score = np.sum(score_per_feature_value[0:len(feature_value)],
                           axis=0)
            score_per_feature_value = score_per_feature_value[
                len(feature_value):]
            scores.append(score)

        return {'leaf': False,
                'counts': None,
                'outcome': None,
                'split': model.list_of_features[scores.index(max(scores))]}

    def start_global_training(self, root=None):
        """
        Create a decision tree model.

        :param root: (Optional) the root of the decision tree
        :type root: `dict`
        :return: None
        """
        if root is None and len(self.fl_model.tree_model) == 0:
            root = self.build_node()
        else:
            root = self.fl_model.tree_model
        logger.info('Root of the tree is built :)')

        self.build_branch(root)
        self.fl_model.tree_model = root

    def get_global_model(self):
        """
        Returns latest tree model stored in fl_model object.

        :return: A dictionary contains the tree structure
        :rtype: `ModelUpdate`
        """
        model_update = ModelUpdate(tree_model=self.fl_model.tree_model)
        return model_update

    def fusion_collected_responses(self, lst_model_updates):
        """
        Receives a list of model updates, where a model update is of the type `ModelUpdate`, \
        using the counts in each model_update,
        it returns the sum of all counts.

        :param list of model updates: Counts of type `ModelUpdate` to be summed up.
        :type list of model updates: `list`
        :return: Model updates with sum of counts
        :rtype: `ModelUpdate`
        """
        c = []
        for update in lst_model_updates:
            c.append(update.get('counts_info'))
        counts = np.sum(np.array(c), axis=0)

        return ModelUpdate(counts_info=counts.tolist())
