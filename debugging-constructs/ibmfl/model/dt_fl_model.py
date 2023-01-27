"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import time
import logging
import json
import pickle
from sklearn.metrics import classification_report
import numpy as np

from ibmfl.util import config
from ibmfl.model.fl_model import FLModel
from ibmfl.model.model_update import ModelUpdate
from ibmfl.exceptions import LocalTrainingException, \
    ModelInitializationException, ModelException

logger = logging.getLogger(__name__)


class DTFLModel(FLModel):
    """
    Class implementation for basic decision tree
    """

    def __init__(self, model_name,
                 model_spec,
                 dt_model=None,
                 latest_counts=None,
                 **kwargs):
        """
        Create a 'DTFLModel' instance for decision tree model.
        For the given model_spec, it will extract the data information
        required to grow the tree including:
        list_of_features, feature_values, list_of_labels.
        If an initial tree structure is given, model_spec is optional.

        :param model_name: String specifying the name of the model
        :type model_name: `str`
        :param model_spec: Specification of the tree model
        :type model_spec: `dict`
        :param dt_model: A tree structure.
        :type dt_model: `dict`
        :param latest_counts: A list of last_counts saved \
        from last query call, served as party model_update information
        :type latest_counts: `list`
        :param kwargs: A dictionary contains other parameter settings on \
         to initialize a decision tree model.
        :type kwargs: `dict`
        """
        super().__init__(model_name, model_spec, **kwargs)

        if model_spec is None and dt_model is None:
            raise ModelInitializationException('Initializing model requires '
                                               'either a model specification '
                                               'or a tree structure as in a '
                                               'dictionary. '
                                               'None was provided.')
        elif model_spec is not None and (type(model_spec) is not dict):
            raise ModelInitializationException('Provided model specification'
                                               'to initialize DTFLModel'
                                               'should be of type dict. '
                                               'Instead they are:' +
                                               str(type(model_spec)))
        if model_spec is not None:
            if 'model_definition' in model_spec:
                model_def = model_spec['model_definition']
                model_def_ap = config.get_absolute_path(model_def)
                with open(model_def_ap, 'r') as f:
                    dt_spec = json.load(f)

            else:
                dt_spec = model_spec
            # In this case we create the model from model_spec
            self.list_of_features, self.feature_values, \
                self.list_of_labels = \
                self.load_data_info_from_spec(dt_spec)
        else:
            self.list_of_features = []
            self.list_of_labels = []
            self.feature_values = []

        if dt_model is None:
            dt_model = {}
        self.tree_model = dt_model

        self.model_type = 'DecisionTreeFL'

        self.latest_counts = latest_counts

    def fit_model(self, train_data, fit_params, **kwargs):
        """
        Computes the count according to split information
        and provided training data

        :param train_data: Training dataset
        :type train_data: `pandas.core.frame.DataFrame`
        :param fit_params: A dictionary containing the aggregator query \
        information, e.g., 'splits', which is a list of node split candidates \
        in the current tree; 'list_of_labels', which is a list of labels needs \
        count information
        :type fit_params: `dict`
        :return: None
        """
        splits = None
        list_of_labels = None
        current_feature_values = None
        current_list_of_features = None
        # a flag to indicate if count information will be updated
        flag_count = False

        try:
            if 'split' in fit_params:
                splits = fit_params['split']
            if splits is None:
                raise ValueError('A current split information that waits for '
                                 'counting information should be provided')

            if 'list_of_labels' in fit_params:
                list_of_labels = fit_params['list_of_labels']
            if list_of_labels is None:
                flag_count = True
                logger.info('Only counts information is computed. '
                            'No list_of_labels information is received.')

            if 'feature_values' in fit_params:
                current_feature_values = fit_params['feature_values']
            if 'list_of_features' in fit_params:
                current_list_of_features = fit_params['list_of_features']

            # splits is a empty list
            if current_feature_values is None and \
                    current_list_of_features is None and len(splits) == 0:
                raise ValueError('Provided splits should be a non-empty list.'
                                 ' Instead an empty list has been given.')

            # current list of features is missing
            if current_feature_values is not None and \
                    current_list_of_features is None:
                raise ValueError('Current list of features is missing for '
                                 'local training.')
            # current features values is missing
            if current_feature_values is None and \
                    current_list_of_features is not None:
                raise ValueError('Current feature values is missing for '
                                 'local training.')
        except Exception as ex:
            raise LocalTrainingException('Incorrect query information is '
                                         'provided by the aggregator, '
                                         'cannot provide counts information '
                                         'from the party. '+ str(ex))

        count = []
        # current_feature_values and current_list_of_features
        # will be None at the same time
        if current_feature_values is None:
            for split in splits:
                f_i = split[0]
                f_v = split[1]
                train_data = train_data.loc[
                    train_data.iloc[:, f_i].astype(str) == str(f_v)]

            if flag_count:
                count.append(train_data.shape[0])
            else:
                for label in list_of_labels:
                    if label in train_data['class'].value_counts():
                        count.append(
                            train_data['class'].value_counts()[label])
                    else:
                        # deal with the missing label
                        logger.info(str(label) + 'is missing, '
                                                 'set count to zero.')
                        count.append(0)
        else:
            data_tmp = train_data
            curr_split = splits[:]
            for split in curr_split:
                f_i = split[0]
                f_v = split[1]
                data_tmp = data_tmp.loc[data_tmp.iloc[:, f_i].astype(str) ==
                                        str(f_v)]

            for current_index in range(len(current_feature_values)):
                feature_value = current_feature_values[current_index]
                for value in feature_value:
                    data = data_tmp.loc[
                        data_tmp.iloc[:, current_list_of_features[
                            current_index]].astype(str) == str(value)]

                    if flag_count:
                        count.append(data.shape[0])
                    else:
                        tmp_list = []
                        for label in list_of_labels:
                            if label in data['class'].value_counts():
                                tmp_list.append(
                                    data['class'].value_counts()[label])
                            else:
                                # deal with the missing label
                                logger.info(str(label) + 'is missing, '
                                                         'set count to zero.')
                                tmp_list.append(0)

                        count += tmp_list

        self.latest_counts = count

    def update_model(self,
                     model_update=None,
                     new_feature_values=None,
                     new_list_of_features=None):
        """
        Update the feature list and
        their corresponding range of feature values.

        :param model_update: Optional, an ModelUpdate object that contains \
        information to update the DTFLModel.
        :type model_update: `ModelUpdate`
        :param new_feature_values: Optional, new range of feature values
        :type new_feature_values: `list`
        :param new_list_of_features: Optional, new list of feature under
        consideration
        :type new_list_of_features `list`
        :return: None
        """
        if model_update is not None:
            if not isinstance(model_update, ModelUpdate):
                raise ValueError('Provided model_update should be of type '
                                 'ModelUpdate. Instead they are:' +
                                 str(type(model_update)))
            if isinstance(model_update.get('tree_model'), dict):
                if len(model_update.get('tree_model')) == 0:
                    logger.warning('Update tree model is empty!')
                self.tree_model = model_update.get('tree_model')
            else:
                raise ValueError('Provided model_update should contain '
                                 'tree structure as type dictionary. '
                                 'Instead they are' +
                                 str(model_update.get('tree_model')))

        if new_list_of_features is not None:
            if type(new_list_of_features) is list:
                self.list_of_features = new_list_of_features
            else:
                raise ValueError('Provided new_list_of_features in DTFLModel'
                                 ' should be of type list. Instead they are:'
                                 + str(type(new_list_of_features)))

        if new_feature_values is not None:
            if type(new_feature_values) is list:
                self.feature_values = new_feature_values
            else:
                raise ValueError('Provided new_feature_values in DTFLModel '
                                 'should be of type list. Instead they are:' +
                                 str(type(new_feature_values)))

    def get_model_update(self):
        """
        Generate a ModelUpdate object that will be sent to the entities

        :return: counts; a list of counts for the latest query
        :rtype: `ModelUpdate`
        """
        return ModelUpdate(counts_info=self.latest_counts)

    def predict(self, x_test, node=None):
        """
        Perform prediction for a data instance.

        :param x_test: A data instance in the format as expected by the model
        :type x_test: `category`
        :param node: A tree root to start searching for leaf node
        :type node: `dict`
        :return: A list of predictions
        :rtype: `list`
        """
        predicts = list()

        if node is None:
            if len(self.tree_model) != 0:
                node = self.tree_model
            else:
                raise ValueError('Decision tree model is empty.')

        if node['leaf']:
            predicts.append(node['outcome'])
            return predicts

        index = node['split']
        val = x_test.iloc[0][index]

        if node[val] is None:
            predicts.append(node['outcome'])
            return predicts

        x_predict = self.predict(x_test, node[val])

        return predicts + x_predict

    def evaluate(self, test_dataset, **kwargs):
        """
        Evaluates the model given testing data.
        :param test_dataset: Testing data, a tuple given in the form \
        (x_test, test) or a datagenerator of of type `keras.utils.Sequence`, \
        `keras.preprocessing.image.ImageDataGenerator`
        :type test_dataset: `np.ndarray`

        :param kwargs: Dictionary of metrics available for the model
        :type kwargs: `dict`
        :return: dict_metrics
        :rtype `dict`
        """

        if type(test_dataset) is tuple:
            x_test = test_dataset[0]
            y_test = test_dataset[1]

            return self.evaluate_model(x_test, y_test)

        else:
            raise ModelException("Invalid test dataset!")

    def evaluate_model(self, x, y=None, class_report=False, **kwargs):
        """
        Evaluate decision tree given x and y

        :param x: Feature vectors
        :type x: `pandas.core.frame.DataFrame`
        :param y: Optional, corresponding labels of x,\
        could be given in x as its last column
        :type y: `list`
        :param class_report: A boolean value indicating \
        if classification report will be included in the evaluation results.
        :type class_report: `bool`
        :param kwargs: Dictionary of metrics available for the model
        :type kwargs: `dict`
        :return: dict_metrics
        :rtype: `dict`
        """
        correct = 0
        reply = {}
        if y is None:
            try:
                y = list(x['class'].astype(str))
            except Exception as ex:
                raise ValueError('True labels are missing for '
                                 'the testing data. ', str(ex))

        elif not isinstance(y, list):
            raise ValueError("Expecting y (labels) to be of type list, "
                             "instead they are of type " + str(type(y)))
        y_pred = []
        for i in range(x.shape[0]):
            predicts = self.predict(x.iloc[[i]])
            y_pred.append(predicts)
            if predicts[0] == y[i]:
                correct = correct + 1

        reply['acc'] = correct/float(len(y))
        reply['classificatio_report'] = \
            classification_report(np.array(y), np.array(y_pred))
        return reply

    def save_model(self, filename=None, path=None):
        """
        Save the tree to a file in the format specific to the framework
        requirements.

        :param filename: Name of the file to store the tree
        :type filename: `str`
        :param path: Path of the folder to store the tree. \
        If no path is specified, the tree will be stored in the default path
        :type path: `str`
        :return: filename
        """
        if filename is None:
            file = self.model_name if self.model_name else self.model_type
            filename = '{}_{}.pickle'.format(file, time.time())

        full_path = super().get_model_absolute_path(filename)

        with open(full_path, 'wb') as f:
            pickle.dump(self.tree_model, f)
        logger.info('Model saved in path: %s.', full_path)
        return filename

    def load_model(self, filename):
        """
        Load a tree from the disk given the filename.

        :param filename;
        :return: A decision tree
        """
        # TODO add implementation

    def print_tree(self, node=None, depth=0):
        """
        Print the tree tht stores in the DTFLModel object.
        :param node: A given tree node to start printing the tree
        :type node: `dict`
        :param depth: Tree depth
        :type depth: `int`
        :return: None
        """
        if node is None:
            node = self.tree_model

        # counts = node['counts']
        split = node['split']
        print_string = ""
        if node['leaf']:
            print_string += (':' + node['outcome'])
            return print_string
        for i in range(len(self.feature_values[split])):
            print_string += '\n'
            for j in range(depth):
                print_string += '|  '
            print_string += (str(split) + '=  ')
            print_string += str(self.feature_values[split][i])
            print_string += self.print_tree(
                node=node[self.feature_values[split][i]],
                depth=depth + 1)
        return print_string

    @staticmethod
    def load_data_info_from_spec(spec):
        """
        Loads data information from provided model_spec, where model_spec
        is a `dict` that contains three items:
        1) Data information, which is a dictionary that contains
         list_of_features, feature_values, list_of_labels, all in 'list' type;
        2)dt_model optional, a dictionary contains the tree structure;
        3)last_counts optional, a list served as a place to store

        :param spec: Model specification contains required information, \
        like list_of_features, feature_values, list_of_labels, to \
        initialize a DTFLModel.
        :type spec: `dict`
        :return: (list_of_features, feature_values, list_of_labels) where \
        list_of_features is a list of features in training data, \
        feature_values is a list of value ranges for each feature \
        in the feature list, \
        and list_of_labels is a list of possible labels for training data
        :rtype: `tuple`
        """
        list_of_features = []
        feature_values = []
        list_of_labels = []
        try:
            if 'list_of_features' in spec:
                list_of_features = spec['list_of_features']
                if type(list_of_features) is not list:
                    raise ValueError('Provided list_of_features in DTFLModel'
                                     ' should be of type list. '
                                     'Instead they are:' +
                                     str(type(list_of_features)))

            if 'feature_values' in spec:
                feature_values = spec['feature_values']
                if type(feature_values) is not list:
                    raise ValueError('Provided feature_values in DTFLModel '
                                     'should be of type list. '
                                     'Instead they are:' +
                                     str(type(feature_values)))

            if 'list_of_labels' in spec:
                list_of_labels = spec['list_of_labels']
                if type(list_of_labels) is not list:
                    raise ValueError('Provided list_of_labels in DTFLModel '
                                     'should be of type list. '
                                     'Instead they are:' +
                                     str(type(list_of_labels)))
        except Exception as ex:
            raise ModelInitializationException('Model specification was '
                                               'badly formed. '+ str(ex))

        return list_of_features[:], feature_values[:], list_of_labels[:]
