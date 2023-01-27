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
import abc

from ibmfl.exceptions import FLException, FusionException, GlobalTrainingException, \
    WarmStartException, QuorumException
from ibmfl.aggregator.metric_service import FLMetricsManager
from ibmfl.aggregator.fusion.fusion_state_service import FLFusionStateManager
import numpy as np
import json

logger = logging.getLogger(__name__)


class FusionHandler(abc.ABC):

    """
    Base class for Fusion
    """

    def __init__(self, hyperparams,
                 protocol_handler,
                 data_handler=None,
                 fl_model=None,
                 evidencia=None,
                 preprocess=None,
                 **kwargs):
        """
        Initializes an `FusionHandler` object with provided fl_model,
        data_handler and hyperparams.

        :param hyperparams: Hyperparameters used for training.
        :type hyperparams: `dict`
        :param protocol_handler: Protocol handler used for handling learning \
        algorithm's request for communication.
        :type protocol_handler: `ProtoHandler`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param fl_model: model to be trained
        :type fl_model: `model.FLModel`
        :param evidencia: evidencia to use
        :type evidencia: `evidencia.EvidenceRecorder`
        :param kwargs: Additional arguments to initialize a fusion handler.
        :type kwargs: `dict`
        """
        self.name = "Fusion-Algorithm-Name"
        self.ph = protocol_handler
        self.hyperparams = hyperparams
        self.data_handler = data_handler
        self.fl_model = fl_model
        self.fusion_state_manager = FLFusionStateManager()
        self.metrics_manager = FLMetricsManager()
        self.metrics_party = {}
        self.termination_metrics_agg = {}
        self.termination_metrics_party = {}
        self.early_stopping = None
        self.termination_reached = False
        self.perc_quorum = 1.
        self.evidencia = evidencia
        self.preprocess = preprocess

        if hyperparams and hyperparams.get('global') is not None:
            if 'perc_quorum' in hyperparams.get('global'):
                self.perc_quorum = hyperparams.get('global').get('perc_quorum')
            self.early_stopping = hyperparams.get('global').get('early_stopping')

        self.warm_start = False
        # load warm start flag if any
        if 'info' in kwargs and kwargs['info'] is not None:
            fusion_info = kwargs['info']
            self.warm_start = kwargs['info'].get('warm_start')
            if not isinstance(self.warm_start, bool):
                logger.info('Warm start flag set to False.')
                self.warm_start = False

        if self.evidencia:
            from ibmfl.evidencia.util.hashing import hash_np_array

    @abc.abstractmethod
    def start_global_training(self):
        """
        Starts global federated learning training process.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_global_model(self):
        """
        Returns the current global model at the aggregator side or
        model parameters that allow parties to reconstruct the global model.
        """
        raise NotImplementedError

    def initialization(self):
        """
        Perform initialization of the global training,
        e.g., warm-start setup etc.

        :return: None
        """
        if self.warm_start and self.fl_model:
            if self.fl_model.is_fitted():
                logger.info('Warm start enabled, '
                            'starting syncing the provided model...')
                try:
                    self.send_global_model()
                except Exception as ex:
                    logger.exception(ex)
                    raise WarmStartException(
                        'Error occurred during syncing the provided model.')
            else:
                raise WarmStartException(
                    'Provided model for warm start is not fitted.')
        elif self.warm_start and not self.fl_model:
            raise WarmStartException(
                'No initial model is provided for warm start process.')
        else:
            logger.info('Warm start disabled.')

    def get_registered_parties(self):
        """
        Returns a list of parties that registered for
        the current federated learning task.

        :return: lst_parties
        :rtype: `list`
        """
        return self.ph.get_registered_parties() 

    def get_available_parties(self):
        """
        Returns a list of parties still available in the 
        current federated learning task.

        :return: lst_parties
        :rtype: `list`
        """
        return self.ph.get_available_parties()

    def query(self, function, payload, lst_parties=None, uniform_payload=True):
        """
        Generic query wrapper function to call arbitrary function defined within
        the local training handler of the party. Returns a list of the return
        values from each of the function, irrespective of whether they provide a
        return value or not.

        :param function: Name of function call that is defined within the local \
        training handler
        :type function: `str`
        :param payload: A dictionary which corresponds to the mapping of the \
        necessary contents to pass into the argument provided in the function \
        header. If `uniform_payload` is True, then distributes the same \
        payload across all parties. If not, then each payload is distributed to \
        each worker as defined by order present in the list of dictionaries.
        :type payload: `dict` or `list` (of type `dict`)
        :param lst_parties: List of parties to receive the query. \
        Each entry of the list should be of type `PartyConnection`, and \
        the length of the `lst_parties` should match the length of `payload`. \
        If `lst_parties` is None, by default it will send queries to all \
        parties as defined by `get_registered_parties`.
        :type lst_parties: `list`
        :param uniform_payload: A boolean indicator to determine whether the \
        provided payload is the same across all parties. The default behavior is \
        defined as distributing the same parameter across all parties.
        :type uniform_payload: `boolean`
        :return: response
        :rtype: `list`
        """
        response = []

        try:
            # Check for Parties
            if lst_parties is None:
                lst_parties = self.get_registered_parties()

            # Validate and Construct Deployment Payload
            if uniform_payload:
                if not isinstance(payload, dict):
                    raise FLException('Message content is not in the correct '
                                      'format. Message content should be in the '
                                      'type of dictionary. '
                                      'Instead it is ' + str(type(payload)))

                lst_payload = [{'func': function, 'args': payload}
                               for i in range(len(lst_parties))]
            else:
                if not all(isinstance(x, dict) for x in payload):
                    raise FLException('One or more of the message content is not '
                                      'in the correct format. Message content '
                                      'should be in the type of list of dict.')

                if len(payload) != len(lst_parties):
                    raise FLException('The number of parties does not match '
                                      'lst_parties.')

                lst_payload = [{'func': function, 'args': p} for p in payload]

            response = self.query_parties(lst_payload, lst_parties)

        except QuorumException:
            raise
        except Exception as ex:
            logger.exception(str(ex))
            logger.info('Error occurred when sending queries to parties.')

        if any(isinstance(x, type(NotImplementedError)) for x in response ):
           raise FusionException("Model updates are not appropriate for this fusion method.  Check local training.")
        return response

    def query_all_parties(self, payload):
        """
        Sending queries to all registered parties.
        The query content is provided in `payload`.

        :param payload: Content of a query.
        :type payload: `dict`
        :return: lst_model_updates: a list of replies gathered from \
        the queried parties, each entry of the list should be \
        of type `ModelUpdate`.
        :rtype: `list`
        """
        lst_parties = self.get_registered_parties()
        lst_model_updates = self.query_parties(payload, lst_parties)
        return lst_model_updates

    def query_parties(self, payload, lst_parties, return_party_list=False):
        """
        Sending queries to the corresponding list of parties.
        The query contents is provided in `payload`.
        The corresponding recipients are provided in `lst_parties`.

        :param payload: Content of a query or contents of multiple queries
        :type payload: `dict` if a single query content will be sent \
        to `lst_parties` or `list` if multiple queries will be sent to \
        the corresponding parties specifying by `lst_parties`.
        :param lst_parties: List of parties to receive the query. \
        Each entry of the list should be of type `PartyConnection`, and \
        the length of the `lst_parties` should match the length of `payload` \
        if multiple queries will be sent.
        :type lst_parties: `list`
        :return: lst_model_updates: a list of replies gathered from \
        the queried parties, each entry of the list should be \
        of type `ModelUpdate`.
        :rtype: `list`
        """
        self.metrics_party = {}
        lst_response = self.ph.query_parties(payload,
                                            lst_parties,
                                            perc_quorum=self.perc_quorum,
                                            collect_metrics=True,
                                            metrics_party=self.metrics_party,
                                            fusion_state=self.fusion_state_manager,
                                            return_responding_parties=return_party_list
                                            )

        if return_party_list:
            lst_model_updates, lst_parties = lst_response
            return lst_model_updates, lst_parties
        else:
            lst_model_updates = lst_response
            return lst_model_updates

    def save_parties_models(self):
        """
        Requests all parties to save local models.
        """
        lst_parties = self.ph.get_available_parties()

        data = {}
        id_request = self.ph.save_model_parties(lst_parties, data)
        logger.info('Finished saving the models.')

    def save_local_model(self, filename=None):
        """Save aggregated model locally
        """
        saved_file = None
        if self.fl_model:
            saved_file = self.fl_model.save_model(filename=filename)

        return saved_file

    def evaluate_model(self):
        """
        Requests all parties to send model evaluations.
        """
        lst_parties = self.ph.get_available_parties()
        # TODO: parties should return evaluation results to aggregator
        data = {}
        id_request = self.ph.eval_model_parties(lst_parties, data)
        logger.info('Finished evaluate model requests.')

    def send_global_model(self):
        """
        Send global model to all the parties
        """
        # Select data parties
        lst_parties = self.ph.get_available_parties()

        model_update = self.get_global_model()
        payload = {'model_update': model_update
                   }

        logger.info('Sync Global Model' + str(model_update))
        self.ph.sync_model_parties(lst_parties, payload)

    def get_current_metrics(self):
        """Returns metrics pertaining to current state of fusion handler
        Includes all the the variables required to bring back fusion handler
        to the current state.
        """
        raise NotImplementedError

    def save_current_state(self):
        """Save current fusion handler state using metrics manager. Save current model,
        collect metrics and use metricsmanager to save them.
        """
        metrics = {}
        fusion_metrics = self.get_current_metrics()
        metrics['fusion'] = fusion_metrics
        metrics['party'] = self.metrics_party
        #model_file = self.save_local_model()
        #metrics['model_file'] = model_file
        # print(f" >>> FusionHandler.save_current_state() metrics: {metrics}")

        self.metrics_manager.save_metrics(metrics)

    def terminate_training(self):
        """This method can be used by external apis to flip the termination flag
        when required. Mostly focusing on the users who have custom aggregator and
        party code.
        """
        self.termination_reached = True

    def is_terminate_with_min_delta(self, curr_round, metrics, monitor_metric, min_delta, window):
        """
        Returns True when termination criteria with min_delta is reached. Checks if \
        monitor metric value is not improving more than min_delta for alteast n rounds \
        where n is equal to window specified in config. Default window size is 5 if \
        if its not provided in config.        
        
        :param curr_round: Number of global rounds that already run
        :type curr_round: `int`
        :param metrics: A dictionary of metrics with monitor_metric values
        :type metrics: `dict`
        :param monitor_metric: Metric key which needs to be monitored as per configuration
        :type monitor_metric: `str`
        :param min_delta: Minimum change in the monitored quantity to qualify as an improvement
        :type min_delta: `float`
        :param window: Number of rounds with no improvement
        :type window: `int` 
        :return: boolean
        :rtype: `boolean`
        """


        ## min delta is calculated for 5 consecutive rounds
        if not metrics or not min_delta: 
            return False

        if not window or window < 2:
            window = 5

        if curr_round <= window: return False

        prev_round = curr_round - window

        while prev_round < curr_round:  
            next_round = prev_round + 1
            prev_round_metrics = metrics.get(prev_round)
            next_round_metrics = metrics.get(next_round)  

            if monitor_metric not in prev_round_metrics or monitor_metric not in next_round_metrics:
                return False;

            diff = abs(next_round_metrics.get(monitor_metric) - prev_round_metrics.get(monitor_metric))
            if(diff > min_delta):
                return False
            
            prev_round +=1
        
        logger.info("Termination criteria reached at round :: " + str(curr_round))
        logger.debug("Metrics considered for this evaluation :: " + str(metrics))
        logger.info("Metric monitored for this evaluation :: "+ monitor_metric )
        logger.info("Conditions satisfied min_delta :: " + str(min_delta) + " and window of "+ str(window))
        return True

    def is_terminate_with_value(self, curr_round, metrics, monitor_metric, value, mode):
        """
        Returns True when termination criteria with value is reached. Checks if \
        monitor metric value is greater than or less than `value` defined in the config. \
        Greater or Less is identified based on the mode. If mode is `min` then this method \
        returns True when metric value is less than `value`

        :param curr_round: Number of global rounds that already run
        :type curr_round: `int`
        :param metrics: A dictionary of metrics with monitor_metric values
        :type metrics: `dict`
        :param monitor_metric: Metric key which needs to be monitored as per configuration
        :type monitor_metric: `str`
        :param value: Value of metric as configured for evaluation
        :type value: `float`
        :param mode: Identifier to check if the metric is increase or decreasing
        :type mode: `strt` 
        :return: boolean
        :rtype: `boolean`
        """
        if not metrics or not value or curr_round == 0:
            return False

        curr_round_metrics = metrics.get(curr_round)
        if monitor_metric not in curr_round_metrics:
            return False

        # Metric should be greater than Value
        if mode == 'max' and curr_round_metrics.get(monitor_metric) > value:
            logger.info("Termination criteria reached at round :: " + str(curr_round))
            logger.debug("Metrics considered for this evaluation :: " + str(metrics))
            logger.info("Metric monitored for this evaluation is"+ str(monitor_metric) + " which is > " + str(value) )

            return True
        
        elif mode == 'min' and curr_round_metrics.get(monitor_metric) < value:
            logger.info("Termination criteria reached at round :: " + str(curr_round))
            logger.debug("Metrics considered for this evaluation :: " + str(metrics))
            logger.info("Metric monitored for this evaluation is "+ monitor_metric + " which is < " + str(value) )
            return True

        return False

    def terminate_with_metrics(self, curr_round):
        """
        Returns True when termination criteria has been reached based on  \
        rules applied on the metrics produced either on aggregator or party \
        If a `DataHandler` has been provided and a targeted variable is given \
        then aggregator metrics are used for evaluating termination criteria. \
        If aggregator metrics are unavailable then party metrics are used for \
        evaluation.
        :param curr_round: Number of global rounds that already run
        :type curr_round: `int`
        :return: boolean
        :rtype: `boolean`
        """
        
        if not self.early_stopping:
            return self.termination_reached

        monitor_metric = self.early_stopping.get('monitor')
        min_delta = self.early_stopping.get('min_delta')
        value = self.early_stopping.get('value')
        mode = self.early_stopping.get('mode')
        window = self.early_stopping.get('window')

        if not monitor_metric: return False
        if not (min_delta or (value and mode)): False

        if self.data_handler and self.fl_model:
            (_, _), test_data = self.data_handler.get_data()
            eval_results = self.fl_model.evaluate(test_data)

            if self.evidencia:
                self.evidencia.add_claim("test_data_hash", "'{}'".format(hash_np_array(test_data[0])))
                self.evidencia.add_claim("test_data_labels_hash", "'{}'".format(hash_np_array(test_data[1])))
                self.evidencia.add_claim("test_data_size", str(test_data[0].shape[0]))
                self.evidencia.add_claim("test_data_labels_number", str(np.unique(test_data[1], axis=0).shape[0]))
                self.evidencia.add_claim("evaluation_results", "{}, '{}'".format(curr_round, json.dumps(eval_results)))

            self.termination_metrics_agg[curr_round] = eval_results

            self.termination_reached |= self.is_terminate_with_min_delta(curr_round, 
                                self.termination_metrics_agg, monitor_metric, min_delta, window)

            self.termination_reached |= self.is_terminate_with_value(curr_round, 
                                self.termination_metrics_agg, monitor_metric, value, mode)

        
        elif self.metrics_party:

            avg_dict = {}
            sum_dict = {}
            
            for party_id, metrics in self.metrics_party.items():
                for metric in metrics:
                    sum_dict[metric] = sum_dict.get(metric, 0) + metrics[metric]

            for key in sum_dict:
                avg_dict[key] = sum_dict[key] / float(len(self.metrics_party))

            self.termination_metrics_party[curr_round] = avg_dict
            logger.debug(self.termination_metrics_party)
            self.termination_reached |= self.is_terminate_with_min_delta(curr_round, 
                                self.termination_metrics_party, monitor_metric, min_delta, window)

            self.termination_reached |= self.is_terminate_with_value(curr_round, 
                                self.termination_metrics_party, monitor_metric, value, mode)
            
        return self.termination_reached


class FusionUtil(abc.ABC):
    """
    Base class for methods that can be used by fusion and local trainin algorithms
    """

    @staticmethod
    def flatten_model_update(lst_layerwise_wts):
        """
        Generates a flattened np array for all of the layerwise weights of an update

        :param lst_layerwise_wts: List of layer weights
        :type lst_layerwise_wts: `list`
        :return: `np.array`
        """
        wt_vector = []
        for w in lst_layerwise_wts:
            t = np.array(w).flatten()
            wt_vector = np.concatenate([wt_vector, t])
        return wt_vector
