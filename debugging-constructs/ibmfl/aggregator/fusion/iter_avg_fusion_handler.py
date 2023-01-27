"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Module to where fusion algorithms are implemented.
"""

# import multiprocessing


# Create a variable to store the captured logging output
# captured_log = io.StringIO()
from io import StringIO
from threading import Thread
from ibmfl.aggregator.debugger.Debugger import runDebugger
from ibmfl.aggregator.fusion.fusion_handler import FusionHandler
from ibmfl.model.model_update import ModelUpdate
import time
import logging
import numpy as np
from queue import Queue
from diskcache import Index
import sys
logger = logging.getLogger(__name__)

formattter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('_app.log', mode="w")
file_handler.setFormatter(formattter)

logger_ibmfl = logging.getLogger("ibmfl")
logger_werkzeug = logging.getLogger("werkzeug")


def startDebugger():
    runDebugger()


class IterAvgFusionHandler(FusionHandler):
    """
    Class for iterative averaging based fusion algorithms.
    An iterative fusion algorithm here referred to a fusion algorithm that
    sends out queries at each global round to registered parties for
    information, and use the collected information from parties to update
    the global model.
    The type of queries sent out at each round is the same. For example,
    at each round, the aggregator send out a query to request local model's
    weights after parties local training ends.
    The iterative algorithms can be terminated at any global rounds.

    In this class, the aggregator requests local model's weights from all
    parties at each round, and the averaging aggregation is performed over
    collected model weights. The global model's weights then are updated by
    the mean of all collected local models' weights.
    """

    def __init__(self, hyperparams,
                 protocol_handler,
                 data_handler=None,
                 fl_model=None,
                 **kwargs):
        """
        Initializes an IterAvgFusionHandler object with provided information,
        such as protocol handler, fl_model, data_handler and hyperparams.

        :param hyperparams: Hyperparameters used for training.
        :type hyperparams: `dict`
        :param protocol_handler: Protocol handler used for handling learning \
        algorithm's request for communication.
        :type protocol_handler: `ProtoHandler`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param fl_model: model to be trained
        :type fl_model: `model.FLModel`
        :param kwargs: Additional arguments to initialize a fusion handler.
        :type kwargs: `Dict`
        :return: None
        """
        super().__init__(hyperparams,
                         protocol_handler,
                         data_handler,
                         fl_model,
                         **kwargs)

        self.sio_stream = StringIO()
        self.sio_stream_handler = logging.StreamHandler(self.sio_stream)
        self.sio_stream_handler.setFormatter(formattter)

        self.std_stream_handler = logging.StreamHandler(sys.stdout)
        self.std_stream_handler.setFormatter(formattter)

        self.debugger_thread = None
        self.name = "Iterative-Weight-Average"
        self.debugger = Index("debug_cache")
        self.faulty_parties = self.getFaultyParties()

        self.params_global = hyperparams.get('global') or {}
        self.params_local = hyperparams.get('local') or None

        self.rounds = 10 #self.params_global.get('rounds') or 1
        self.curr_round = 0
        self.global_accuracy = -1
        self.termination_accuracy = self.params_global.get(
            'termination_accuracy')

        if fl_model and fl_model.is_fitted():
            model_update = fl_model.get_model_update()
        else:
            model_update = None

        self.current_model_weights = \
            model_update.get('weights') if model_update else None

        if self.evidencia:
            from ibmfl.evidencia.util.hashing import hash_model_update
        

    
    def getFaultyParties(self):
        remove_party2info = self.debugger.get("faulty_parties", {})
        faulty_parties = remove_party2info.get("faulty_parties", [])
        return faulty_parties
    
 
    def start_global_training(self, current_round=0):
        """
        Starts an iterative global federated learning training process.
        """
        self.curr_round = current_round

        while not self.reach_termination_criteria(self.curr_round):
            self._launchDebugger()
            logger.info(
                f"\n\n ***************** Training Round {self.curr_round} ***************\n\n")
            round2info = self._singleRoundTraining()
            logger.info(f"parties matrics {self.metrics_party}")
            

            log = None
            round2info["log"] = log
            self.debugger[f"round{self.curr_round-1}"] = round2info
            self._executeDebuggerCommands()



        self._waitForDebuggerCommands()
        self._closeDebugger()

    def update_weights(self, lst_model_updates):
        """
        Update the global model's weights with the list of collected
        model_updates from parties.
        In this method, it calls the self.fusion_collected_response to average
        the local model weights collected from parties and update the current
        global model weights by the results from self.fusion_collected_response.

        :param lst_model_updates: list of model updates of type `ModelUpdate` to be averaged.
        :type lst_model_updates: `list`
        :return: None
        """
        self.current_model_weights = self.fusion_collected_responses(
            lst_model_updates)

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
            try:
                update = np.array(update.get(key), dtype=object)
            except Exception as ex:
                update = IterAvgFusionHandler.transform_update_to_np_array(
                    update.get(key))

            v.append(update)
        results = np.mean(np.array(v), axis=0)

        return results.tolist()

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
        return ModelUpdate(weights=self.current_model_weights)

    def get_current_metrics(self):
        """Returns metrics pertaining to current state of fusion handler

        :return: metrics
        :rtype: `dict`
        """
        fh_metrics = {}
        fh_metrics['rounds'] = self.rounds
        fh_metrics['curr_round'] = self.curr_round
        fh_metrics['acc'] = self.global_accuracy
        # fh_metrics['model_update'] = self.model_update
        return fh_metrics

    @staticmethod
    def transform_update_to_np_array(update):
        """
        Transform a update of type list of numpy.ndarray to a numpy.ndarray 
        of numpy.ndarray.
        This method is a way to resolve the ValueError raised by numpy when 
        all the numpy.ndarray inside the provided list have the same 
        first dimension.

        A example of the possible case:
        a = [b, c], where a is of type list, b and c is of type numpy.ndarray.
        When b.shape[0] == c.shape[0] and b.shape[1] != c.shape[1], 
        the following line of code will cause numpy to raise a ValueError: 
        Could not broadcast input array from shape XXX(b.shape) into shape XX (c.shape).

        np.array(a)

        :param update: The input list of numpy.ndarray.
        :type update: `list`
        :return: the resulting update of type numpy.ndarray
        :rtype: `np.ndarray`
        """
        if update[0].shape[0] != 2:
            update.append(np.zeros((2,)))
            update = np.array(update)
        else:
            update.append(np.zeros((3,)))
            update = np.array(update)
        return update[:-1]

    def _singleRoundTraining(self):
        self.faulty_parties = self.getFaultyParties()  #self.debugger.get("faulty_parties", [])
        # print("Faulty parties: ", self.faulty_parties)
        round2info = {}
        # construct ModelUpdate
        if self.current_model_weights:
            model_update = ModelUpdate(weights=self.current_model_weights)
        else:
            model_update = None

        if model_update:
            # log to Evidentia
            if self.evidencia:
                self.evidencia.add_claim("sent_global_model",
                                         "{}, '\"{}\"'".format(self.curr_round + 1,
                                                               hash_model_update(model_update)))

        payload = {'hyperparams': {'local': self.params_local},
                   'model_update': model_update
                   }
        round2info["queries_payload"] = payload  # debugger log
        # print(f">>Parties: {round2info['parties']}")
        logger.info('Model update' + str(model_update))

        # query all available parties
        qtime_start = time.time()

        parties_list = [p for p in self.get_registered_parties()
                        if p not in self.faulty_parties]
        round2info["parties"] = parties_list

        lst_replies = self.query_parties(payload, parties_list)
        qtime_end = time.time()  # ---------------------------------------> Query time

        # print("list of replies: ", lst_replies[0])

        logger.info(f"Toal number of parties queried: {len(lst_replies)}")
        round2info['parties_replies'] = lst_replies
        round2info["parties_metrics"] = self.metrics_party

        agg_time_start = time.time()
        # log to Evidentia
        if self.evidencia:
            updates_hashes = []
            for update in lst_replies:
                updates_hashes.append(hash_model_update(update))
                self.evidencia.add_claim("received_model_update_hashes",
                                         "{}, '{}'".format(self.curr_round + 1,
                                                           str(updates_hashes).replace('\'', '"')))

        self.update_weights(lst_replies)
        round2info["global_model_weights"] = self.current_model_weights

        # Update model if we are maintaining one
        if self.fl_model is not None:
            self.fl_model.update_model(
                ModelUpdate(weights=self.current_model_weights))
            round2info["global_model"] = self.fl_model

        self.curr_round += 1
        self.save_current_state()
        agg_time_end = time.time()  # ---------------------------------------> Aggregation time
        return round2info

    def _launchDebugger(self):
        breakpoint2info = self.debugger["breakpoint"]
        breakpoint_round_id = breakpoint2info["round"]
        # breakpoint_party_id = breakpoint2info["party"]
        # +1 because curr_round is 0-based and we want to complete the round training
        if breakpoint_round_id == self.curr_round:
            # remove the default handlers
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)

            # logger_ibmfl.addHandler(file_handler)
            # logger_werkzeug.addHandler(file_handler)
            logger_ibmfl.addHandler(self.sio_stream_handler)
            logger_werkzeug.addHandler(self.sio_stream_handler)

            print(">> Breakpoiont Round: ", breakpoint_round_id)
            print(">> Please interact with debugger terminal to continue...")
            self.debugger_thread = Thread(target=startDebugger)
            self.debugger_thread.start()


    def _checkDebuggingStatus(self):
        breakpoint_status = self.debugger.get('breakpoint_status', {})
        # print(breakpoint_status)
        # just resuming without retraining
        if "retrain_round_id" in breakpoint_status and breakpoint_status["retrain_round_id"] != -1:
            self.debugger["breakpoint_status"] = {}
            self.curr_round = breakpoint_status["retrain_round_id"]
            self._updateStreams()
            return True
        # just resuming without retraining
        elif "retrain_round_id" in breakpoint_status and breakpoint_status["retrain_round_id"] == -1:
            self.debugger["breakpoint_status"] = {}
            self._updateStreams()
        return False

    # def setGlobalModelToDebugModelFromRound(self, round):
    #     self.current_model_weights = self.debugger.cache[f"round{round}"]["global_model_weights"]

    def _updateGlobalModelWithoutFaultyParties(self, benign_replies):
        self.current_model_weights = self.fusion_collected_responses(
            benign_replies)

    def _closeDebugger(self):
        if self.debugger_thread:
            self.debugger_thread.join()  # wait for the debugger process to finish

    def _getDebuggerCommands(self):
        commands = {
            "resume": lambda: self._commandResumeTraining(),
            "agg": lambda: self._commandEvaluatePartialAggregatedModel(),
            'remove_party': lambda: self._commandRemovePartyFromRound(),
        }
        return commands
    
    def _executeDebuggerCommands(self):
        commands = self._getDebuggerCommands()
        command = self.debugger.get("command", "")
        if command in commands:
            commands[command]()
        
    def _waitForDebuggerCommands(self):
        commands = self._getDebuggerCommands()
        while True: # wait for debugger commands
            command = self.debugger.get("command", "")
            if command == "break":
                break
            elif command in commands:
                commands[command]()
            else:
                time.sleep(1)
    

    def _commandResumeTraining(self):
        stream_output = self._updateStreams()
        print(stream_output)
        self.debugger["command"] = "break" # to break the loop

       

    def _commandRemovePartyFromRound(self):
        self.faulty_parties = self.getFaultyParties()

        remove_party2info = self.debugger.get("faulty_parties", {}) 
        self.curr_round = remove_party2info.get("round")
        self.sio_stream.truncate(0)  # clear the stream buffer
        round2info = self.debugger[f"round{self.curr_round}"]
        parties = round2info["parties"]
        parties_replies = round2info["parties_replies"]
        for i in range(len(parties)):
            if parties[i] in self.faulty_parties:
                parties_replies[i] = None

        parties_replies = [x for x in parties_replies if x is not None]
        print("> Faulty party contribution is removed from global model ... ", len(
            parties_replies))
        
        print(
            f"> Retraining without including faulty parties {self.faulty_parties}")
        self._updateGlobalModelWithoutFaultyParties(parties_replies)
        
        command2result = {}
        command2result["result"] = {"model_weights": self.current_model_weights}
        command2result["status"] = "success"
        self.debugger["command_result"] = command2result
        self.debugger["command"] = "break" # to break the loop

        
        
        self.curr_round += 1  # because current round is fixed  so start from next one
        self._updateStreams()
        self.start_global_training(self.curr_round)
         
    def _commandEvaluatePartialAggregatedModel(self):
        partialagg2info = self.debugger.get("partial_agg", {})
        pids = partialagg2info.get("pids", [])
        round_id = partialagg2info.get("round", -1)

        partial_lst_of_replies = []
        parties_list = self.debugger[f"round{round_id}"]["parties"]
        for i in range(len(parties_list)):
            if parties_list[i] in pids:
                partial_lst_of_replies.append(
                    self.debugger[f"round{round_id}"]["parties_replies"][i])

        # print(f"> Evaluating partial aggregated model with parties {pids}")

        partial_agg_weights = self.fusion_collected_responses(
            partial_lst_of_replies)

        partial_model = ModelUpdate(weights=partial_agg_weights)
        payload = {'hyperparams': {'local': self.params_local},
                   'model_update': partial_model,
                   }
        lst_replies = self.query_parties(payload, parties_list)

        command2result = {}
        command2result["result"] = {"replies":lst_replies, "accuracies": self.metrics_party}
        command2result["status"] = "success"
        self.debugger["command_result"] = command2result
        self.debugger["command"] = ""  # to break the loop

    def _updateStreams(self):
        logger_ibmfl.removeHandler(self.sio_stream_handler)
        logger_werkzeug.removeHandler(self.sio_stream_handler)

        stream_output = f"stream output { self.sio_stream.getvalue()}"

        logger_ibmfl.addHandler(self.std_stream_handler)
        logger_werkzeug.addHandler(self.std_stream_handler)
        return stream_output
