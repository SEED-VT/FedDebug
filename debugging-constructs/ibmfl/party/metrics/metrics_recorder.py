"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import json
import time


class MetricsEntry():
    """
    Stores metrics for one round. The each round has values that can be added at the respective
    hook. The hooks should only fill their values and can use any keys they like (for now).
    """

    def __init__(self):
        """
        Just initializes a dictionary for each hook and creates a variable to keep track of which
        round this entry corresponds to.
        :param: None
        """
        self.round_no = None
        self.pre_train = {}
        self.post_train = {}
        self.pre_update = {}
        self.post_update = {}

    def to_dict(self):
        """
        Converts the entry to a dictionary, simply mapping the variable names to keys.
        :return: dictionary with key, value pairs for all instance variables
        :rtype: `dict`
        """
        return {
            "round_no": self.round_no,
            "pre_train": self.pre_train,
            "post_train": self.post_train,
            "pre_update": self.pre_update,
            "post_update": self.post_update
        }


class MetricsRecorder():

    def __init__(self, output_file, output_type, compute_pre_train_eval, compute_post_train_eval):
        """
        Converts the entry to a dictionary, simply mapping the variable names to keys.
        :param output_file: path to write metrics file (without extension)
        :type output_file: `str`
        :param compute_pre_train_eval: whether to run an evaluation before local training as part of
        the train command (i.e. after the last round's sync)
        :type compute_pre_train_eval: `bool`
        :param compute_post_train_eval: whether to run an evaluation after local training as part of
        the train command (i.e. before syncing)
        :type compute_post_train_eval: `bool`
        """
        self._data = []
        self.output_file = output_file
        self.output_type = output_type
        self.compute_pre_train_eval = True
        self.compute_post_train_eval = True

    def add_entry(self):
        """
        Create a new entry, i.e. for the upcoming round.

        :param: None
        :return: None
        """
        self._data += [MetricsEntry()]

    def set_round_no(self, round_no):
        """
        Update the most recent entry with the round number.

        :param round_no: the current round number
        :type round_no: `int`
        :return: None
        """
        entry = self._data[-1]
        entry.round_no = round_no

    def pre_train_hook(self, eval_results, additional_metrics):
        """
        Fill in the pre_train dictionary of the latest entry

        :param eval_results: result from latest evaluation if compute_pre_train_eval == True \
        (else None); TODO standardize keys inside the eval_results dictionary
        :type eval_results: `dict`
        :return: None
        """
        entry = self._data[-1]
        entry.pre_train['ts'] = time.time()
        if isinstance(eval_results, dict):
            for k,v in eval_results.items():
                entry.pre_train[f'eval:{k}'] = v

        for k,v in additional_metrics:
            entry.pre_train[f'extra:{k}'] = v

    def post_train_hook(self, train_results, eval_results, additional_metrics):
        """
        Fill in the post_train dictionary of the latest entry

        :param train_results: result from the training that just took place
        TODO standardize keys inside the train_results dictionary
        :type train_results: `dict`
        :param eval_results: result from latest evaluation if compute_post_train_eval == True \
        (else None); TODO standardize keys inside the eval_results dictionary
        :type eval_results: `dict`
        :return: None
        """
        entry = self._data[-1]
        entry.post_train['ts'] = time.time()
        if isinstance(train_results, dict):
            for k,v in train_results.items():
                entry.post_train[f'train:{k}'] = v
        if isinstance(eval_results, dict):
            for k,v in eval_results.items():
                entry.post_train[f'eval:{k}'] = v

        for k,v in additional_metrics.items():
            entry.post_train[f'extra:{k}'] = v

    def pre_update_hook(self):
        """
        Fill in the pre_update dictionary of the latest entry; for now we just take timestamps

        :param: None
        :return: None
        """
        entry = self._data[-1]
        entry.pre_update['ts'] = time.time()

    def post_update_hook(self):
        """
        Fill in the post_update dictionary of the latest entry; for now we just take timestamps

        :param: None
        :return: None
        """
        entry = self._data[-1]
        entry.post_update['ts'] = time.time()

    def get_output_file(self):
        """
        Return the formatted output file, including extension

        :param: None
        :return: the full filepath to the output file
        :rtype: `str`
        """
        return '{}.{}'.format(self.output_file, self.output_type)

    def get_output_type(self):
        """
        Return the extension (and hence type) of the output, just "json" for now

        :param: None
        :return: the extension/type of how the metrics will be written to file
        :rtype: `str`
        """
        return self.output_type

    def to_json(self):
        """
        Generate json for all of the metrics data currently stored

        :param: None
        :return: json-formatted metrics data
        :rtype: `str`
        """
        entry_list = []
        for entry in self._data:
            entry_list += [entry.to_dict()]
        return json.dumps(entry_list)
