"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import logging
import pickle
from ibmfl.data.data_handler import DataHandler
from ibmfl.util.datasets import load_wikipedia

logger = logging.getLogger(__name__)


class WikipediaDoc2VecDataHandler(DataHandler):
    def __init__(self, data_config=None):
        super().__init__()
        self.pfile = None

        if data_config is not None:
            if 'pickled_file' in data_config:
                self.pfile = data_config['pickled_file']

    def get_data(self, num_examples=500, starting_index=0):
        """
        Gets pre-processed 2017 Wikipedia training data. Because this method
        is for testing it takes as input the number of examples to be included
        in the training and testing set.

        :return: training data
        :rtype: list<TaggedDocument>
        """
        if self.pfile is not None:
            with open(self.pfile, "rb") as pickled_file:
                training_set = pickle.load(pickled_file)

        else:
            training_set = load_wikipedia(num_examples, starting_index)

        return training_set
