"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Data Handler for Pendulum Environment
"""

import logging

from ibmfl.data.env_data_handler import EnvDataHandler
from ibmfl.data.env_spec import EnvHandler
from ibmfl.util.config import get_class_by_name

logger = logging.getLogger(__name__)


class PendulumEnvDataHandler(EnvDataHandler):
    '''
        Class to load  pendulum environment for reinforcement learning.
    '''
    def __init__(self, data_config):
        if data_config is not None and data_config.get('env_spec') is not None:
            env_spec = data_config.get('env_spec')
            env_definition = env_spec.get('env_definition')
            env_name = env_spec.get('env_name')
            if env_definition is None:
                raise ValueError('env specification requires environment definition')
            self.env_class_ref = get_class_by_name(env_definition, env_name)
        else:
            raise ValueError('Initializing env data handler requires environment specification')

    def get_data(self):
        pass

    def get_env_class_ref(self) -> EnvHandler:
        return self.env_class_ref
