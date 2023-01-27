"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Module to where FL Models are implemented.
"""
import json
import logging
import time

import ray
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.ddpg as ddpg
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.impala as impala
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

import ibmfl.util.data_handlers.rollout as rollout
from ibmfl.exceptions import LocalTrainingException, ModelException
from ibmfl.model.fl_model import FLModel
from ibmfl.model.model_update import ModelUpdate
from ibmfl.util.config import get_class_by_name

logger = logging.getLogger(__name__)


class RLlibFLModel(FLModel):
    """
    Wrapper class for importing RLlib models.
    """
    _run_config = {
        'iterations': 1,
        'checkpoint_frequency': 1,
        'logging_level': 'ERROR'
    }
    _evaluation_run_config = {
        'steps': 1000,
        'render': False
    }

    def __init__(self, model_name, model_spec, **kwargs):
        """
        Create a `RLlibFLModel` instance from a RLlib model.

        :param model_name: String specifying the type of model e.g., RLlib
        :type model_name: `str`
        :param model_spec: Specification of the RLlib model
        :type model_spec: `dict`
        :param kwargs: A dictionary contains other parameter settings on \
         to initialize a RL model.
        :type kwargs: `dict`
        """
        super().__init__(model_name, model_spec, **kwargs)

        if model_spec is None or (not isinstance(model_spec, dict)):
            raise ValueError('Initializing model requires '
                             'a model specification')
        self.model_spec = model_spec
        # load training run config
        params = model_spec.get('params') or dict()
        training = params.get('training') or dict()
        RLlibFLModel._run_config.update(training.get('run_config') or dict())
        self.train_result = None
        self.checkpoint = None
        self.test_env_config = None
        self.custom_policy_model = None
        # test environment configuration for evaluation
        evaluation = params.get('evaluation') or dict()
        self.test_env_config = evaluation.get('env_config')
        RLlibFLModel._evaluation_run_config.update(
            evaluation.get('run_config') or dict())

        self.policy_definition = None
        self.model = None
        self.custom_default_policy = None
        self.custom_policy_model = None
        self.model_type = 'RLLib'

    def get_train_result(self):
        return {k: v for k, v in iter(self.train_result.items()) if 'episode_reward' in k}

    def fit_model(self, train_data=None, fit_params=None, **kwargs):
        """
        - Training policy through environment interaction
        - Policy checkpoint
        """

        try:
            for i in range(RLlibFLModel._run_config.get('iterations')):
                # Perform one iteration of training
                train_result_dict = self.model.train()
                self.train_result = {'episode_reward_mean': train_result_dict.get('episode_reward_mean')}
                if i % RLlibFLModel._run_config.get('checkpoint_frequency') == 0:
                    self.save_model()
        except Exception as e:
            logger.exception(str(e))
            raise LocalTrainingException(
                'Error occurred while training model')



    def update_model(self, model_update):
        """
       Update RLlib model with provided model_update, where model_update
       should be generated according to `RLlibFLModel.get_model_update()`.

       :param model_update: `ModelUpdate` object that contains the weights \
       that will be used to update the model.
       :type model_update: `ModelUpdate`
       :return: None
        """


        if isinstance(model_update, ModelUpdate):
            w = model_update.get("weights")
            if w is not None:
                self.model.get_policy().set_weights(w)
        else:
            raise LocalTrainingException('Provided model_update should be of '
                                         'type ModelUpdate. '
                                         'Instead they are:' +
                                         str(type(model_update)))


    def get_model_update(self):
        """
        Generates a `ModelUpdate` object that will be sent to other entities.

        :return: ModelUpdate
        :rtype: `ModelUpdate`
        """
        w = self.model.get_policy().get_weights()
        logger.info('Episode Reward Mean: %s',
                    self.train_result.get('episode_reward_mean') or 0.0)
        return ModelUpdate(weights=w, train_result=self.train_result)

    def predict(self, x, **kwargs):
        pass

    def evaluate(self, test_dataset, **kwargs):
        """
        Evaluates the model given testing data.
        :param test_dataset: Testing data, a tuple given in the form \
        (x_test, test) or a datagenerator of of type `keras.utils.Sequence`,
        `keras.preprocessing.image.ImageDataGenerator`
        :type test_dataset: `np.ndarray`

        :param kwargs: Dictionary of metrics available for the model
        :type kwargs: `dict`
        """
        return self.evaluate_model()

    def evaluate_model(self, x=None, y=None, batch_size=128, **kwargs):
        """
        Evaluates the RLlib model with test environment
        and latest checkpoint using RLlib rollout
        Returns the mean episode reward
        """

        if self.checkpoint is None:
            raise ValueError('No checkpoint available for model evaluation')

        test_env_config_json = json.dumps({'env_config': self.test_env_config})

        parser = rollout.create_parser()
        args_list = ['--run', self.policy_definition, '--env', 'test_env',
                     self.checkpoint,
                     '--steps', str(RLlibFLModel._evaluation_run_config.get('steps')),
                     ]

        # display environment output
        if not RLlibFLModel._evaluation_run_config.get('render'):
            args_list.append('--no-render')

        if self.test_env_config is not None:
            args_list.extend(['--config', test_env_config_json])

        # set custom model of policy for evaluation
        if self.custom_default_policy is not None:
            custom_default_policy_json = json.dumps(self.custom_default_policy)
            args_list.extend(['--custom_default_policy',
                              custom_default_policy_json])

        args = parser.parse_args(args=args_list)
        result = rollout.run(args, parser)

        eval_results = {'episode_reward_mean': result}
        return eval_results

    def save_model(self, filename=None, path=None):
        """
        Save a model to file in the ray results folder.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is \
        specified, the model will be stored in the default data location of \
        the library `DATA_PATH`.
        :type path: `str`
        :return: None
        """
        if filename is not None:
            path = super().get_model_absolute_path(filename)
        self.checkpoint = self.model.save(path)
        logger.info("Checkpoint saved in path: %s", str(self.checkpoint))

    def load_model(self, filename):
        """
        Loads a model from disk given the specified file_name

        :param file_name: Name of the file that contains the model to be loaded.
        :type filename: `str`
        """
        self.model.restore(filename)

    def create_rl_trainer(self, env_class_ref, train_data=None, test_data=None):
        """
        creates RL trainer using model_spec where model_spec['model_definition']
        specifies RL algorithm and env_class_ref contains environment definition.
        """

        # Register the training and test environment definition class
        register_env("training_env", lambda env_config: env_class_ref(
            data=train_data, env_config=env_config))
        register_env("test_env", lambda env_config: env_class_ref(
            data=test_data, env_config=env_config))

        # sets RLlib trainer reference
        if self.model_spec.get('policy_definition') is not None:
            policy_definition = self.model_spec['policy_definition'].strip(
            ).upper()
            if policy_definition == "PPO":
                rl_trainer = ppo.PPOTrainer
                config = ppo.DEFAULT_CONFIG.copy()
            elif policy_definition == "A3C":
                rl_trainer = a3c.A3CTrainer
                config = a3c.DEFAULT_CONFIG.copy()
            elif policy_definition == "DDPG":
                rl_trainer = ddpg.DDPGTrainer
                config = ddpg.DEFAULT_CONFIG.copy()
            elif policy_definition == "DQN":
                rl_trainer = dqn.DQNTrainer
                config = dqn.DEFAULT_CONFIG.copy()
            elif policy_definition == "SIMPLEQ":
                policy_definition = "SimpleQ"
                rl_trainer = dqn.SimpleQTrainer
                config = dqn.SIMPLE_Q_DEFAULT_CONFIG.copy()
            elif policy_definition == "IMPALA":
                rl_trainer = impala.ImpalaTrainer
                config = impala.DEFAULT_CONFIG.copy()
            else:
                raise ValueError('Specified RLlib algorithm : '
                                 + policy_definition + ' not supported')
        else:
            raise ValueError('Initializing model requires model definition')

        # Start Ray cluster
        if not ray.is_initialized():
            ray.init(logging_level=logging._nameToLevel.get(
                RLlibFLModel._run_config.get('logging_level')))

        # load custom policy using model config or extending default policy
        policy_custom = self.model_spec.get('custom_policy') or dict()

        # set custom model of policy using model config
        self.custom_policy_model = policy_custom.get('custom_policy_model')
        if self.custom_policy_model is not None:
            custom_policy_model_definition = self.custom_policy_model.get(
                'path')
            custom_policy_model_name = self.custom_policy_model.get('name')
            custom_policy_model_ref = get_class_by_name(custom_policy_model_definition,
                                                        custom_policy_model_name)
            ModelCatalog.register_custom_model(
                custom_policy_model_name, custom_policy_model_ref)
            config["model"] = {"custom_model": custom_policy_model_name}

        # set custom model of policy by extending default_policy implementation
        self.custom_default_policy = policy_custom.get('custom_default_policy')
        if self.custom_default_policy is not None:
            custom_default_policy_definition = self.custom_default_policy.get(
                'path')
            custom_default_policy_name = self.custom_default_policy.get('name')
            custom_default_policy_ref = get_class_by_name(custom_default_policy_definition,
                                                          custom_default_policy_name)
            rl_trainer = rl_trainer.with_updates(
                default_policy=custom_default_policy_ref)

        # Initializes RLlib trainer with model and environment configuration
        params = self.model_spec.get('params') or dict()
        training = params.get('training') or dict()
        model_config = training.get('model_config') or dict()
        config.update(model_config)
        env_config = training.get('env_config')
        if env_config is not None:
            config["env_config"] = env_config
        self.policy_definition = policy_definition
        self.model = rl_trainer(config=config, env="training_env")

        # restore model from checkpoint path if available
        checkpoint_path = RLlibFLModel._run_config.get('checkpoint_path')
        if checkpoint_path is not None:
            self.checkpoint = checkpoint_path
            self.load_model(checkpoint_path)
