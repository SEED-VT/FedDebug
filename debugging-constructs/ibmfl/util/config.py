"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Module providing utility functions helpful for both party and aggregator
"""

import os
import yaml
import logging
import logging.config
import importlib
from importlib.machinery import SourceFileLoader
from pathlib import PurePath, Path

import ibmfl.envs as fl_envs
import ibmfl.util.log_config as log_config
from ibmfl.exceptions import InvalidConfigurationException, FLException


"""
Module providing utility functions helpful for both party and aggregator
"""
# logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)
handler = logging.FileHandler('_util.config.log', mode="w")
logger.addHandler(handler)




def read_yaml_config(config_file):
    """
    reads the yaml config file and generates a dictionary

    :param config_file: yaml file containing the definitions of formatter and
    handler
    :return : dictionary object containing the config details
    """
    if config_file and os.path.exists(config_file):
        with open(config_file, 'rt') as cf:
            try:
                return yaml.safe_load(cf.read())
            except InvalidConfigurationException as ice:
                logger.error('Failed to load yaml configurations. ' + str(ice))
                logging.error('Could not load yaml configuration')
    else:
        logger.warn('Yaml configuration file not found: ' + str(config_file))

    return None


def configure_logging_from_file(config_file='log_config.yaml',
                                log_level='INFO'):
    """
    configures logging for application based on the configuration file

    :param config_file: yaml file containing the definitions of formatter and handler
    :param log_level: should be a value from [DEBUG, INFO, WARNING, ERROR, CRITICAL]
            based on the required granularity
    :return: a boolean object. False for default basic config True otherwise
    """
    log_level = logging.getLevelName(log_level)
    config = read_yaml_config(config_file)

    result = log_config.configure_logging(config, log_level)

    return result


def get_aggregator_config(**kwargs):
    """Retrieve connections settings from config file or arguments

    :param \\*\\*kwargs: Arguments that are passed into the aggregator instance
    """

    if not kwargs:
        raise InvalidConfigurationException(
            'No arguments given to Aggregator at runtime')

    config_file = kwargs.get('config_file')

    if config_file:
        return get_config_from_file(config_file=config_file)
    else:
        return get_config_from_args(**kwargs)


def get_party_config(**kwargs):
    """
    Retrieve connections settings from config file or arguments

    :param \\*\\*kwargs: Arguments that are passed into the party instance
    """

    if not kwargs:
        raise InvalidConfigurationException(
            'No arguments given to Party at runtime')

    config_file = kwargs.get('config_file')

    if config_file:
        logger.info('Getting config from file')
        return get_config_from_file(config_file=config_file)
    else:
        return get_config_from_args(**kwargs)


def get_config_from_file(config_file):
    """
    Reads a yaml file and resolves the string configuration to appropriate
    class instances.
    :param config_file: yaml file containing the configurations
    :type config_file: `str`
    :return: dictionary containing class references and additional information
    :rtype: `dict`
    """
    logger.info('Getting details from config file.')
    logger.debug('Reading Configuration File : %s', config_file)
    config_dict = read_yaml_config(config_file)

    cls_config = get_cls_by_config(config_dict)

    return cls_config


def get_config_from_args(config_dict=None, **kwargs):
    """
    Reads arguments and resolves the string configuration to appropriate
    class instances.
    :param config_file: yaml file containing the configurations
    :type config_file: `str`
    :return: dictionary of class references
    :rtype: `dict`
    """
    logger.info('Getting Aggregator details from arguments.')

    cls_config = {}
    if config_dict:
        cls_config = get_cls_by_config(config_dict)
    else:
        cls_config = get_cls_by_config(kwargs)

        # TODO: iterate through **kwargs and resolve any configurations given
        # as arguments to corresponding class
    return cls_config


def get_cls_by_config(config_dict):
    """
    Resolve the class name string references given in config file
    to its actual class reference
    :param config_dict: dictionary of configuration
    :type config_dict: `dict`
    :return: dictionary of class references
    :rtype: `dict`
    """
    cls_config = {}
    # TODO: since all the configuration blocks carry same signature this
    # should be done in iterative way for all keys in the dictionary
    cls_config['data'] = get_data_from_config(config_dict.get('data'))
    cls_config['metrics_recorder'] = get_mrec_from_config(config_dict.get('metrics_recorder'))
    cls_config['model'] = get_model_from_config(config_dict.get('model'))
    cls_config['fusion'] = get_fusion_from_config(config_dict.get('fusion'))
    cls_config['connection'] = get_connection_from_config(
        config_dict.get('connection'))
    cls_config['protocol_handler'] = get_ph_from_config(
        config_dict.get('protocol_handler'))
    cls_config['hyperparams'] = config_dict.get('hyperparams')
    cls_config['aggregator'] = config_dict.get('aggregator')
    cls_config['local_training'] = get_lt_from_config(
        config_dict.get('local_training'))
    cls_config['preprocess'] = get_preprocess_from_config(config_dict.get('preprocess'))
    cls_config['privacy'] = config_dict.get('privacy')
    cls_config['metrics'] = get_mh_from_config(config_dict.get('metrics'))
    cls_config['evidencia'] = get_evidencia_from_config(config_dict.get('evidencia'))

    return cls_config


def get_connection_from_config(config):
    """ Load connection class information from params provided in config file
    :param config: dictionary of configuration
    :type config: `dict`
    :return: dictionary of class references
    :rtype: `dict`
    """
    # TODO: validation needs to be added
    connection_config = {}
    if config:
        try:
            connection_config['cls_ref'] = get_class_by_name(
                config['path'], config['name'])
            connection_config['info'] = config['info']
            connection_config['sync'] = bool(config.get('sync', False))

        except InvalidConfigurationException as ex:
            logger.exception(ex)

            raise InvalidConfigurationException(
                'Error occurred while loading connection config.')

    else:
        logger.info('No connection config provided for this setup.')

    return connection_config


def get_data_from_config(config):
    """ Load data class information from params provided in config file
    :param config: dictionary of configuration
    :type config: `dict`
    :return: dictionary of class references
    :rtype: `dict`
    """
    # TODO: validation needs to be added
    data_config = None
    if config:
        data_config = {}
        try:
            if 'class' in config:
                data_config['cls_ref'] = config['class']
            else:
                data_config['cls_ref'] = get_class_by_name(
                    config['path'], config['name'])
            data_config['info'] = config['info']
        except InvalidConfigurationException as ex:
            logger.exception(ex)
            raise InvalidConfigurationException(
                'Error occurred while loading data config.')

    else:
        logger.info('No data config provided for this setup.')
    return data_config


def get_model_from_config(config):
    """ Load model class information from params provided in config file
    :param config: dictionary of configuration
    :type config: `dict`
    :return: dictionary of class references
    :rtype: `dict`
    """
    # TODO: validation needs to be added
    model_config = None
    if config:
        model_config = {}
        try:
            model_config['cls_ref'] = get_class_by_name(
                config['path'], config['name'])

            model_config['spec'] = config['spec']

            if 'info' in config:
                model_config['info'] = config['info']

            if 'model_file' in config:
                model_config['model_file'] = config['model_file']

        except Exception as ex:
            logger.exception(ex)
            raise InvalidConfigurationException(
                'Error occurred while loading model config.')

    else:
        logger.info('No model config provided for this setup.')

    return model_config


def get_fusion_from_config(config):
    """
    Load fusion class information from params provided in config file.

    :param config: dictionary of configuration
    :type config: `dict`
    :return: dictionary of class references
    :rtype: `dict`
    """
    # TODO: validation needs to be added
    fusion_config = {}
    if config:
        try:
            fusion_config['cls_ref'] = get_class_by_name(
                config['path'], config['name'])
            if 'info' in config:
                fusion_config['info'] = config['info']
        except Exception as ex:
            logger.exception(ex)

            raise InvalidConfigurationException(
                'Error occurred while loading fusion config.')

    else:
        logger.debug('No fusion config provided for this setup.')

    return fusion_config


def get_ph_from_config(config):
    """ Load ph class information from params provided in config file
    :param config: dictionary of configuration
    :type config: `dict`
    :return: dictionary of class references
    :rtype: `dict`
    """
    # TODO: validation needs to be added
    ph_config = {}
    if config:
        try:
            ph_config['cls_ref'] = get_class_by_name(
                config['path'], config['name'])
            if 'info' in config:
                ph_config['info'] = config['info']
        except Exception as ex:
            logger.exception(ex)

            raise InvalidConfigurationException(
                'Error occurred while loading protocol handler config.')

    else:
        logger.info('No ph config provided for this setup.')

    return ph_config


def get_evidencia_from_config(config):
    """ Evidencia recorder class information from params provided in config file
    :param config: dictionary of configuration
    :type config: `dict`
    :return: dictionary of class references
    :rtype: `dict`
    """
    # TODO: validation needs to be added
    evidencia_config = {}
    if config:
        try:
            evidencia_config['cls_ref'] = get_class_by_name(
                config['path'], config['name'])
            if 'info' in config:
                evidencia_config['info'] = config['info']

        except Exception as ex:
            logger.exception(ex)

            raise InvalidConfigurationException(
                'Error occurred while loading evidencia recorder config.')

    else:
        logger.info('No evidencia recordeer config provided for this setup.')

    return evidencia_config


def get_lt_from_config(config):
    """ Load local training class information from params provided in config file
    :param config: dictionary of configuration
    :type config: `dict`
    :return: dictionary of class references
    :rtype: `dict`
    """
    # TODO: validation needs to be added
    lt_config = {}
    if config:
        try:
            lt_config['cls_ref'] = get_class_by_name(
                config['path'], config['name'])
            if 'info' in config:
                lt_config['info'] = config['info']

        except Exception as ex:
            logger.exception(ex)

            raise InvalidConfigurationException(
                'Error occurred while loading local training config.')

    else:
        logger.debug('No local training config provided for this setup.')

    return lt_config


def get_preprocess_from_config(config):
    """ Load preprocess class information from params provided in config file
    :param config: dictionary of configuration
    :type config: `dict`
    :return: dictionary of class references
    :rtype: `dict`
    """
    preprocess_config = {}
    if config:
        try:
            preprocess_config['cls_ref'] = get_class_by_name(config['path'], config['name'])
            if 'spec' in config:
                preprocess_config['spec'] = config['spec']
            return preprocess_config
        except Exception as ex:
            logger.exception(ex)
            raise InvalidConfigurationException('Error occurred while loading local training config.')
    else:
        logger.debug('No local training config provided for this setup.')


def get_mh_from_config(config):
    """ Load metrics handler class information from params provided in config file
    :param config: dictionary of configuration
    :type config: `dict`
    :return: dictionary of class references
    :rtype: `dict`
    """
    # TODO: validation needs to be added
    mh_config = None
    if config:
        mh_config = {}
        try:
            mh_config['cls_ref'] = get_class_by_name(
                config['path'], config['name'])
            mh_config['info'] = config.get('info')
        except InvalidConfigurationException as ex:
            logger.exception(ex)
            raise InvalidConfigurationException(
                'Error occurred while loading mh config.')

    else:
        logger.info('No metrics config provided for this setup.')
    return mh_config


def get_mrec_from_config(config):
    """ Load metrics recorder class information from params provided in config file
    :param config: dictionary of configuration
    :type config: `dict`
    :return: dictionary of class references
    :rtype: `dict`
    """
    # TODO: validation needs to be added
    pm_config = None
    if config:
        pm_config = {}
        try:
            pm_config['cls_ref'] = get_class_by_name(
                config['path'], config['name'])
            pm_config['output_file'] = config['output_file']
            pm_config['output_type'] = config['output_type']
            pm_config['compute_pre_train_eval'] = config['compute_pre_train_eval']
            pm_config['compute_post_train_eval'] = config['compute_post_train_eval']
        except InvalidConfigurationException as ex:
            logger.exception(ex)
            raise InvalidConfigurationException(
                'Error occurred while loading data config.')

    else:
        logger.info('No metrics recorder config provided for this setup.')
    return pm_config


def get_class_by_name(path, name_class):
    """
    Gets object according to the name_class and the path where it is located.

    :param path: logical location of the object
    :type path: `str`
    :param name_class: name of the object
    :type name_class: `str`

    :return: Object
    """
    try:

        cls_ref = get_attr_from_path(path, name_class)
    except Exception as ex:
        logger.exception(ex)
        raise InvalidConfigurationException(
            'Error occurred while loading class '+name_class+'from path ' + path)
    return cls_ref


def get_attr_from_path(path, attr):
    """
    Gets attr from a file with given path.

    :param path: logical location of the object
    :type path: `str`
    :param attr: name of the attr
    :type attr: `str`

    :return: Object/method
    """
    file_path = Path(PurePath(path))
    if is_file_path(path):

        absolute_path = get_absolute_path(path)

        if absolute_path:
            loader = SourceFileLoader(file_path.stem, absolute_path)
            module = loader.load_module()
        else:
            raise InvalidConfigurationException(
                "File does not exist. "+path+" not found in current or working directory.")

    else:
        module = importlib.import_module(path)

    return getattr(module, attr)


def is_file_path(path):
    """
    Validate if a given path is directory file.

    :param path: path given in configuration
    :type path: `str`

    :return: Boolean
    """
    file_path = Path(PurePath(path))

    return file_path.suffix == '.py'


def get_absolute_path(path):
    """
    Get absolute path for `file path` given in config.
    If given path is relative- then absolute path is constructed by adding
    either the current directory or working directory to the path.
    If a file is found in current directory then working directory is not
    checked.

    :param path: logical location of the object
    :type path: `str`

    :return: absolute path
    """
    file_path = Path(PurePath(path))
    absolute_path = path  # when absolute path is provided

    if file_path.is_absolute == True and not file_path.exists():
        raise InvalidConfigurationException(
            "File does not exist. "+path+" not found.")

    elif get_current_dir_ap(path):
        absolute_path = get_current_dir_ap(path)

    else:
        absolute_path = get_working_dir_ap(path)

    if absolute_path is None or os.path.exists(absolute_path) is False:
        raise InvalidConfigurationException(
            "File does not exist. "+path+" not found in current or working directory.")

    return absolute_path


def get_current_dir_ap(path):
    """
    Get absolute path using current directory for `file path` given in config.
    If given path is relative, then absolute path is constructed by adding
    current directory as a prefix to path

    :param path: logical location of the object
    :type path: `str`

    :return: absolute path
    """
    file_path = Path(PurePath(path))

    current_dir = os.getcwd()
    absolute_path = Path(
        PurePath(current_dir).joinpath(PurePath(file_path)))
    if absolute_path.exists():
        return str(absolute_path)

    return None


def get_working_dir_ap(path):
    """
    Get absolute path using working_dir env variable for `file path` given in config.
    If given path is relative - then absolute path is constructed by adding
    working directory as a prefix to path

    :param path: logical location of the object
    :type path: `str`

    :return: absolute path
    """
    file_path = Path(PurePath(path))
    working_dir = fl_envs.working_directory
    absolute_path = Path(
        PurePath(working_dir).joinpath(PurePath(file_path)))
    if absolute_path.exists():
        return str(absolute_path)

    return None


def convert_zip_to_bytes(filename):
    """
    Open a zip file and return a byte array. 
    :param filename: name of the file
    :type filename: `str`
    :return: byte array
    """
    byte_arr = bytearray()

    if filename is not None:
        try:
            abs_path = get_absolute_path(filename)
            with open(abs_path, "rb") as f:
                byte_arr = f.read()

        except InvalidConfigurationException as ex:
            logger.error(str(ex))
            raise InvalidConfigurationException("Zip file does not exist.")
        except Exception as identifier:
            raise FLException(
                "Error while converting zip file to bytes.")

    return byte_arr


def convert_bytes_to_zip(byte_arr, destination_file):
    """
    Copy the bytes into a file 
    :param byte_arr: byte array which needs to be copied into destination file
    :param destination_file: name of the file
    :type destination_file: `str`
    :return: byte array
    """
    try:
        f_out = open(destination_file, 'w+b')
        f_out.write(byte_arr)
        f_out.close

    except Exception as identifier:
        raise FLException(
            "Error while creating a file at " + destination_file)
