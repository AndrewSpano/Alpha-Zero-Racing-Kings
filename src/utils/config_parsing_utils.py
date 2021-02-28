"""
Created on February 16 2021

@author: Andreas Spanopoulos

Utility functions for data parsing purposes.
"""


import re

from configparser import ConfigParser
from src.utils.error_utils import InvalidConfigurationError


def str_to_bool(s):
    """
    :param str s:  String to be parsed and converted to a boolean.

    :return:  The boolean value that the input string represents.
    :rtype:   bool

    :raises:
        InvalidConfigurationError:  Error raised if a field of the configuration file is invalid.
    """
    if s == 'TRUE' or s == 'True' or s == 'true':
        return True
    elif s == 'False' or s == 'False' or s == 'false':
        return False
    else:
        raise InvalidConfigurationError(
            msg=f"Unrecognizable boolean patten '{s}'. The field should be either true, True,"
                f"TRUE, false, False or FALSE. ")


def str_to_int(s):
    """
    :param str s:  String to be parsed and converted to an integer.

    :return:  The integer value that the input string represents.
    :rtype:   int

    :raises:
        InvalidConfigurationError:  Error raised if a field of the configuration file is invalid.
    """
    if len(re.findall(r'\D', re.sub(r'\s+', '', s))) > 0:
        raise InvalidConfigurationError(msg=f"Unrecognizable scalar patten '{s}'. The fields "
                                            f"should be a positive integer.")

    numbers = re.findall(r'\d+', s)
    if len(numbers) != 1:
        raise InvalidConfigurationError(
            msg=f"Unrecognizable scalar patten '{s}'. The fields should contain exactly one "
                f"positive integer value.")

    return int(numbers[0])


def str_to_float(s):
    """
    :param str s:  String to be parsed and converted to a floating point number.

    :return:  The float value that the input string represents.
    :rtype:   float

    :raises:
        InvalidConfigurationError:  Error raised if a field of the configuration file is invalid.
    """
    if len(re.findall(r'[^\d.e-]', re.sub(r'\s+', '', s))) > 0:
        raise InvalidConfigurationError(msg=f"Unrecognizable scalar patten '{s}'. The field should "
                                            f"be a float value, e.g.: 1.0")

    numbers = re.findall(r'(\d+e-\d+|\d+\.\d+|\d+)', s)
    if len(numbers) != 1:
        print(numbers)
        raise InvalidConfigurationError(msg=f"Unrecognizable scalar patten '{s}'. The fields "
                                            f"should contain exactly one floating point value.")

    return float(numbers[0])


def str_to_int_or_tuple(s):
    """
    :param str s:  String to be parsed and converted to an integer or tuple of integers.

    :return:  The corresponding value (integer of tuple of integers) of the input string.
    :rtype:   Union[int, Tuple[int]]

    :raises:
        InvalidConfigurationError:  Error raised if a field of the configuration file is invalid.
    """
    if len(re.findall(r'[^\d(),]', re.sub(r'\s+', '', s))) > 0:
        raise InvalidConfigurationError(
            msg=f"Unrecognizable patten '{s}'. The field should either be a positive integer or a "
                f"tuple of positive integers.")

    target_string = re.findall(r'([\d]+|\([\d]+,[\d]+\))', re.sub(r'\s+', '', s))
    if len(target_string) != 1:
        raise InvalidConfigurationError(
            msg=f"Unrecognizable patten '{s}'. The field should contain exactly one value: "
                f"Union[positive int, Tuple(positive ints)]")

    matches = re.findall(r'[\d]+', target_string[0])
    if len(matches) == 1:
        # input was a scalar
        if len(re.findall(r'[(),]', s)) > 0:
            raise InvalidConfigurationError(
                msg=f"Unrecognizable scalar patten '{s}'. The field should either be a positive "
                    f"integer or a tuple of positive integers.")
        return int(matches[0])
    else:
        # input was a tuple
        return tuple(int(match) for match in matches)


def str_to_float_list(s, num):
    """
    :param str s:    String to be parsed and converted to list of floating point numbers.
    :param int num:  Number of floats to parse.

    :return:  The corresponding value (list of 3 floats) of the input string.
    :rtype:   List[float]

    :raises:
        InvalidConfigurationError:  Error raised if a field of the configuration file is invalid.
    """
    s = re.sub(r'\s+', '', s)
    if len(re.findall(r'[^\d\[\],.e-]', s)) > 0:
        raise InvalidConfigurationError(
            msg=f"Unrecognizable patten '{s}'. The field should either be a positive float or a "
                f"list of positive floats.")

    float_regex = r'(\d+e-\d+|\d+\.\d+|\d+)'
    list_regex = r'\[' + float_regex
    for _ in range(num - 1):
        list_regex += r',' + float_regex
    list_regex += r']'
    target_string = re.findall(list_regex, s)
    if len(target_string) != 1:
        raise InvalidConfigurationError(
            msg=f"Unrecognizable patten '{s}'. The field should contain exactly one value: "
                f"List[float]")

    return [float(match) for match in target_string[0]]


def configuration_to_architecture_dict(conf):
    """
    :param ConfigParser conf:  The Configuration Parser that has parsed an configuration file
                                containing information about the architecture of the Neural Network.

    :return:  A dictionary that has parsed the NN architecture information from the ConfigParser.
    :rtype:   dict
    """
    params = {'conv': {}, 'res': {}, 'policy': {}, 'value': {}}

    # parse information about the convolutional block
    params['conv']['out_channels'] = str_to_int(conf['convolutional_block']['out_channels'])
    params['conv']['kernel_size'] = str_to_int_or_tuple(conf['convolutional_block']['kernel_size'])
    params['conv']['stride'] = str_to_int_or_tuple(conf['convolutional_block']['stride'])

    # parse information about the residual block
    params['res']['num_res_blocks'] = str_to_int(conf['residual_blocks']['num_residual_blocks'])
    params['res']['out_channels'] = str_to_int(conf['residual_blocks']['out_channels'])
    params['res']['kernel_size'] = str_to_int_or_tuple(conf['residual_blocks']['kernel_size'])
    params['res']['stride'] = str_to_int_or_tuple(conf['residual_blocks']['stride'])

    # parse information about the policy head block
    params['policy']['out_channels'] = str_to_int(conf['policy_head_block']['out_channels'])
    params['policy']['kernel_size'] = str_to_int_or_tuple(conf['policy_head_block']['kernel_size'])
    params['policy']['stride'] = str_to_int_or_tuple(conf['policy_head_block']['stride'])

    # parse information about the value head block
    params['value']['out_channels'] = str_to_int(conf['value_head_block']['out_channels'])
    params['value']['kernel_size'] = str_to_int_or_tuple(conf['value_head_block']['kernel_size'])
    params['value']['stride'] = str_to_int_or_tuple(conf['value_head_block']['stride'])
    params['value']['fc_output_dim'] = str_to_int(conf['value_head_block']['fc_output_dim'])

    return params


def configuration_to_generic_architecture_dict(conf):
    """
    :param ConfigParser conf:  The Configuration Parser that has parsed an configuration file
                                containing information about the architecture of the Neural Network.

    :return:  A dictionary that has parsed the Generic NN architecture information from the
                ConfigParser.
    :rtype:   dict
    """
    params = {'conv': {}, 'res': {}, 'policy': {}, 'value': {}, 'padding': {}}

    # parse information about the convolutional block
    params['conv']['out_channels'] = str_to_int(conf['convolutional_block']['out_channels'])
    params['conv']['kernel_size'] = str_to_int_or_tuple(conf['convolutional_block']['kernel_size'])
    params['conv']['stride'] = str_to_int_or_tuple(conf['convolutional_block']['stride'])
    params['conv']['padding'] = str_to_int_or_tuple(conf['convolutional_block']['padding'])

    # parse information about the residual block
    params['res']['num_res_blocks'] = str_to_int(conf['residual_blocks']['num_residual_blocks'])
    params['res']['out_channels_1'] = str_to_int(conf['residual_blocks']['out_channels_1'])
    params['res']['kernel_size_1'] = str_to_int_or_tuple(conf['residual_blocks']['kernel_size_1'])
    params['res']['stride_1'] = str_to_int_or_tuple(conf['residual_blocks']['stride_1'])
    params['res']['padding_1'] = str_to_int_or_tuple(conf['residual_blocks']['padding_1'])
    params['res']['out_channels_2'] = str_to_int(conf['residual_blocks']['out_channels_2'])
    params['res']['kernel_size_2'] = str_to_int_or_tuple(conf['residual_blocks']['kernel_size_2'])
    params['res']['stride_2'] = str_to_int_or_tuple(conf['residual_blocks']['stride_2'])
    params['res']['padding_2'] = str_to_int_or_tuple(conf['residual_blocks']['padding_2'])

    # parse information about the policy head block
    params['policy']['out_channels'] = str_to_int(conf['policy_head_block']['out_channels'])
    params['policy']['kernel_size'] = str_to_int_or_tuple(conf['policy_head_block']['kernel_size'])
    params['policy']['stride'] = str_to_int_or_tuple(conf['policy_head_block']['stride'])
    params['policy']['padding'] = str_to_int_or_tuple(conf['policy_head_block']['padding'])

    # parse information about the value head block
    params['value']['out_channels'] = str_to_int(conf['value_head_block']['out_channels'])
    params['value']['kernel_size'] = str_to_int_or_tuple(conf['value_head_block']['kernel_size'])
    params['value']['stride'] = str_to_int_or_tuple(conf['value_head_block']['stride'])
    params['value']['fc_output_dim'] = str_to_int(conf['value_head_block']['fc_output_dim'])
    params['value']['padding'] = str_to_int_or_tuple(conf['value_head_block']['padding'])

    # parse information about padding
    params['padding']['same_padding'] = str_to_bool(conf['padding']['same'])
    params['padding']['mode'] = conf['padding']['mode']

    return params


def configuration_to_mcts_hyperparameters(conf):
    """
    :param ConfigParser conf:  The Configuration Parser that has parsed an configuration file
                                containing information about the hyperparameters of MCTS.

    :return:  A dictionary that has parsed the MCTS hyperparameters information from the
                ConfigParser.
    :rtype:   dict
    """
    return {'num_iterations': str_to_int(conf['hyperparameters']['num_iterations']),
            'c_puct': str_to_float(conf['hyperparameters']['c_puct']),
            'dirichlet_alpha': str_to_float_list(conf['hyperparameters']['dirichlet_alpha'], 3),
            'dirichlet_epsilon': str_to_float(conf['hyperparameters']['dirichlet_epsilon']),
            'temperature_tau': str_to_float(conf['hyperparameters']['temperature_tau']),
            'degrade_at_step': str_to_int(conf['hyperparameters']['degrade_at_step'])}


def configuration_to_training_parameters(conf):
    """
    :param ConfigParser conf:  The Configuration Parser that has parsed an configuration file
                                containing information about the training procedure.

    :return:  A dictionary that has parsed the training hyperparameters information from the
                ConfigParser.
    :rtype:   dict
    """
    return {'learning_rate': str_to_float(conf['hyperparameters']['learning_rate']),
            'milestones': str_to_float_list(conf['hyperparameters']['milestones'], 3),
            'gamma': str_to_float(conf['hyperparameters']['gamma']),
            'momentum': str_to_float(conf['hyperparameters']['momentum']),
            'c': str_to_float(conf['hyperparameters']['c']),
            'epochs': str_to_int(conf['hyperparameters']['epochs']),
            'batch_size': str_to_int(conf['hyperparameters']['batch_size']),
            'clip': str_to_float(conf['hyperparameters']['gradient_clipping']),
            'iterations': str_to_int(conf['self_play']['iterations']),
            'self_play_episodes': str_to_int(conf['self_play']['self_play_episodes']),
            'max_deque_len': str_to_int(conf['self_play']['max_deque_len']),
            'max_game_len': str_to_int(conf['self_play']['max_game_len']),
            'checkpoint_every': str_to_int(conf['self_play']['checkpoint_every'])}


def configuration_to_supervised_training_parameters(conf):
    """
    :param ConfigParser conf:  The Configuration Parser that has parsed an configuration file
                                containing information about the supervised training procedure.

    :return:  A dictionary that has parsed the supervised training hyperparameters information
                from the ConfigParser.
    :rtype:   dict
    """
    return {'learning_rate': str_to_float(conf['hyperparameters']['learning_rate']),
            'milestones': str_to_float_list(conf['hyperparameters']['milestones'], 3),
            'gamma': str_to_float(conf['hyperparameters']['gamma']),
            'momentum': str_to_float(conf['hyperparameters']['momentum']),
            'c': str_to_float(conf['hyperparameters']['c']),
            'batch_size': str_to_int(conf['hyperparameters']['batch_size']),
            'epochs': str_to_int(conf['hyperparameters']['epochs']),
            'min_white_elo': str_to_int(conf['imitation_constraints']['min_white_elo']),
            'min_black_elo': str_to_int(conf['imitation_constraints']['min_black_elo']),
            'worse_games': str_to_int(conf['imitation_constraints']['worse_games'])}


def parse_config_file(filepath, _type='nn_architecture'):
    """
    :param str filepath:  The (absolute/relative) path to the configuration file.
    :param str _type:     One of 'nn_architecture', 'generic_nn_architecture',
                                 'mcts_hyperparams', 'training', 'supervised_training'

    :return:  The dictionary describing the configuration file.
    :rtype:   dict

    :raises:
        ValueError:  If the _type parameter does not correspond to a known configuration type file.
    """
    config_parser = ConfigParser()
    config_parser.read(filepath)

    if _type == 'nn_architecture':
        return configuration_to_architecture_dict(config_parser)
    elif _type == 'generic_nn_architecture':
        return configuration_to_generic_architecture_dict(config_parser)
    elif _type == 'mcts_hyperparams':
        return configuration_to_mcts_hyperparameters(config_parser)
    elif _type == 'training':
        return configuration_to_training_parameters(config_parser)
    elif _type == 'supervised_training':
        return configuration_to_supervised_training_parameters(config_parser)
    else:
        raise ValueError('Unknown configuration file type.')


if __name__ == "__main__":

    import pprint
    pp = pprint.PrettyPrinter()
    file = '../../configurations/mcts_hyperparams.ini'
    pp.pprint(parse_config_file(file, _type='mcts_hyperparams'))
