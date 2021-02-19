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
            msg=f"Unrecognizable scalar patten '{s}'. The field same should be either true, True,"
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
        raise InvalidConfigurationError(
            msg=f"Unrecognizable scalar patten '{s}'. The fields out_channels and fc_output_dim "
                f"should be positive integers.")

    numbers = re.findall(r'\d+', s)
    if len(numbers) != 1:
        raise InvalidConfigurationError(
            msg=f"Unrecognizable scalar patten '{s}'. The fields out_channels and fc_output_dim "
                f"should contain exactly one positive integer value.")

    return int(numbers[0])


def str_to_int_or_tuple(s):
    """
    :param str s:  String to be parsed and converted to an integer or tuple of integers.

    :return:  The corresponding value (integer of tuple of integers) of the input string.
    :rtype:   Union[int, Tuple(int)]

    :raises:
        InvalidConfigurationError:  Error raised if a field of the configuration file is invalid.
    """
    if len(re.findall(r'[^\d(),\s]', s)) > 0:
        raise InvalidConfigurationError(
            msg=f"Unrecognizable patten '{s}'. The fields kernel_size and stride should either be "
                f"positive integers or tuples of positive integers.")

    target_string = re.findall(r'([\d]+|\(\s*[\d]+,\s*[\d]+\s*\))', s)
    if len(target_string) != 1:
        raise InvalidConfigurationError(
            msg=f"Unrecognizable patten '{s}'. The fields kernel_size and stride should contain "
                f"exactly one value: Union[positive int, Tuple(positive ints)]")

    matches = re.findall(r'[\d]+', target_string[0])
    if len(matches) == 1:
        # input was a scalar
        if len(re.findall(r'[(),]', s)) > 0:
            raise InvalidConfigurationError(
                msg=f"Unrecognizable scalar patten '{s}'. The fields kernel_size and stride should "
                    f"either be positive integers or tuples of positive integers.")
        return int(matches[0])
    else:
        # input was a tuple
        return tuple(int(match) for match in matches)


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


def parse_config_file(filepath, _type='nn_architecture'):
    """
    :param str filepath:  The (absolute/relative) path to the configuration file.
    ToDo: add more types when needed
    :param str _type:     One of 'nn_architecture', 'generic_nn_architecture'

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
    else:
        raise ValueError('Unknown configuration file type.')


# w = str_to_int('256')
# print(w)


# import pprint
# pp = pprint.PrettyPrinter()
# file = '../../configurations/generic_neural_network_architecture.ini'
# pp.pprint(parse_config_file(file))
