"""
Created on February 24 2021

@author: Andreas Spanopoulos

Utility function for the src/train.py driver script.
"""

import os
import torch
import argparse
import itertools


def save_pytorch_model(model, path):
    """ wrapper function used to save a pytorch model, given it's save path """
    torch.save(model.state_dict(), path)


def load_pytorch_model(model, path, device):
    """ wrapper function used to load a pytorch model, given it's save path """
    model.load_state_dict(torch.load(path, map_location=device))


def deque_slice(deque, start, end=-1):
    """ return a sliced part of a deque as a list """
    if end == -1:
        end = len(deque)
    return list(itertools.islice(deque, start, end))


def parse_train_input(args=None):
    """ parse input from the terminal/command line for the train.py python script """

    # default files
    default_train_config = os.path.join('..', 'configurations', 'training_hyperparams.ini')
    default_network_config = os.path.join('..', 'configurations', 'neural_network_architecture.ini')
    default_nn_checkpoints = os.path.join('..', 'models', 'checkpoints')
    default_mcts_config = os.path.join('..', 'configurations', 'mcts_hyperparams.ini')

    parser = argparse.ArgumentParser(description='Training script for Alpha Zero algorithm adapted'
                                                 'for the Racing Kings Chess variant.')

    # path to the configuration file for the training procedure
    parser.add_argument('-t', '--train-config', type=str, action='store',
                        metavar='path_to_train_configuration_file', nargs='?',
                        default=default_train_config,
                        help='Path to the configuration file for the training procedure')

    # path to the configuration file for the Neural Network
    parser.add_argument('-n', '--nn-config', type=str, action='store',
                        metavar='path_to_neural_network_configuration_file', nargs='?',
                        default=default_network_config,
                        help='Path to the configuration file for the Neural Network.')

    # whether the Generic Neural Network should be used, or the simpler one
    parser.add_argument('-g', '--generic', action='store_true',
                        help='If specified, then the GenericNeuralNetwork class will be used '
                             'instead of the NeuralNetwork class. Note that in this case the '
                             'configuration file given should be the one for the generic network.')

    # the path of the directory in which the NN checkpoints will be stored
    parser.add_argument('-c', '--nn-checkpoints', type=str, action='store',
                        metavar='path_to_the_directory_checkpoints', nargs='?',
                        default=default_nn_checkpoints,
                        help='Relative/Absolute path to the directory used for saving NN weights.')

    # the path of the stored NN weights
    parser.add_argument('-p', '--pre-trained-weights', type=str, action='store',
                        metavar='path_to_the_pretrained_nn_weights', required=False,
                        help='Relative/Absolute path to the pre-trained and saved NN weights.')

    # path to the configuration file for the Monte Carlo Tree Search
    parser.add_argument('-m', '--mcts-config', type=str, action='store',
                        metavar='path_to_monte_carlo_tree_search_configuration_file', nargs='?',
                        default=default_mcts_config,
                        help='Path to the configuration file for the Monte Carlo Tree Search.')

    # which device to use for the model: either 'cpu' or 'cuda'
    parser.add_argument('-d', '--device', type=str, action='store', metavar='device',
                        choices=['cpu', 'cuda'], nargs='?', default='cpu',
                        help="Which device to use when running the model: 'cpu' or 'cuda'.")

    return parser.parse_args(args)


def parse_evaluate_input(args=None):
    """ parse input from the terminal/command line for the evaluate.py python script """

    # default files
    default_network_config = os.path.join('..', 'configurations', 'neural_network_architecture.ini')
    default_mcts_config = os.path.join('..', 'configurations', 'mcts_hyperparams.ini')

    parser = argparse.ArgumentParser(description='Validation script for Alpha Zero algorithm '
                                                 'adapted for the Racing Kings Chess variant.')

    # path to the configuration file for the Neural Network
    parser.add_argument('-n', '--nn-config', type=str, action='store',
                        metavar='path_to_neural_network_configuration_file', nargs='?',
                        default=default_network_config,
                        help='Path to the configuration file for the Neural Network.')

    # whether the Generic Neural Network should be used, or the simpler one
    parser.add_argument('-g', '--generic', action='store_true',
                        help='If specified, then the GenericNeuralNetwork class will be used '
                             'instead of the NeuralNetwork class. Note that in this case the '
                             'configuration file given should be the one for the generic network.')

    # the path of the stored NN weights
    parser.add_argument('-p', '--pre-trained-weights', type=str, action='store',
                        metavar='path_to_the_pretrained_nn_weights', required=True,
                        help='Relative/Absolute path to the pre-trained and saved NN weights.')

    # path to the configuration file for the Monte Carlo Tree Search
    parser.add_argument('-m', '--mcts-config', type=str, action='store',
                        metavar='path_to_monte_carlo_tree_search_configuration_file', nargs='?',
                        default=default_mcts_config,
                        help='Path to the configuration file for the Monte Carlo Tree Search.')

    # which device to use for the model: either 'cpu' or 'cuda'
    parser.add_argument('-d', '--device', type=str, action='store', metavar='device',
                        choices=['cpu', 'cuda'], nargs='?', default='cpu',
                        help="Which device to use when running the model: 'cpu' or 'cuda'.")

    # whether the non-alpha_zero player wants to start was white (if specified)
    parser.add_argument('-w', '--white', action='store_true',
                        help='If specified, then Alpha Zero plays black. Else, white.')

    return parser.parse_args(args)
