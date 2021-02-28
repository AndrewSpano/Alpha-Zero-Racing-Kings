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


def pad_to_maxlen(lst, maxlen):
    """ pads a list with it's last element until it's length becomes maxlen """
    while len(lst) < maxlen:
        lst.append(lst[-1])


def get_legal_move_from_player(legal_moves_uci, legal_moves_san, warning_frequency=3):
    """ asks user to input a move for a specific chess position until the move he gives is legal """
    count_mistakes = 0
    move = input('Enter a move for the current position: ')
    while move not in legal_moves_uci and move not in legal_moves_san and move != 'resign':
        print(f'Move {move} is invalid for the current position.')
        count_mistakes += 1
        if count_mistakes == warning_frequency:
            print('Available moves are:')
            print(f"\tStandard Algebraic Notation: {legal_moves_san}\n"
                  f"\tUniversal Chess Interface Notation: {legal_moves_uci}\n"
                  f"\tResign: 'resign'")
            count_mistakes = 0
        move = input('Enter another move: ')
    return move


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
                        help='Path to the configuration file for the training procedure.')

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

    # whether the non-alpha_zero player wants to start as white (if specified)
    parser.add_argument('-w', '--white', action='store_true',
                        help='If specified, then Alpha Zero plays black. Else, white.')

    # whether the user does not wish to have a display of the board printed in the screen
    parser.add_argument('-v', '--no-display', action='store_true',
                        help='If specified, then no display of the chess position will appear on'
                             'the screen. Else, it will appear normally.')

    return parser.parse_args(args)


def parse_supervised_train_input(args=None):
    """ parse input from the terminal/command line for the train_supervised.py python script """

    # default files
    default_train_config = os.path.join('..', 'configurations', 'training_hyperparams.ini')
    default_supervised_train_config = os.path.join('..', 'configurations',
                                                   'supervised_training_hyperparams.ini')
    default_network_config = os.path.join('..', 'configurations', 'neural_network_architecture.ini')
    default_nn_checkpoints = os.path.join('..', 'models', 'checkpoints')
    default_root_directory = os.path.join('..', 'Dataset')
    default_destination = os.path.join('..', 'Dataset', 'parsed_data.pickle')
    default_mcts_config = os.path.join('..', 'configurations', 'mcts_hyperparams.ini')

    parser = argparse.ArgumentParser(description='Training script for Alpha Zero algorithm adapted'
                                                 'for the Racing Kings Chess variant.')

    # path to the configuration file for the training procedure
    parser.add_argument('-t', '--train-config', type=str, action='store',
                        metavar='path_to_train_configuration_file', nargs='?',
                        default=default_train_config,
                        help='Path to the configuration file for the training procedure.')

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

    # path to the configuration file for the supervised training configuration file
    parser.add_argument('-s', '--supervised-train-config', type=str, action='store',
                        metavar='path_to_the_supervised_training_configuration_file', nargs='?',
                        default=default_supervised_train_config,
                        help='Path to the configuration file for the supervised training '
                             'procedure.')

    # root directory containing all the data files for the supervised training
    parser.add_argument('-r', '--data-root-directory', type=str, action='store',
                        metavar='path_to_the_root_directory_containing_data_files', nargs='?',
                        default=default_root_directory,
                        help='Path to the root directory that contains the .pgn data files.')

    # path to store the parsed data, in order to avoid parsing it again
    parser.add_argument('-f', '--parsed-data-destination-file', type=str, action='store',
                        metavar='parsed_data_destination_file', nargs='?',
                        default=default_destination,
                        help='The path of the destination pickle file to store the parsed data.')

    # path of already stored parsed data, to avoid re-parsing it
    parser.add_argument('-p', '--parsed-data', type=str, action='store',
                        metavar='path_to_already_parsed_data',
                        help='Relative/absolute path to a pickle file containing already parsed'
                             'data from a previous run, to avoid re-parsing it.')

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
