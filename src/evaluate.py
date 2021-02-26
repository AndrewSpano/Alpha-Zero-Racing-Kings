"""
Created on February 25 2021

@author: Andreas Spanopoulos

Script used to load a ChessAgent model in order to play against it. This script is very similar
to the train.py script. They differ in the command line/terminal arguments.
"""

import torch
import logging

from src.utils.main_utils import parse_evaluate_input
from src.utils.config_parsing_utils import parse_config_file

from src.environment.variants.racing_kings import RacingKingsEnv
from src.environment.variants.actions.racing_kings_actions import RacingKingsActions
from src.neural_network.network import NeuralNetwork
from src.neural_network.generic_network import GenericNeuralNetwork
from src.agent.chess_agent import RacingKingsChessAgent


def main(args):
    """ main() driver function """

    # create the environment and an API used to translate actions into their corresponding IDs
    env = RacingKingsEnv()
    mvt = RacingKingsActions()

    # parse the specific configuration files in order to start building the class objects
    model_configuration = parse_config_file(args.nn_config, _type='nn_architecture')
    mcts_configuration = parse_config_file(args.mcts_config, _type='mcts_hyperparams')

    # determine the device on which to build and train the NN
    device = torch.device(args.device)

    # add additional information to the NN configuration and initialize it
    model_configuration['input_shape'] = torch.Tensor(env.current_state_representation).shape
    model_configuration['num_actions'] = mvt.num_actions
    if args.generic:
        model = GenericNeuralNetwork(model_configuration, device).to(device)
    else:
        model = NeuralNetwork(model_configuration, device).to(device)

    # create the Chess agent
    chess_agent = RacingKingsChessAgent(env=env,
                                        mvt=mvt,
                                        nn=model,
                                        device=device,
                                        mcts_config=mcts_configuration,
                                        pretrained_w=args.pre_trained_weights)

    # play a game against the agent
    result = chess_agent.play_against(args.white)

    # print winner information
    if result == 1:
        logging.info('White won!')
    elif result == -1:
        logging.info('Black won!')
    else:
        logging.info('Draw!')


if __name__ == "__main__":
    print()
    arg = parse_evaluate_input()
    main(arg)
