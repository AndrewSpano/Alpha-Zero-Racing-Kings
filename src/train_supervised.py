"""
Created on February 28 2021

@author: Andreas Spanopoulos

Script used to train the Neural Network of the AlphaZero agent using Human Expert play data
(behavioural cloning). After, the AlphaZero agent can be trained normally with self-play,
while having stored the previous games.

Example usage:

python3 train_supervised.py
    --train-config ../configurations/training_hyperparams.ini
    --nn-config ../configurations/neural_network_architecture.ini
    --nn-checkpoints ../models/checkpoints
    --supervised-train-config ../configurations/supervised_training_hyperparams.ini
    --data-root-directory ../Dataset
    --parsed-data-destination-file ../Dataset/parsed_data.pickle
    --mcts-config ../configurations/mcts_hyperparams.ini
    --device cpu
"""

import torch
import logging

from src.utils.main_utils import parse_supervised_train_input
from src.utils.config_parsing_utils import parse_config_file

from src.environment.variants.racing_kings import RacingKingsEnv
from src.environment.actions.racing_kings_actions import RacingKingsActions
from src.neural_network.network import NeuralNetwork
from src.neural_network.generic_network import GenericNeuralNetwork
from src.agent.chess_agent import AlphaZeroChessAgent


def main(args):
    """ main() driver function """

    # set logging format
    fmt = "(%(filename)s:%(lineno)d) [%(levelname)s]: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)

    # create the environment and an API used to translate actions into their corresponding IDs
    env = RacingKingsEnv()
    mvt = RacingKingsActions()

    # parse the specific configuration files in order to start building the class objects
    model_configuration = parse_config_file(args.nn_config, _type='nn_architecture')
    mcts_configuration = parse_config_file(args.mcts_config, _type='mcts_hyperparams')
    train_configuration = parse_config_file(args.train_config, _type='training')
    supervised_train_configuration = parse_config_file(args.supervised_train_config,
                                                       _type='supervised_training')

    # add the checkpoints dictionary path to the training configuration dictionary
    train_configuration['checkpoints_directory'] = args.nn_checkpoints

    # determine the device on which to build and train the NN
    device = torch.device(args.device)

    # add additional information to the NN configuration and initialize it
    model_configuration['input_shape'] = torch.Tensor(env.current_state_representation).shape
    model_configuration['num_actions'] = mvt.num_actions
    if args.generic:
        model = GenericNeuralNetwork(model_configuration, device).to(device)
    else:
        model = NeuralNetwork(model_configuration, device).to(device)

    # finally create the Chess agent
    logging.info('Creating AlphaZero agent.')
    chess_agent = AlphaZeroChessAgent(env=env,
                                      mvt=mvt,
                                      nn=model,
                                      device=device,
                                      mcts_config=mcts_configuration,
                                      train_config=train_configuration,
                                      pretrained_w=args.pre_trained_weights)

    # train the NN of the agent using supervised learning
    logging.info('Starting supervised learning.\n')
    chess_agent.train_agent_supervised(root_directory=args.data_root_directory,
                                       destination=args.parsed_data_destination_file,
                                       supervised_train_params=supervised_train_configuration,
                                       already_parsed_data=args.parsed_data)
    logging.info('Supervised Learning Complete.\n')

    # train the Chess agent using self play, while also keeping the previously observed examples
    logging.info('Starting self-play training.')
    chess_agent.train_agent(replay_buffer_had_data=True)
    logging.info('\nSelf-play training has been completed successfully.')


if __name__ == "__main__":
    print()
    arg = parse_supervised_train_input()
    main(arg)
