"""
Created on February 16 2021

@author: Andreas Spanopoulos

Example usage:

python3 train.py
    --train-config ../configurations/training_hyperparams.ini
    --nn-config ../configurations/neural_network_architecture.ini
    --mcts-config ../configurations/mcts_hyperparams.ini
    --device cuda
"""

import torch

from src.utils.main_utils import parse_train_input
from src.utils.config_parsing_utils import parse_config_file

from src.environment.variants.racing_kings import RacingKingsEnv
from src.environment.actions.racing_kings_actions import RacingKingsActions
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
    train_configuration = parse_config_file(args.train_config, _type='training')

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
    chess_agent = RacingKingsChessAgent(env=env,
                                        mvt=mvt,
                                        nn=model,
                                        device=device,
                                        mcts_config=mcts_configuration,
                                        train_config=train_configuration,
                                        pretrained_w=args.pre_trained_weights)

    # train the Chess agent using self play
    chess_agent.train_agent()


if __name__ == "__main__":
    print()
    arg = parse_train_input()
    main(arg)
