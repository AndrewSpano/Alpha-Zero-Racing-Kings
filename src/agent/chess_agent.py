"""
Created on February 2021

@author: Andreas Spanopoulos

Self-playing Racing Kings Chess agent that interacts with the RacingKings chess environment.
"""

import torch
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from collections import deque
from src.utils.main_utils import load_pytorch_model
from src.environment.variants.base_chess_env import ChessEnv
from src.environment.variants.actions.action_representations import MoveTranslator
from src.monte_carlo_search_tree.mcts import MCTS
from src.datasets.dataset import SelfPlayDataset


class RacingKingsChessAgent:
    """ Agent class """

    def __init__(self, env, mvt, nn, device, mcts_config, train_config=None, pretrained_w=None):
        """
        :param ChessEnv env:         The environment that interacts with the agent.
        :param MoveTranslator mvt:   API used to translate action to their corresponding IDs.
        :param torch.nn.Module nn:   The Neural Network to be trained from self play.
        :param torch.device device:  Device on which to train the model.
        :param dict mcts_config:     Dictionary containing information about the MCTS.
        :param dict train_config:    Dictionary containing information about the training procedure.
        :param str pretrained_w:     Path to the pre-trained weights of the NN.
        """
        self._env = env
        self._mvt = mvt
        self._nn = nn
        self._device = device
        self._mcts_config = mcts_config
        self._train_params = train_config
        if pretrained_w:
            load_pytorch_model(self._nn, path=pretrained_w, device=device)

    def play_episode(self):
        """
        :return:  A list of tuples of the sort (state, action_probabilities, outcome), where every
                    entry is describes the outcome from the POV of the corresponding player.
        :rtype:   list[tuple[list[int], list[list[list[int]]], int, list[float]]]

        Plays an episode of self play, keeps the state, outcome, and sampled action-value
        probabilities pi (from MCTS) and returns them in a list.
        """
        # reset the environment in order to start the game from the beginning
        self._env.reset()

        # list to store all the data for this episode
        examples = []

        # while the game is not over and it has not surpassed the maximum number of moves
        while not self._env.is_finished and self._env.moves < self._train_params['max_game_len']:

            # get the legal moves possible in this state and it's representation
            legal_actions = self._mvt.get_move_ids_from_uci(self._env.legal_moves)
            st = self._env.current_state_representation

            # start a Monte Carlo Tree Search in order to get the "optimal policy" pi
            mcts = MCTS(self._env, self._nn, self._mvt, self._mcts_config)

            # simulate the game, then sample the "optimal policy" and get the best action from it
            mcts.simulate()
            pi = mcts.sample_probabilities()
            best_action = np.argmax(pi)

            # create a training example (None will be replaced later with the POV winner)
            examples.append((legal_actions, st, None, pi))

            # play the next move in the environment
            self._env.play_move(self._mvt.move_from_id(best_action))

        # outcome of the game
        result = 0 if self._env.moves == self._train_params['max_game_len'] else self._env.winner

        # recompute the examples list, but this time it has also the outcome of the game
        data = []
        for ply, (legal_actions, st, _, pi) in enumerate(examples):

            # if the game was drawn
            if result == 0:
                pov_outcome = 0
            # if white (or black) is to move, and white (or black respectively) ended up winning
            elif (ply % 2 == 0 and result == 1) or (ply % 2 == 1 and result == -1):
                pov_outcome = 1
            # if white (or black) it to move, and black (or white respectively) ended up winning
            else:
                pov_outcome = -1

            data.append((legal_actions, st, pov_outcome, pi))

        return data

    def train_agent(self):
        """
        :return:  None
        :rtype:   None

        Trains the agent using self play for a fixed number of games, specified in the training
        configuration file.
        """

        # define the optimizer and the scheduler
        optimizer = optim.SGD(self._nn.parameters(),
                              lr=self._train_params['learning_rate'],
                              momentum=self._train_params['momentum'],
                              weight_decay=self._train_params['c'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=self._train_params['milestones'],
                                                   gamma=self._train_params['gamma'])

        # define the queue where the max deque_maxlen training examples will be stored
        training_deque = deque([], maxlen=self._train_params['max_deque_len'])

        # repeat the following pipeline: a) Play a number of self play episodes, b) update NN
        for iteration in tqdm(range(self._train_params['iterations']), desc='Self-play Pipeline'):

            # play episodes of self play, and update the deque accordingly
            for _ in range(self._train_params['self_play_episodes']):
                training_deque.append(*self.play_episode())

            # train the NN with the most recent examples from self play
            self._train_nn(training_deque, optimizer, scheduler)

    def _train_nn(self, training_deque, optimizer, scheduler):
        """
        :param deque training_deque:                      Deque with the most recent training data.
        :param optim.SGD optimizer:                       Optimizer for the loss function.
        :param optim.lr_scheduler.MultiStepLR scheduler:  Scheduler used to adapt the learning rate.
        :return:
        """
        # create a dataset with the current examples and a dataloader for it
        train_dataset = SelfPlayDataset(training_deque, device=self._device)
        # noinspection PyUnresolvedReferences
        train_dataloader = torch.utils.data.Dataloader(dataset=train_dataset, shuffle=True,
                                                       batch_size=self._train_params['batch_size'])

        for epoch in range(self._train_params['epochs']):
            pass

    def play_against(self, player_colour):
        """
        :param bool player_colour:  True if the opponent of AlphaZero starts with White; Else False.

        :return:  The outcome of the game (1 if white wins, -1 if black wins and 0 for draw)
        :rtype:   int

        Play a game against a trained alpha zero agent. The board should be displayed in your
        screen (make sure you have followed the steps in the docstring of the
        src/environment/base_chess_env.py script).
        """
        pass
