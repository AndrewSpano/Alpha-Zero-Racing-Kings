"""
Created on February 23 2021

@author: Andreas Spanopoulos

Self-playing Racing Kings Chess agent that interacts with the RacingKings chess environment.
"""

import os
import logging
import pandas as pd
import torch
import torch.utils.data
import torch.optim as optim

from tqdm import tqdm
from collections import deque
from chessboard import display
from src.utils.main_utils import save_pytorch_model, load_pytorch_model, pad_to_maxlen, \
    get_legal_move_from_player
from src.environment.variants.base_chess_env import ChessEnv
from src.environment.actions.action_representations import MoveTranslator
from src.monte_carlo_search_tree.mcts import MCTS
from src.datasets.dataset import SelfPlayDataset
from src.datasets.supervised_dataset import SupervisedDataset
from src.datasets.dataset_utils import parse_data


class AlphaZeroChessAgent:
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
        self._replay_buffer = None
        if pretrained_w:
            load_pytorch_model(self._nn, path=pretrained_w, device=device)

    def play_episode(self):
        """
        :return:  A list of tuples of the sort (state, action_probabilities, outcome), where every
                    entry is describes the outcome from the POV of the corresponding player.
        :rtype:   list[tuple[list[int], list[list[list[int]]], int, int]]

        Plays an episode of self play, keeps the state, it's legal actions, the outcome and the
        sampled optimal action pi (from MCTS), and returns them in a list.
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
            # pad the legal actions list, as they all must have the same length for the dataloader
            pad_to_maxlen(legal_actions, maxlen=self._mvt.legal_moves_upper_bound)

            # start a Monte Carlo Tree Search in order to get the "optimal policy" pi
            mcts = MCTS(self._env, self._nn, self._mvt, self._mcts_config, self._device)

            # simulate the game, then sample the "optimal policy" and get the best action from it
            mcts.simulate()
            pi = mcts.sample_action()

            # create a training example (POV result will be added later)
            examples.append((legal_actions, st, pi))

            # play the next move in the environment
            self._env.play_move(self._mvt.move_from_id(pi))

        # outcome of the game
        result = 0 if self._env.moves == self._train_params['max_game_len'] else self._env.winner

        # recompute the examples list, but this time it has also the outcome of the game
        data = []
        for ply, (legal_actions, st, pi) in enumerate(examples):

            # if the game was drawn
            if result == 0:
                z = 0
            # if white (or black) is to move, and white (or black respectively) ended up winning
            elif (ply % 2 == 0 and result == 1) or (ply % 2 == 1 and result == -1):
                z = 1
            # if white (or black) it to move, and black (or white respectively) ended up winning
            else:
                z = -1

            data.append((legal_actions, st, z, pi))

        return data

    def _train_nn(self, dataloader, optimizer, scheduler, epochs):
        """
        :param torch.utils.data.Dataloader dataloader:    Deque with the most recent training data.
        :param optim.SGD optimizer:                       Optimizer for the loss function.
        :param optim.lr_scheduler.MultiStepLR scheduler:  Scheduler used to adapt the learning rate.
        :param int epochs:                                Number of epochs to train the model.

        :return:  None
        :rtype:   None
        """

        # place the neural network in train mode
        self._nn.train()

        # perform 'epochs' passed through the dataset of training
        for epoch in range(epochs):

            # store the sum of the losses to print the average in the end of the epoch
            sum_loss = 0

            # for legal_actions, st, z, pi in train_dataloader:
            for legal_actions, st, z, pi in dataloader:

                # reshape target
                z = z.reshape(-1, 1)

                # reset gradients
                optimizer.zero_grad()

                # forward pass with the NN
                p, v = self._nn(st)

                # compute the loss
                z = z.to(self._device).float()
                pi = pi.to(self._device).long()
                loss = self._nn.criterion(z, v, pi, p, legal_actions)
                sum_loss += loss.item()
                # avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(self._nn.parameters(), self._train_params['clip'])

                # backpropagate
                loss.backward()

                # optimizer and scheduler steps
                optimizer.step()
                scheduler.step()

            # log information about the average epoch loss
            logging.info(f'Average Loss of epoch {epoch + 1}: {sum_loss / len(dataloader):.2f}')

    def _save_nn_weights(self, iteration):
        """
        :param int iteration:  Current iteration of the self-play-update pipeline.

        :return:  None
        :rtype:   None

        Saves the NN weights in the directory specified by the training configuration file.
        The name of the file containing the weights is:
            parent_dir/iteration_{iteration}_weights.pth
        """
        path = os.path.join(self._train_params['checkpoints_directory'],
                            f'iteration_{iteration}_weights.pth')
        logging.info(f'Saving Model weights at path: {path}.')
        save_pytorch_model(self._nn, path)

    def train_agent(self, replay_buffer_had_data=False):
        """
        :param bool replay_buffer_had_data:  True if the self.replay_buffer object has already
                                                data in it; Else False.

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

        # define the replay buffer where the most recent deque_maxlen training examples are stored
        if replay_buffer_had_data is False:
            self._replay_buffer = deque([], maxlen=self._train_params['max_deque_len'])

        # repeat the following pipeline: a) Play a number of self play episodes, b) update NN
        for iteration in tqdm(range(self._train_params['iterations']), desc='\nTraining Pipeline'):

            # log some information
            logging.info(f'Starting self-play iteration {iteration + 1}.')

            # place the neural network in evaluation mode for faster inference
            self._nn.eval()

            # execute episodes of self play, and update the deque accordingly
            for _ in tqdm(range(self._train_params['self_play_episodes']), position=0, leave=True):
                self._replay_buffer += self.play_episode()

            # create a dataset with the current examples and a dataloader for it
            dataset = SelfPlayDataset(self._replay_buffer, device=self._device)
            dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                     batch_size=self._train_params['batch_size'],
                                                     shuffle=True)

            # log some information
            logging.info(f'Starting NN training iteration {iteration + 1}.')

            # train the NN with the most recent examples from self play
            self._train_nn(dataloader, optimizer, scheduler, epochs=self._train_params['epochs'])

            # save the weights if it's time to
            if iteration % self._train_params['checkpoint_every'] == 0:
                self._save_nn_weights(iteration + 1)

    def train_agent_supervised(self, root_directory, destination, supervised_train_params,
                               already_parsed_data=None):
        """
        :param str root_directory:            The root directory containing the data.
        :param str destination:               The destination file to store the parsed data.
        :param dict supervised_train_params:  Dictionary containing information about the
                                                supervised training procedure.
        :param str already_parsed_data:       Path to a pickle file containing already parsed
                                                data, to avoid re-parsing it. If not specified,
                                                then the data is parsed from scratch.

        :return:  None
        :rtype:   None

        Trains agent using behavioural cloning, by using Human Expert data.
        """
        # load the data if it had already been parsed
        if already_parsed_data is not None:
            self._replay_buffer = pd.read_pickle(already_parsed_data)
            logging.info(f'Successfully read parsed data. The number of training examples is: '
                         f'{len(self._replay_buffer)}')
        else:
            # else parse the data and create a replay buffer
            self._replay_buffer = parse_data(root_directory=root_directory,
                                             env=self._env.copy(),
                                             mvt=self._mvt,
                                             deque_maxlen=self._train_params['max_deque_len'],
                                             min_white_elo=supervised_train_params['min_white_elo'],
                                             min_black_elo=supervised_train_params['min_black_elo'],
                                             worse_games=supervised_train_params['worse_games'])

        # create a dataset from the observed examples, save them and create a dataloader
        dataset = SupervisedDataset(self._replay_buffer, device=self._device)
        if already_parsed_data is None:
            dataset.save_data_to_destination(destination)
            logging.info(f'Successfully saved the parsed data in the destination: {destination}.')
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=supervised_train_params['batch_size'],
                                                 shuffle=True)

        # define the optimizer and the scheduler
        optimizer = optim.SGD(self._nn.parameters(),
                              lr=supervised_train_params['learning_rate'],
                              momentum=supervised_train_params['momentum'],
                              weight_decay=supervised_train_params['c'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=supervised_train_params['milestones'],
                                                   gamma=supervised_train_params['gamma'])

        # train the NN with the human play data and save the weights after
        epochs = supervised_train_params['epochs']
        logging.info(f"Starting Neural Network training for {epochs} epochs.")
        self._train_nn(dataloader, optimizer, scheduler, epochs=epochs)
        self._save_nn_weights(0)

    def play_against(self, player_has_white, use_display=False):
        """
        :param bool player_has_white:  True if the opponent of AlphaZero starts with White;
                                        Else False.
        :param bool use_display:       True if the user wants a display of the board to be printed
                                        in the screen. If False, then only the moves can be seen
                                        from the console.

        :return:  The outcome of the game (1 if white wins, -1 if black wins and 0 for draw)
        :rtype:   int

        Play a game against a trained alpha zero agent. The board should be displayed in your
        screen (make sure you have followed the steps in the docstring of the
            src/environment/variants/base_chess_env.py script).
        """
        # first reset the environment, as this method may be called multiple times
        self._env.reset()

        # display the board, and keep it up until the game is over
        if use_display:
            display.start(self._env.fen)

        # if the player starts as white, get his first move
        if player_has_white:
            move = get_legal_move_from_player(self._env.legal_moves, self._env.legal_moves_san)
            if move == 'resign':
                if use_display:
                    display.terminate()
                return -1
            self._env.play_move(move)
            if use_display:
                display.update(self._env.fen)

        # while the game is not over, keep iterating
        while not self._env.is_finished:

            # agent makes a move
            mcts = MCTS(self._env, self._nn, self._mvt, self._mcts_config, self._device)
            mcts.simulate(evaluation=True)
            agent_move = self._mvt.move_from_id(mcts.sample_best_action())
            print(f'Agent plays: {self._env.san_from_uci(agent_move)}\n')
            self._env.play_move(agent_move)
            if use_display:
                display.update(self._env.fen)

            # if the game terminates with the agents move
            if self._env.is_finished:
                if use_display:
                    display.terminate()
                return self._env.winner

            # else, get the next move from the player
            move = get_legal_move_from_player(self._env.legal_moves, self._env.legal_moves_san)
            if move == 'resign':
                if use_display:
                    display.terminate()
                return -1 if player_has_white else 1
            self._env.play_move(move)
            if use_display:
                display.update(self._env.fen)

        # game has finished, close the display and return the result
        if use_display:
            display.terminate()
        return self._env.winner
