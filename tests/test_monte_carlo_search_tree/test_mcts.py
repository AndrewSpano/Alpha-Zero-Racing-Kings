"""
Created on February 28 2021

@author: Andreas Spanopoulos
"""

import torch
import unittest

from src.utils.config_parsing_utils import parse_config_file
from src.environment.variants.racing_kings import RacingKingsEnv
from src.environment.actions.racing_kings_actions import RacingKingsActions
from src.neural_network.network import NeuralNetwork
from src.monte_carlo_search_tree.mcts import MCTS


class TestMonteCarloTreeSearch(unittest.TestCase):
    """ implements tests for the Monte Carlo Tree Search Class """

    @classmethod
    def setUpClass(cls) -> None:
        # initialize here class variables
        cls.env = RacingKingsEnv()
        cls.mvt = RacingKingsActions()

        # build NN
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        arch = parse_config_file('./neural_network_architecture.ini', _type='nn_architecture')
        arch['input_shape'] = torch.Tensor(cls.env.current_state_representation).shape
        arch['num_actions'] = cls.mvt.num_actions
        cls.nn = NeuralNetwork(arch, cls.device).eval()

        # store monte carlo tree search hyperparameters
        cls.mcts_hyperparams = parse_config_file('./mcts_hyperparams.ini', _type='mcts_hyperparams')
        cls.mcts = None

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def setUp(self) -> None:
        self.env.reset()
        self.mcts = MCTS(self.env, self.nn, self.mvt, self.mcts_hyperparams, self.device)

    def tearDown(self) -> None:
        pass

    def test_constructor(self):
        """ test the constructor of the object (basically a sanity check) """
        legal_actions = self.mvt.get_move_ids_from_uci(self.env.legal_moves)
        self.assertEqual(sorted(self.mcts._root_node.actions), sorted(legal_actions))
        self.assertFalse(self.mcts._root_node.is_leaf)
        self.assertFalse(self.mcts._root_node.is_terminal)

    def test_select_expand_backup(self):
        """ test the protected method() _select_expand_backup() """
        v = self.mcts._select_expand_backup(self.mcts._root_node, self.env.copy())
        self.assertTrue(-1 <= v <= 1)

    def test_simulate(self):
        """ test the simulate() method of the class, and make sure that the Q values are legit """
        self.mcts.simulate()
        legal_actions = self.mvt.get_move_ids_from_uci(self.env.legal_moves)
        for action in legal_actions:
            self.assertTrue(-1 <= self.mcts._root_node.Q[action] <= 1)

    def test_sample_action(self):
        """ test the sample_action() method """
        self.mcts.simulate()
        pi = self.mcts.sample_action()
        legal_actions = self.mvt.get_move_ids_from_uci(self.env.legal_moves)
        self.assertTrue(pi in legal_actions)

    def test_sample_best_action(self):
        """ test the sample_best_action() method returns the action with the highest visit count """
        self.mcts.simulate()
        pi = self.mcts.sample_best_action()
        legal_actions = self.mvt.get_move_ids_from_uci(self.env.legal_moves)
        best_action, best_visit_count = -1, -1
        same_best_counts = []
        for action in legal_actions:
            if self.mcts._root_node.N[action] > best_visit_count:
                best_action, best_visit_count = action, self.mcts._root_node.N[action]
                same_best_counts = [action]
            elif self.mcts._root_node.N[action] == best_visit_count:
                same_best_counts.append(action)
        self.assertTrue(pi in same_best_counts)
