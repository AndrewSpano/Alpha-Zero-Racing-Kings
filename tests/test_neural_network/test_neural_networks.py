"""
Created on February 28 2021

@author: Andreas Spanopoulos
"""

import torch
import torch.optim as optim
import unittest

from src.utils.config_parsing_utils import parse_config_file
from src.environment.variants.racing_kings import RacingKingsEnv
from src.environment.actions.racing_kings_actions import RacingKingsActions
from src.neural_network.network import NeuralNetwork
from src.neural_network.generic_network import GenericNeuralNetwork


class TestNeuralNetworks(unittest.TestCase):
    """ implements tests for the Monte Carlo Tree Search Class """

    @classmethod
    def setUpClass(cls) -> None:
        # initialize here class variables
        cls.env = RacingKingsEnv()
        cls.mvt = RacingKingsActions()

        # build NNs
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        arch1 = parse_config_file('./neural_network_architecture.ini', _type='nn_architecture')
        arch1['input_shape'] = torch.Tensor(cls.env.current_state_representation).shape
        arch1['num_actions'] = cls.mvt.num_actions
        cls.nn = NeuralNetwork(arch1, cls.device).eval()
        arch2 = parse_config_file('./generic_neural_network_architecture.ini',
                                  _type='generic_nn_architecture')
        arch2['input_shape'] = torch.Tensor(cls.env.current_state_representation).shape
        arch2['num_actions'] = cls.mvt.num_actions
        cls.gnn = GenericNeuralNetwork(arch2, cls.device).eval()

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def setUp(self) -> None:
        self.env.reset()

    def tearDown(self) -> None:
        pass

    def test_forward(self):
        """ test the forward() method of the NNs - make sure to error occurs """
        # convert from (99, 8, 8) -> (1, 99, 8, 8)
        st = torch.FloatTensor(self.env.current_state_representation).to(self.device).unsqueeze(0)
        # test nn
        with torch.no_grad():
            p, v = self.nn(st)
        self.assertEqual(p.shape[1], self.mvt.num_actions)
        self.assertTrue(-1 <= v.item() <= 1)
        # test generic nn
        with torch.no_grad():
            p, v = self.gnn(st)
        self.assertEqual(p.shape[1], self.mvt.num_actions)
        self.assertTrue(-1 <= v.item() <= 1)

    def test_criterion(self):
        """ test the loss function of the Networks """
        # convert from (99, 8, 8) -> (1, 99, 8, 8)
        st = torch.FloatTensor(self.env.current_state_representation).to(self.device).unsqueeze(0)
        legal_actions = torch.LongTensor([self.mvt.get_move_ids_from_uci(self.env.legal_moves)])
        # test nn
        with torch.no_grad():
            p, v = self.nn(st)
        z = torch.FloatTensor([[1]]).to(self.device)
        pi = torch.LongTensor([self.mvt.id_from_move('f2d4')]).to(self.device)
        self.nn.criterion(z, v, pi, p, legal_actions)
        # test generic nn
        with torch.no_grad():
            p, v = self.gnn(st)
        z = torch.FloatTensor([[1]]).to(self.device)
        pi = torch.LongTensor([self.mvt.id_from_move('f2d4')]).to(self.device)
        self.gnn.criterion(z, v, pi, p, legal_actions)

    def test_backprop(self):
        """ test the networks to see if they backpropagate successfully """
        # optimizers and schedulers
        nn_optimizer = optim.SGD(self.nn.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
        nn_scheduler = optim.lr_scheduler.MultiStepLR(nn_optimizer, milestones=[1, 5, 9], gamma=0.1)
        nn_optimizer.zero_grad()
        gnn_optimizer = optim.SGD(self.gnn.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
        gnn_scheduler = optim.lr_scheduler.MultiStepLR(gnn_optimizer, milestones=[1, 5], gamma=0.1)
        gnn_optimizer.zero_grad()

        # convert from (99, 8, 8) -> (1, 99, 8, 8)
        st = torch.FloatTensor(self.env.current_state_representation).to(self.device).unsqueeze(0)
        legal_actions = torch.LongTensor([self.mvt.get_move_ids_from_uci(self.env.legal_moves)])

        # test nn
        p, v = self.nn(st)
        z = torch.FloatTensor([[1]]).to(self.device)
        pi = torch.LongTensor([self.mvt.id_from_move('f2d4')]).to(self.device)
        loss = self.nn.criterion(z, v, pi, p, legal_actions)
        loss.backward()
        nn_optimizer.step()
        nn_scheduler.step()
        # test generic nn
        p, v = self.gnn(st)
        z = torch.FloatTensor([[1]]).to(self.device)
        pi = torch.LongTensor([self.mvt.id_from_move('f2d4')]).to(self.device)
        loss = self.gnn.criterion(z, v, pi, p, legal_actions)
        loss.backward()
        gnn_optimizer.step()
        gnn_scheduler.step()
