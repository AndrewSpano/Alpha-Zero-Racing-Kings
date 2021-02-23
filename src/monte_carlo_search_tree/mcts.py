"""
Created on February 20 2021

@author: Andreas Spanopoulos

Implements Monte Carlo Tree Search.
"""


import torch
import torch.nn.functional as F
import operator
import numpy as np

from src.racing_kings_environment.racing_kings import RacingKingsEnv
from src.racing_kings_environment.action_representations import MoveTranslator
from src.neural_network.network import NeuralNetwork
from src.utils.parsing_utils import parse_config_file


class Node:
    """ class used to represent a node (state) in a Monte Carlo Search Tree """

    def __init__(self, is_terminal=False):
        """
        :param bool is_terminal:  Whether the state which the Node represents it terminal or not.
        """
        # list containing all the action IDs that can be taken from this Node (state)
        self.actions = []
        # children nodes: {action -> Node}
        self.children = {}
        # visit count of every available action from node
        self.N = {}
        # sum of action values for all actions take in this node
        self.W = {}
        # W / N
        self.Q = {}
        # prior probability as computed by NN
        self.P = {}

        self._is_leaf = True
        self._is_terminal = is_terminal

    @property
    def is_leaf(self):
        """
        :return:  True if the Node is a leaf Node; Else False.
        :rtype:   bool
        """
        return self._is_leaf

    @property
    def is_terminal(self):
        """
        :return:  True if the Node represents a terminal state s_T; Else False.
        :rtype:   bool
        """
        return self._is_terminal

    def get_child_node_from_action(self, action):
        """
        :param int action: Action ID available from current Node.

        :return:  The Node corresponding to that action being taken from the current Node.
        :rtype:   Node
        """
        return self.children[action]

    def expand(self, actions, prior_probabilities, terminal_actions):
        """
        :param list[int] actions:           A list of action IDs that can be taken.
        :param dict prior_probabilities:    Dictionary that maps an action ID to its prior
                                                probability.
        :param list[int] terminal_actions:  A sub-set list of the available actions, that lead to
                                                a terminal state

        :return:  None
        :rtype:   None

        Expands a Node by initializing it's class variables.
        """
        self.actions = actions
        for action in actions:
            is_terminal = action in terminal_actions
            self.children[action] = Node(is_terminal=is_terminal)
        self.N = {action: 0 for action in actions}
        self.W = {action: 0 for action in actions}
        self.Q = {action: 0 for action in actions}
        self.P = {action: prior_probabilities[action] for action in actions}
        self._is_leaf = False

    def backup(self, action, value):
        """
        :param int action:   The action that was taken during a MCTS.
        :param float value:  The backed up value for that action.

        :return:  None
        :rtype:   None

        Back-ups the value computed from a lower lever of the MCTS by:
            1) Incrementing the visit count of that action by 1
            2) Adding the value to the sum of action values
            3) Re-computing the average action-value
        """
        self.N[action] += 1
        self.W[action] += value
        self.Q[action] = self.W[action] / self.N[action]

    def add_dirichlet_noise_to_prior_probabilities(self, alpha, epsilon):
        """
        :param list[float] alpha:  List of 3 choices (depending on number of action) for Dirichlet
                                    noise input.
        :param float epsilon:      Proportion of Dirichlet Noise.

        :return:  None
        :rtype:   None

        Adds Dirichlet Noise to the prior probabilities using the formula:

                P(s, a) = (1 - epsilon)p_a + epsilon * eta_a, where eta_a ~ Dir(alpha)
        """
        # list values should be in decreasing order
        a = alpha[0] if len(self.actions) < 7 else alpha[1] if len(self.actions) < 15 else alpha[2]
        # compute the noise distribution and then recompute the prior probabilities
        noise = np.random.dirichlet([a] * len(self.actions))
        for action, dirichlet_noise in zip(self.actions, noise):
            self.P[action] = (1 - epsilon) * self.P[action] + epsilon * dirichlet_noise

    def _ucb_score(self, action, c_puct, sum_n):
        """ Computes UCB score for an action using the formula:
            UCB(a) = Q(a) + U(a) = Q(a) + c_puct * P(a) * sqrt(sum(N) / (N(a) + 1)) """
        return self.Q[action] + c_puct * self.P[action] * np.sqrt(sum_n / (self.N[action] + 1))

    def action_with_highest_ucb_score(self, c_puct=1.0):
        """
        :param float c_puct:  Exploration parameter.

        :return:  The action with the highest UCB score. If a Node has not been expanded
                    (i.e. is a leaf), then the score is infinite.
        :rtype:   int

        Computes the Upper Confidence Bound score for every available action using the formula:

                UCB(a) = Q(a) + U(a) = Q(a) + c_puct * P(a) * sqrt(sum(N) / (N(a) + 1))

        and returns the action a with the highest UCB(a) value.
        """
        # sum of visit counts
        sum_n = sum([visit_count for action, visit_count in self.N.items()
                     if not self.children[action].is_leaf])

        # UCB score for every action
        ucbs = {action: self._ucb_score(action, c_puct, sum_n) for action in self.actions}

        # Node with highest UCB score
        return max(ucbs.items(), key=operator.itemgetter(1))[0]


class MCTS:
    """ class used to implement the Monte Carlo Tree Search algorithm """

    def __init__(self, _env, _nn, _mvt, _hyperparams):
        """
        :param RacingKingsEnv _env:  Current Environment.
        :param NeuralNetwork _nn:    Neural Network used for prior probability prediction.
        :param MoveTranslator _mvt:  Move Translator object used to convert move to their IDs.
        :param dict _hyperparams:    Dictionary containing hyperparameters for the model.
        """
        # basic variables of the class
        self.env = _env
        self.nn = _nn
        self.mvt = _mvt
        self.hyperparameters = _hyperparams
        self.root_node = Node()

        # compute the available action from the root node
        available_actions_from_root = self.mvt.get_move_ids_from_uci(self.env.legal_moves)
        terminal_actions = self._actions_that_lead_to_terminal_state(_env,
                                                                     available_actions_from_root)

        # compute prior probabilities for each available action using the NN
        with torch.no_grad():
            p, v = self.nn(torch.Tensor(self.env.current_state_representation).unsqueeze(0))
        action_to_prior = self._compute_prior_probabilities(available_actions_from_root, p)

        # initialize Search Tree: expand the root Node
        self.root_node.expand(available_actions_from_root, action_to_prior, terminal_actions)
        self.root_node.add_dirichlet_noise_to_prior_probabilities(_hyperparams['dirichlet_alpha'],
                                                                  _hyperparams['dirichlet_epsilon'])

    def _actions_that_lead_to_terminal_state(self, _env, available_actions):
        """
        :param RacingKingsEnv _env:          The environment of the agent.
        :param list[int] available_actions:  A list containing all the available action IDs.

        :return:  A list containing all the action IDs that lead to a terminal state.
        :rtype:   list[int]
        """
        terminal_actions = []
        for action in available_actions:
            env_copy = _env.copy()
            env_copy.play_move(self.mvt.get_move(action))
            if env_copy.is_finished:
                terminal_actions.append(action)

        return terminal_actions

    def _compute_prior_probabilities(self, actions, p):
        """
        :param list[int] actions:  A list of available actions from a Node.
        :param torch.Tensor p:     The output of the NN when given a state representation.

        :return:  A dictionary mapping {action -> prior probability}
        :rtype:   dict
        """
        # remove batch dimension
        p = p.squeeze(0)

        # initialize a mask tensor with -infinite values where the actions are illegal
        mask = torch.Tensor([float('-inf')] * self.mvt.num_actions)
        for action in actions:
            mask[action] = 0

        # run the masked output through Softmax to get the prior probabilities of legal actions
        prior_probabilities = F.softmax(p + mask, dim=0)

        return {action: prior_probabilities[action] for action in actions}

    def select(self, node, env_copy):
        """
        :param Node node:                The current Node we are in the Search Tree.
        :param RacingKingsEnv env_copy:  A copy of the original environment used for MCTS.

        :returns:  The backup value of the first leaf or terminal node that is encountered.
        :rtype:    float

        Selects iteratively the Node with the highest UCB score in each level of the Search Tree,
        until it finds a leaf Node.
        """
        # if current Node is not a leaf or terminal Node, proceed to the next Node
        if not (node.is_leaf or node.is_terminal):

            # pick the next node from the next action
            best_action = node.action_with_highest_ucb_score(self.hyperparameters['c_puct'])
            env_copy.play_move(self.mvt.get_move(best_action))
            next_node = node.get_child_node_from_action(best_action)

            # get the backed up value from the child Node
            value = self.select(next_node, env_copy)

            # backup the value for the current Node also
            node.backup(best_action, value)

            # back the value computed to a higher level of the search Tree
            return value

        # else, if we are at a terminal or leaf Node
        else:

            # if the current Node is a terminal Node, return the result of the game
            if node.is_terminal:
                winner = env_copy.winner
                # draw
                if winner == 0:
                    return 0
                elif winner == 1:
                    # white won and white was to play in root state
                    if self.env.side_to_move == 'w':
                        return 1
                    # white won and black was to play in root state
                    else:
                        return -1
                else:
                    # black won and white was to play in root state
                    if self.env.side_to_move == 'w':
                        return -1
                    # black won and black was to play in root state
                    else:
                        return 1

            # else if the Node is a leaf Node
            if node.is_leaf:

                # get the available actions and those that lead to a terminal state
                available_actions = self.mvt.get_move_ids_from_uci(env_copy.legal_moves)
                terminal_actions = self._actions_that_lead_to_terminal_state(env_copy,
                                                                             available_actions)

                # get action probabilities and value for current Node
                with torch.no_grad():
                    p, v = self.nn(torch.Tensor(env_copy.current_state_representation).unsqueeze(0))
                action_to_prior = self._compute_prior_probabilities(available_actions, p)

                # expand the node
                node.expand(available_actions, action_to_prior, terminal_actions)

                # backup the value to the previous edges
                return v

    def simulate(self):
        """
        :return:  None
        :rtype:   None

        Simulates num_iterations searches in the MCTS, expanding the Tree and backing up values
        along the way.
        """
        simulations = self.hyperparameters['num_iterations']
        # simulations = 1
        for _ in range(simulations):
            print(_)
            self.select(self.root_node, self.env.copy())


# for testing purposes
if __name__ == "__main__":

    env = RacingKingsEnv()
    mvt = MoveTranslator()

    nn_config_path = '../../configurations/neural_network_architecture.ini'
    arch = parse_config_file(nn_config_path, _type='nn_architecture')
    arch['input_shape'] = torch.Tensor(env.current_state_representation).shape
    arch['num_actions'] = mvt.num_actions
    model = NeuralNetwork(arch)
    model.eval()

    mcts_config_path = '../../configurations/mcts_hyperparams.ini'
    mcts_hyperparams = parse_config_file(mcts_config_path, _type='mcts_hyperparams')

    mcts = MCTS(env, model, mvt, mcts_hyperparams)

    mcts.simulate()
