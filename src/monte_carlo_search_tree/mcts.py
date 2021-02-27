"""
Created on February 20 2021

@author: Andreas Spanopoulos

Implements Monte Carlo Tree Search.
"""


import time
import torch
import torch.nn.functional as F
import numpy as np

from src.utils.config_parsing_utils import parse_config_file
from src.environment.variants.base_chess_env import ChessEnv
from src.environment.actions.action_representations import MoveTranslator


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
        sum_n = sum([visit_count for action, visit_count in self.N.items()])

        # UCB score for every action
        ucbs = {action: self._ucb_score(action, c_puct, sum_n) for action in self.actions}

        # Node with highest UCB score
        return max(ucbs, key=lambda key: ucbs[key])

    def sample_optimal_action(self, tau):
        """
        :param float tau:  Temperature parameter used to promote exploration early in the game.

        :return:  The ID of the best action sampled.
        :rtype:   int
        """
        if tau != 0:
            # sum of visit counts
            sum_n = sum([visit_count ** (1 / tau) for action, visit_count in self.N.items()])
            # create a list with the action value probability pi(a|s) for each legal action
            pi = [(self.N[action] ** (1 / tau)) / sum_n for action in self.actions]

            # sample randomly from the distribution pi(a|s)
            return np.random.choice(self.actions, p=pi)
        else:
            # return child with highest visit count
            return max(self.N, key=lambda key: self.N[key])


class MCTS:
    """ class used to implement the Monte Carlo Tree Search algorithm """

    def __init__(self, env, nn, mvt, hyperparams):
        """
        :param ChessEnv env:        Current Chess Environment the agent operates in.
        :param torch.nn.Module.nn:  Neural Network used for prior probability prediction.
        :param MoveTranslator mvt:  Move Translator object used to convert move to their IDs.
        :param dict hyperparams:    Dictionary containing hyperparameters for the model.
        """
        # basic variables of the class
        self._env = env
        self._nn = nn
        self._mvt = mvt
        self._hyperparameters = hyperparams
        self._root_node = Node()

        # compute the available action from the root node
        available_actions_from_root = self._mvt.get_move_ids_from_uci(self._env.legal_moves)
        terminal_actions = self._actions_that_lead_to_terminal_state(env,
                                                                     available_actions_from_root)

        # compute prior probabilities for each available action using the NN
        with torch.no_grad():
            p, v = self._nn(torch.Tensor(self._env.current_state_representation).unsqueeze(0))
        action_to_prior = self._compute_prior_probabilities(available_actions_from_root, p)

        # initialize Search Tree: expand the root Node
        self._root_node.expand(available_actions_from_root, action_to_prior, terminal_actions)
        self._root_node.add_dirichlet_noise_to_prior_probabilities(hyperparams['dirichlet_alpha'],
                                                                   hyperparams['dirichlet_epsilon'])

    def _actions_that_lead_to_terminal_state(self, _env, available_actions):
        """
        :param ChessEnv _env:                Current Chess Environment the agent operates in.
        :param list[int] available_actions:  A list containing all the available action IDs.

        :return:  A list containing all the action IDs that lead to a terminal state.
        :rtype:   list[int]
        """
        return [action for action in available_actions
                if _env.is_terminal_move(self._mvt.move_from_id(action))]

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
        mask = torch.Tensor([float('-inf')] * self._mvt.num_actions)
        mask[actions] = 0

        # run the masked output through Softmax to get the prior probabilities of legal actions
        prior_probabilities = F.softmax(p + mask, dim=0)

        return {action: prior_probabilities[action].item() for action in actions}

    def _select_expand_backup(self, node, env_copy):
        """
        :param Node node:          The current Node we are in the Search Tree.
        :param ChessEnv env_copy:  A copy of the original environment used for MCTS.

        :returns:  The backup value of the first leaf or terminal node that is encountered.
        :rtype:    float

        Selects iteratively the Node with the highest UCB score in each level of the Search Tree,
        until it finds a leaf Node. If it is not terminal, it expands it, and then backups its
        action value to the higher Nodes of the search tree.
        """
        # if current Node is not a leaf or terminal Node, proceed to the next Node
        if not (node.is_leaf or node.is_terminal):

            # pick the next node from the next action
            best_action = node.action_with_highest_ucb_score(self._hyperparameters['c_puct'])
            env_copy.play_move(self._mvt.move_from_id(best_action))
            next_node = node.get_child_node_from_action(best_action)

            # get the backed up value from the child Node
            value = self._select_expand_backup(next_node, env_copy)

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
                elif (winner == 1 and self._env.side_to_move == 'w') or \
                     (winner == -1 and self._env.side_to_move == 'b'):
                    return 1
                else:
                    return -1

            # else if the Node is a leaf Node
            if node.is_leaf:

                # get the available actions and those that lead to a terminal state
                available_actions = self._mvt.get_move_ids_from_uci(env_copy.legal_moves)
                terminal_actions = self._actions_that_lead_to_terminal_state(env_copy,
                                                                             available_actions)

                # get action probabilities and value for current Node
                with torch.no_grad():
                    st = torch.Tensor(env_copy.current_state_representation).unsqueeze(0)
                    p, v = self._nn(st)
                action_to_prior = self._compute_prior_probabilities(available_actions, p)

                # expand the node
                node.expand(available_actions, action_to_prior, terminal_actions)

                # backup the value to the previous edges
                return v.item()

    def simulate(self):
        """
        :return:  None
        :rtype:   None

        Simulates num_iterations searches in the MCTS, expanding the Tree and backing up values
        along the way.
        """
        simulations = self._hyperparameters['num_iterations']
        for _ in range(simulations):
            self._select_expand_backup(self._root_node, self._env.copy())

    def sample_action(self):
        """
        :return:  The sampled action from the Root Node.
        :rtype:   int

        Computes the value: pi(a|s) = N(s_root, a) ^ {1/tau} / sum_b N(s_root, b) ^ {1/tau},
        and then samples and action from that distribution.
        """
        tau = self._hyperparameters['temperature_tau']
        degrade_at_step = self._hyperparameters['degrade_at_step']
        tau = tau if self._env.moves < degrade_at_step else 0

        return self._root_node.sample_optimal_action(tau=tau)

    def sample_best_action(self):
        """
        :return:  The best action from the Root Node, the one with the highest visit count.
        :rtype:   int

        Returns the legal action with the highest visit count N(root_node, a).
        """
        return self._root_node.sample_optimal_action(tau=0)


# for testing purposes
if __name__ == "__main__":

    from src.environment.variants.racing_kings import RacingKingsEnv
    environment = RacingKingsEnv()
    from src.environment.actions.racing_kings_actions import RacingKingsActions
    move_translator = RacingKingsActions()

    nn_config_path = '../../configurations/neural_network_architecture.ini'
    arch = parse_config_file(nn_config_path, _type='nn_architecture')
    arch['input_shape'] = torch.Tensor(environment.current_state_representation).shape
    arch['num_actions'] = move_translator.num_actions
    from src.neural_network.network import NeuralNetwork
    model = NeuralNetwork(arch, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()

    mcts_config_path = '../../configurations/mcts_hyperparams.ini'
    mcts_hyperparams = parse_config_file(mcts_config_path, _type='mcts_hyperparams')

    mcts = MCTS(environment, model, move_translator, mcts_hyperparams)

    start_time = time.time()
    mcts.simulate()
    print(f'Time taken to execute the MCTS is {time.time() - start_time}')

    pi_pred = mcts.sample_action()
    print(f'Optimal policy found: {pi_pred}')
    print(f'Move corresponding to it: {move_translator.move_from_id(pi_pred)}')
