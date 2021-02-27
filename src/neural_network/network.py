"""
Created on February 15 2021

@author: Andreas Spanopoulos

Implements the Alpha Zero Neural Network class.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from src.neural_network.network_utils import same_padding


class NeuralNetwork(nn.Module):
    """ Neural Network class used during Training of the Alpha Zero algorithm """

    def __init__(self, architecture, device):
        """
        :param dict architecture:    Dictionary describing the architecture of the model.
        :param torch.device device:  The device in which the model currently operates.
        """
        super(NeuralNetwork, self).__init__()
        self.device = device

        # define the architecture specifics
        self.input_shape = architecture['input_shape']
        self.conv_architecture = architecture['conv']
        self.res_architecture = architecture['res']
        self.num_residual_blocks = architecture['res']['num_res_blocks']
        self.policy_head_architecture = architecture['policy']
        self.num_actions = architecture['num_actions']
        self.value_head_architecture = architecture['value']

        # build convolutional block
        self.convolutional_block = self._build_conv_block()

        # build tower of pre-residual blocks
        self.preresidual_tower_block = [
            self._build_preresidual_block(self.conv_architecture['out_channels'])]
        for _ in range(self.num_residual_blocks - 1):
            self.preresidual_tower_block.append(
                self._build_preresidual_block(
                    self.res_architecture['out_channels']))

        # build the policy head block
        self.policy_head_block = self._build_policy_head_block()

        # build the value head block
        self.value_head_block = self._build_value_head_block()

    def _build_conv_block(self):
        """
        :return:  A PyTorch Sequential model representing the first convolutional block of the NN.
        :rtype:   torch.nn.Sequential
        """

        return nn.Sequential(
            nn.Conv2d(in_channels=self.input_shape[0],
                      out_channels=self.conv_architecture['out_channels'],
                      kernel_size=self.conv_architecture['kernel_size'],
                      stride=self.conv_architecture['stride'],
                      padding=same_padding((self.input_shape[0], self.input_shape[1]),
                                           self.conv_architecture['kernel_size'],
                                           self.conv_architecture['stride']),
                      padding_mode='zeros'),
            nn.BatchNorm2d(num_features=self.conv_architecture['out_channels']),
            nn.ReLU()
        )

    def _build_preresidual_block(self, in_channels):
        """
        :param int in_channels:  Number of input channels (filters) for the first convolutional
                                    layer of the pre-residual block.

        :return:  A PyTorch Sequential model representing the residual block until (before) the
                    skip connection takes place.
        :rtype:   torch.nn.Sequential
        """

        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=self.res_architecture['out_channels'],
                      kernel_size=self.res_architecture['kernel_size'],
                      stride=self.res_architecture['stride'],
                      padding=same_padding((self.input_shape[0], self.input_shape[1]),
                                           self.res_architecture['kernel_size'],
                                           self.res_architecture['stride']),
                      padding_mode='zeros'),
            nn.BatchNorm2d(num_features=self.res_architecture['out_channels']),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.res_architecture['out_channels'],
                      out_channels=self.res_architecture['out_channels'],
                      kernel_size=self.res_architecture['kernel_size'],
                      stride=self.res_architecture['stride'],
                      padding=same_padding((self.input_shape[0], self.input_shape[1]),
                                           self.res_architecture['kernel_size'],
                                           self.res_architecture['stride']),
                      padding_mode='zeros'),
            nn.BatchNorm2d(num_features=self.res_architecture['out_channels'])
        )

    @staticmethod
    def _forward_residual_block(res_block, x):
        """
        :param torch.nn.Sequential res_block:  Residual block to run through input x.
        :param torch.Tensor x:                 Tensor to be run through a residual block

        :return:  The output of the given residual block when inputted x.
        :rtype:   torch.Tensor
        """
        return F.relu(res_block(x) + x)

    def _build_policy_head_block(self):
        """
        :return:  A PyTorch Sequential model representing the policy head of the Neural Network.
        :rtype:   torch.nn.Sequential
        """

        in_features = np.prod(self.input_shape[1:]) * self.policy_head_architecture['out_channels']

        return nn.Sequential(
            nn.Conv2d(in_channels=self.res_architecture['out_channels'],
                      out_channels=self.policy_head_architecture['out_channels'],
                      kernel_size=self.policy_head_architecture['kernel_size'],
                      stride=self.policy_head_architecture['stride'],
                      padding=(0, 0),
                      padding_mode='zeros'),
            nn.BatchNorm2d(num_features=self.policy_head_architecture['out_channels']),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=in_features,
                      out_features=self.num_actions)
        )

    def _build_value_head_block(self):
        """
        :return:  A PyTorch Sequential model representing the value head of the Neural Network.
        :rtype:   torch.nn.Sequential
        """

        in_features = np.prod(self.input_shape[1:]) * self.value_head_architecture['out_channels']

        return nn.Sequential(
            nn.Conv2d(in_channels=self.res_architecture['out_channels'],
                      out_channels=self.value_head_architecture['out_channels'],
                      kernel_size=self.value_head_architecture['kernel_size'],
                      stride=self.value_head_architecture['stride'],
                      padding=(0, 0),
                      padding_mode='zeros'),
            nn.BatchNorm2d(num_features=self.value_head_architecture['out_channels']),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=in_features,
                      out_features=self.value_head_architecture['fc_output_dim']),
            nn.ReLU(),
            nn.Linear(in_features=self.value_head_architecture['fc_output_dim'],
                      out_features=1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        :param x:  Input Tensor to be forwarded through the Neural Network. [N x input_shape]

        :return:  The action policy probability and the value predictions [N x num_actions], [N x 1]
        :rtype:   torch.Tensor, torch.Tensor

        Run input through the following pipeline:

                convolutional block -> residual tower block -> policy & value head blocks
        """
        x = self.convolutional_block(x)
        for res_block in self.preresidual_tower_block:
            x = self._forward_residual_block(res_block, x)
        p = self.policy_head_block(x).exp()
        v = self.value_head_block(x)

        return p, v

    @staticmethod
    def criterion(z, v, pi, p, legal_actions):
        """
        :param torch.FloatTensor z:         Tensor containing the target values for the value head.
                                                [N x 1]
        :param torch.FloatTensor v:         Tensor containing the predicted value head values.
                                                [N x 1]
        :param torch.LongTensor pi:         Tensor containing the target values for the policy head.
                                                [N x 1]
        :param torch.FloatTensor p:         Tensor containing the predicted policy head values.
                                                [N x num_actions]
        :param torch.Tensor legal_actions:  Tensor containing the legal actions for every batch.

        :return:  The value of the Alpha Zero Loss function when given the above tensors.
        :rtype:   torch.FloatTensor

        Computes the loss function of the Alpha Zero algorithm using the following formula:

            L = (z - v)^2 - pi^T * log(p)
        """
        # create the mask for the legal actions
        mask = torch.Tensor([[float('-inf')] * p.shape[1]] * p.shape[0])
        for idx, batch_legal_actions in enumerate(legal_actions):
            mask[idx][batch_legal_actions] = 0

        # add the mask to the action value probabilities and compute the loss
        masked_p = p + mask
        return F.mse_loss(v, z) + F.cross_entropy(masked_p, pi)


# for testing purposes
if __name__ == "__main__":
    pass
    """
    from src.environment.variants.racing_kings import RacingKingsEnv
    env = RacingKingsEnv()
    from src.environment.actions.racing_kings_actions import RacingKingsActions
    mvt = RacingKingsActions()

    from src.utils.config_parsing_utils import parse_config_file
    config_path = '../../configurations/generic_neural_network_architecture.ini'
    arch = parse_config_file(config_path, _type='generic_nn_architecture')
    arch['input_shape'] = torch.Tensor(env.current_state_representation).shape
    arch['num_actions'] = mvt.num_actions

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNetwork(arch, device).to(device)

    inp = torch.FloatTensor(env.current_state_representation)
    # convert from (99, 8, 8) -> (1, 99, 8, 8)
    inp = inp.unsqueeze(0)
    pi_pred, v_pred = model(inp)

    print(pi_pred.shape)
    print(v_pred.shape)

    torch.set_printoptions(threshold=10_000)
    print(torch.sum(pi_pred))
    """

    # la = [[2, 7], [1, 3, 4], [12, 13]]
    # x = torch.randn(3, 20)
    # print(x)
    #
    # mask = torch.Tensor([[float('-inf')] * x.shape[1]] * x.shape[0])
    # print(mask)
    #
    # for idx, batch in enumerate(la):
    #     mask[idx][batch] = 0
    #
    # print(mask)
