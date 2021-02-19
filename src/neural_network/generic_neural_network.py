"""
Created on February 18 2021

@author: Andreas Spanopoulos

Implements the Generic Neural Networks class. Tha main differences from the basic Neural Network
are that this one is highly configurable, but harder to read and understand. It can be used for
experimenting with different hyperparameters like number-of-residual-blocks, convolutional
kernel_sizes, strides, etc.

I strongly recommend first understanding the other NN (network.py) and then moving onto this one.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from src.neural_network.network_utils import compute_output_shape, same_padding
from src.utils.parsing_utils import parse_config_file
from src.racing_kings_environment.racing_kings import RacingKingsEnv


class GenericNeuralNetwork(nn.Module):
    """ Neural Network class used during Training of the Alpha Zero algorithm """

    def __init__(self, architecture):
        """
        :param dict architecture:  Dictionary describing the architecture of the model.
        """
        super(GenericNeuralNetwork, self).__init__()

        # define the architecture specifics
        self.conv_architecture = architecture['conv']
        self.res_architecture = architecture['res']
        self.policy_head_architecture = architecture['policy']
        self.value_head_architecture = architecture['value']

        # useful values
        self.same_padding = architecture['padding']['same_padding']
        self.padding_mode = architecture['padding']['mode']
        self.num_residual_blocks = architecture['res']['num_res_blocks']

        # keep track of the shape of the data throughout the model pipeline
        self.input_shape = architecture['input_shape']
        self.num_actions = architecture['num_actions']

        # building the convolutional block
        in_shape = (self.input_shape[1], self.input_shape[2])
        in_channels = self.input_shape[0]
        self.convolutional_block, conv_out_shape = self._build_conv_block(in_shape, in_channels)

        # build the first preresidual block
        conv_out_channels = self.conv_architecture['out_channels']
        res_block, out_shape = self._build_preresidual_block(conv_out_shape, conv_out_channels)

        # start building the tower (list) of residual blocks
        self.preresidual_tower_block = [res_block]
        for _ in range(self.num_residual_blocks - 1):
            out_channels = self.res_architecture['out_channels_2']
            res_block, out_shape = self._build_preresidual_block(out_shape, out_channels)
            self.preresidual_tower_block.append(res_block)
        out_channels = self.res_architecture['out_channels_2']

        # build the policy head block
        self.policy_head_block = self._build_policy_head_block(out_shape, out_channels)

        # build the value head block
        self.value_head_block = self._build_value_head_block(out_shape, out_channels)

    def _build_conv_block(self, in_shape, in_channels):
        """
        :param tuple in_shape:   Shape of the input that will run through the convolutional block.
        :param int in_channels:  Number of input channels (filters) for the convolutional layer of
                                    the convolutional block.

        :return:  A PyTorch Sequential model representing a convolutional block, and the output
                    shape of an object that would run through it.
        :rtype:   torch.nn.Sequential, tuple
        """

        if self.same_padding:
            padding = same_padding(in_shape,
                                   self.conv_architecture['kernel_size'],
                                   self.conv_architecture['stride'])
            out_shape = in_shape
        else:
            padding = self.conv_architecture['padding']
            out_shape = compute_output_shape(in_shape,
                                             self.conv_architecture['kernel_size'],
                                             self.conv_architecture['stride'],
                                             padding)

        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=self.conv_architecture['out_channels'],
                      kernel_size=self.conv_architecture['kernel_size'],
                      stride=self.conv_architecture['stride'],
                      padding=padding,
                      padding_mode=self.padding_mode),
            nn.BatchNorm2d(num_features=self.conv_architecture['out_channels']),
            nn.ReLU()
        ), out_shape

    def _build_preresidual_block(self, in_shape, in_channels_1):
        """
        :param tuple in_shape:     Shape of the input that will run through the pre-residual block.
        :param int in_channels_1:  Number of input channels (filters) for the convolutional layer of
                                    the pre-residual block.

        :return:  A PyTorch Sequential model representing the residual block until (before) the
                    skip connection takes place, and the output shape of an object that would run
                    through it.
        :rtype:   torch.nn.Sequential, tuple
        """

        if self.same_padding:
            padding_1 = same_padding(in_shape,
                                     self.res_architecture['kernel_size_1'],
                                     self.res_architecture['stride_1'])
            res_conv1_out_shape = in_shape
            padding_2 = same_padding(res_conv1_out_shape,
                                     self.res_architecture['kernel_size_2'],
                                     self.res_architecture['stride_2'])
            res_conv2_out_shape = res_conv1_out_shape
        else:
            padding_1 = self.res_architecture['padding_1']
            res_conv1_out_shape = compute_output_shape(in_shape,
                                                       self.res_architecture['kernel_size_1'],
                                                       self.res_architecture['stride_1'],
                                                       padding_1)
            padding_2 = self.res_architecture['padding_2']
            res_conv2_out_shape = compute_output_shape(res_conv1_out_shape,
                                                       self.res_architecture['kernel_size_2'],
                                                       self.res_architecture['stride_2'],
                                                       padding_2)

        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels_1,
                      out_channels=self.res_architecture['out_channels_1'],
                      kernel_size=self.res_architecture['kernel_size_1'],
                      stride=self.res_architecture['stride_1'],
                      padding=padding_1,
                      padding_mode=self.padding_mode),
            nn.BatchNorm2d(num_features=self.res_architecture['out_channels_1']),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.res_architecture['out_channels_1'],
                      out_channels=self.res_architecture['out_channels_2'],
                      kernel_size=self.res_architecture['kernel_size_2'],
                      stride=self.res_architecture['stride_2'],
                      padding=padding_2,
                      padding_mode=self.padding_mode),
            nn.BatchNorm2d(num_features=self.res_architecture['out_channels_2'])
        ), res_conv2_out_shape

    @staticmethod
    def _forward_residual_block(res_block, x):
        """
        :param torch.nn.Sequential res_block:  Residual block to run through input x.
        :param torch.Tensor x:                 Tensor to be run through a residual block

        :return:  The output of the given residual block when inputted x.
        :rtype:   torch.Tensor
        """
        return F.relu(res_block(x) + x)

    def _build_policy_head_block(self, in_shape, in_channels):
        """
        :param tuple in_shape:   Shape of the input that will run through the policy head block.
        :param int in_channels:  Number of input channels (filters) for the convolutional layer of
                                    the policy head block.

        :return:  A PyTorch Sequential model representing the policy head of the Neural Network.
        :rtype:   torch.nn.Sequential
        """

        out_shape = compute_output_shape(in_shape,
                                         self.policy_head_architecture['kernel_size'],
                                         self.policy_head_architecture['stride'],
                                         self.policy_head_architecture['padding'])
        in_features = np.prod(out_shape) * self.policy_head_architecture['out_channels']

        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=self.policy_head_architecture['out_channels'],
                      kernel_size=self.policy_head_architecture['kernel_size'],
                      stride=self.policy_head_architecture['stride'],
                      padding=self.policy_head_architecture['padding'],
                      padding_mode=self.padding_mode),
            nn.BatchNorm2d(num_features=self.policy_head_architecture['out_channels']),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=in_features,
                      out_features=self.num_actions)
        )

    def _build_value_head_block(self, in_shape, in_channels):
        """
        :param tuple in_shape:   Shape of the input that will run through the value head block.
        :param int in_channels:  Number of input channels (filters) for the convolutional layer of
                                    the value head block.

        :return:  A PyTorch Sequential model representing the value head of the Neural Network.
        :rtype:   torch.nn.Sequential
        """

        out_shape = compute_output_shape(in_shape,
                                         self.value_head_architecture['kernel_size'],
                                         self.value_head_architecture['kernel_size'],
                                         self.value_head_architecture['padding'])
        in_features = np.prod(out_shape) * self.value_head_architecture['out_channels']

        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=self.value_head_architecture['out_channels'],
                      kernel_size=self.value_head_architecture['kernel_size'],
                      stride=self.value_head_architecture['stride'],
                      padding=self.value_head_architecture['padding'],
                      padding_mode=self.padding_mode),
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
    def criterion(z, v, pi, p):
        """
        :param torch.Tensor z:   Tensor containing the target values for the value head.
                                    [N x num_actions]
        :param torch.Tensor v:   Tensor containing the predicted value head values.
                                    [N x num_actions]
        :param torch.Tensor pi:  Tensor containing the target values for the policy head.
                                    [N x 1]
        :param torch.Tensor p:   Tensor containing the predicted policy head values.
                                    [N x 1]

        :return:  The value of the Alpha Zero Loss function when given the above tensors.
        :rtype:   torch.Tensor

        Computes the loss function of the Alpha Zero algorithm using the following formula:

            L = (z - v)^2 - pi^T * log(p)
        """
        # return F.mse_loss(v, z) + F.cross_entropy(pi. p)
        return torch.square(z - v) + F.cross_entropy(pi, p)


env = RacingKingsEnv()
inp = torch.FloatTensor(env.representation_of_starting_fen(t_history=8))
# convert from (99, 8, 8) -> (1, 99, 8, 8)
inp = inp.unsqueeze(0)

config_path = '../../configurations/generic_neural_network_architecture.ini'
arch = parse_config_file(config_path, _type='generic_nn_architecture')
arch['input_shape'] = (99, 8, 8)
arch['num_actions'] = 8 * 8 * 64

model = GenericNeuralNetwork(arch)

pi_pred, v_pred = model(inp)

print(pi_pred.shape)
print(v_pred.shape)

torch.set_printoptions(threshold=10_000)
print(torch.sum(pi_pred))
