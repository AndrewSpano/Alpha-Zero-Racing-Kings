"""
Created on February 25 2021

@author: Andreas Spanopoulos

Implements a Dataset class used to group the data that the Alpha Zero agent gathers from self play.
"""

import torch
import torch.utils.data

from collections import deque


class SelfPlayDataset(torch.utils.data.Dataset):
    """ class used to represent the Dataset created by Alpha Zero while performing multiple
        episodes of self play """

    def __init__(self, train_deque, device):
        """
        :param deque train_deque:    Deque containing all the most recent training examples.
        :param torch.device device:  The device on which the data should be transferred to.
        """
        self.deque = train_deque
        self.device = device
        self.n_samples = len(train_deque)

    def __getitem__(self, idx):
        """
        :param int idx:  The index of the data example we want to retrieve.

        :return:  A tuple containing information for the data pointed by the specified index.
        :rtype:   torch.LongTensor, torch.Tensor, int, int
        """
        legal_actions, st, z, pi = self.deque[idx]
        legal_actions = torch.LongTensor(legal_actions).to(self.device)
        st = torch.Tensor(st).to(self.device)
        return legal_actions, st, z, pi

    def __len__(self):
        """
        :return:  The length of the Dataset <=> The number of examples it contains.
        :rtype:   int
        """
        return self.n_samples
