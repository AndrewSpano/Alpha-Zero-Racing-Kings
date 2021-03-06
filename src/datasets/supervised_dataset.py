"""
Created on February 27 2021

@author: Andreas Spanopoulos

Implements a Dataset class used to group data from Human Expert play.
"""

import torch
import torch.utils.data
import pandas as pd

from collections import deque


class SupervisedDataset(torch.utils.data.Dataset):
    """ class used to represent the Dataset created by Human Expert play """

    def __init__(self, train_deque, device):
        """
        :param deque train_deque:    Deque containing game data from human play.
        :param torch.device device:  The device on which the data should be transferred to.
        """
        self.deque = train_deque
        self.device = device
        self.n_samples = len(train_deque)

    def save_data_to_destination(self, destination):
        """
        :param str destination:  The path to the destination file where the deque will be saved as
                                    a pickle file.

        :return:  None
        :rtype:   None
        """
        pd.to_pickle(self.deque, destination)

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
