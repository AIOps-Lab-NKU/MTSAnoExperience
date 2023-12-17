import pathlib
import torch
import torch.nn as nn

import numpy as np
from dagmm.compression_net import CompressionNet
from dagmm.estimation_net import EstimationNet
from dagmm.gmm import GMM
from data_config import *

class Model(nn.Module):
    def __init__(self, comp_hiddens, comp_activation, est_hiddens, est_activation, est_dropout_ratio, x_dim, z_dim, origin_samples=global_origin_samples) -> None:
        super(Model, self).__init__()
        n_comp = est_hiddens[-1]
        self.comp_net = CompressionNet(comp_hiddens, x_dim, comp_activation).to(device=global_device)
        self.est_net = EstimationNet(est_hiddens, z_dim, est_activation, est_dropout_ratio).to(device=global_device)
        self.gmm = GMM(n_comp, origin_samples=origin_samples).to(device=global_device)

    def forward(self, x, is_train=True):
        if is_train:
            z, x_dash, z_c  = self.comp_net.inference(x)
            gamma = self.est_net.inference(z)
            self.gmm.fit(z, gamma)
            energy = self.gmm.energy(z)
            # self.gmm.assign_param()
            return z, x_dash, energy, gamma, z_c
        else:
            z, x_dash, z_c  = self.comp_net.inference(x)
            energy = self.gmm.energy(z)
            return z, x_dash, energy, None, None