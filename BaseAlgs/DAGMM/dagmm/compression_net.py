import torch
import torch.nn as nn
from data_config import *

class CompressionNet(nn.Module):
    def __init__(self, hidden_layer_sizes: list, x_dim, activation=nn.Tanh()):
        """
        Parameters
        ----------
        hidden_layer_sizes : list of int
            list of the size of hidden layers.
            For example, if the sizes are [n1, n2],
            the sizes of created networks are:
            input_size -> n1 -> n2 -> n1 -> input_sizes
            (network outputs the representation of "n2" layer)
        activation : function
            activation function of hidden layer.
            the last layer uses linear function.
        """
        super(CompressionNet, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.x_dim = x_dim
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        cur_size = self.x_dim
        # print(f"CompressionNet: {cur_size}+{self.hidden_layer_sizes}")
        for index, size in enumerate(self.hidden_layer_sizes):
            self.encoder.add_module(name=f"{index}linear", module=nn.Linear(cur_size, size, dtype=torch.float64))
            if index < len(self.hidden_layer_sizes)-1:
                self.encoder.add_module(name=f"{index}act", module=self.activation)
            # self.encoder.add_module(name=f"{index}bn", module=nn.BatchNorm1d(num_features=size, dtype=torch.float64))
            cur_size = size

        
        self.z_dim = self.hidden_layer_sizes[-1]
        cur_size = self.z_dim
        # print(f"CompressionNet: {cur_size}+{self.hidden_layer_sizes[::-1][1:]+[self.x_dim]}")
        for index, size in enumerate(self.hidden_layer_sizes[::-1][1:]+[self.x_dim]):
            self.decoder.add_module(name=f"{index}linear", module=nn.Linear(cur_size, size, dtype=torch.float64))
            if index < len(self.hidden_layer_sizes)-1:
                self.decoder.add_module(name=f"{index}act", module=self.activation)       
            # self.decoder.add_module(name=f"{index}bn", module=nn.BatchNorm1d(num_features=size, dtype=torch.float64))
            cur_size = size


    def compress(self, x: torch.Tensor):
        return self.encoder(x)
    
    def reverse(self, z: torch.Tensor):
        return self.decoder(z)

    def loss(self, x, x_dash):
        def euclid_norm(x):
            return torch.sqrt(torch.sum(torch.square(x), dim=1))
        def oushi_dist(x1, x2):
            return torch.sum(torch.square(x1-x2), dim=-1)
        def qiebixuefu_dist(x1, x2):
            max_data, _ = torch.max(torch.square(x1-x2), dim=-1)
            return max_data
        def manhadun_dist(x1, x2):
            return torch.sum(torch.abs(x1-x2), dim=1)

        # Calculate Euclid norm, distance
        norm_x = euclid_norm(x)
        norm_x_dash = euclid_norm(x_dash)
        dist_x = euclid_norm(x - x_dash)
        dot_x = torch.sum(x * x_dash, dim=1)

        # Based on the original paper, features of reconstraction error
        # are composed of these loss functions:
        #  1. loss_E : relative Euclidean distance
        #  2. loss_C : cosine similarity
        min_val = 1e-3
        # loss_E_norm = dist_x  / (norm_x + min_val)
        loss_E_abs = oushi_dist(x, x_dash)
        loss_qie = qiebixuefu_dist(x, x_dash)
        loss_man = manhadun_dist(x, x_dash)
        # loss_C = 1.0 - dot_x / (norm_x * norm_x_dash + min_val)
        # print(f"loss_E_norm:{loss_E_norm.shape} loss_E_abs:{loss_E_abs.shape} loss_qie:{loss_qie.shape} loss_man:{loss_man.shape}")
        # return torch.concat([loss_E_norm[:,None], loss_E_abs[:,None], loss_qie[:,None], loss_man[:,None], loss_C[:,None]], dim=1)
        return torch.concat([loss_E_abs[:,None], loss_qie[:,None], loss_man[:,None]], dim=1)

    def extract_feature(self, x, x_dash, z_c):
        z_r = self.loss(x, x_dash)
        # print(f"extract_feature z_r:{torch.mean(z_r[:, 0])} {torch.mean(z_r[:, 1])}")
        # print(f"extract_feature: z_c:{z_c.shape} z_r{z_r.shape} x:{x.shape} x_dash:{x_dash.shape}")
        return torch.concat([z_c, z_r], dim=1)

    def inference(self, x):
        """ convert input to output tensor, which is composed of
        low-dimensional representation and reconstruction error.

        Parameters
        ----------
        x : tf.Tensor shape : (n_samples, n_features)
            Input data

        Results
        -------
        z : tf.Tensor shape : (n_samples, n2 + 2)
            Result data
            Second dimension of this data is equal to
            sum of compressed representation size and
            number of loss function (=2)

        x_dash : tf.Tensor shape : (n_samples, n_features)
            Reconstructed data for calculation of
            reconstruction error.
        """

        z_c = self.compress(x)
        x_dash = self.reverse(z_c)

        # compose feature vector
        z = self.extract_feature(x, x_dash, z_c)

        return z, x_dash, z_c


    def reconstruction_error(self, x, x_dash):
        # index_loss_weight_tensor = torch.tensor(index_loss_weight).to(device=global_device)
        # torch.save(x, "x.npy")
        # torch.save(x_dash, "x_dash.npy")
        # print("data saved!")
        t = torch.sum(torch.square(x - x_dash), dim=-1)
        return torch.mean(t)
        # return torch.mean(torch.sum(torch.square(x - x_dash), dim=1), dim=0)