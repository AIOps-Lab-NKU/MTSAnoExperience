# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class EstimationNet(nn.Module):
    """ Estimation Network

    This network converts input feature vector to softmax probability.
    Bacause loss function for this network is not defined,
    it should be implemented outside of this class.
    """
    def __init__(self, hidden_layer_sizes, z_dim, activation=nn.ReLU(), dropout_ratio=None):
        """
        Parameters
        ----------
        hidden_layer_sizes : list of int
            list of sizes of hidden layers.
            For example, if the sizes are [n1, n2],
            layer sizes of the network are:
            input_size -> n1 -> n2
            (network outputs the softmax probabilities of "n2" layer)
        activation : function
            activation function of hidden layer.
            the funtcion of last layer is softmax function.
        """
        super(EstimationNet, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.z_dim = z_dim
        self.model = nn.Sequential()
        cur_size = self.z_dim
        # print(f"EstimationNet {cur_size}+{self.hidden_layer_sizes}")
        for index, size in enumerate(self.hidden_layer_sizes):
            self.model.add_module(name=f"{index}linear", module=nn.Linear(cur_size, size, dtype=torch.float64))
            if index < len(self.hidden_layer_sizes)-1:
                self.model.add_module(name=f"{index}act", module=self.activation)
                if dropout_ratio is not None:
                    self.model.add_module(name=f"{index}dropout", module=nn.Dropout(p=dropout_ratio))
            # self.model.add_module(name=f"{index}bn", module=nn.BatchNorm1d(num_features=size, dtype=torch.float64))
            cur_size = size

        self.model.add_module(name=f"softmax", module=nn.Softmax(dim=-1))


    def inference(self, z):
        """ Output softmax probabilities

        Parameters
        ----------
        z : tf.Tensor shape : (n_samples, n_features)
            Data inferenced by this network
        dropout_ratio : tf.Tensor shape : 0-dimension float (optional)
            Specify dropout ratio
            (if None, dropout is not applied)

        Results
        -------
        probs : tf.Tensor shape : (n_samples, n_classes)
            Calculated probabilities
        """
        # print(f"EstimationNet inference:z{z.shape}")
        return self.model(z)
