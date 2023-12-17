# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from data_config import *

class GMM(nn.Module):
    """ Gaussian Mixture Model (GMM) """
    def __init__(self, n_comp, origin_samples = global_origin_samples):
        super(GMM, self).__init__()
        self.n_comp = n_comp
        self.origin_samples = origin_samples
        self.phi = None
        self.phi_new = None
        self.mu = None
        self.mu_new = None
        self.sigma = None
        self.sigma_new = None
        self.L = None
        self.L_new = None
        self.training = False

    def update_param(self, new_phi, new_mu, new_sigma, new_sample_num, min_vals):
        #print(f"self.phi_new:{self.phi_new}   self.phi:{self.phi}")
        if self.phi_new is None:
        # if True:
            self.phi_new = new_phi
            self.mu_new = new_mu
            self.sigma_new = new_sigma
            # print(f"sigma:{self.sigma}")
            self.L_new = torch.linalg.cholesky(new_sigma + min_vals[None,:,:])
        else:
            alpha = new_sample_num/(new_sample_num+self.origin_samples)
            self.phi_new = alpha * new_phi + (1-alpha) * self.phi
            # self.phi_new = new_phi
            self.mu_new = alpha * new_mu + (1-alpha) * self.mu
            # self.mu_new = new_mu
            
            self.sigma_new = alpha * (torch.einsum('ikl,ikm->klm', new_mu[None, :, :], new_mu[None, :, :]) + new_sigma) +(1-alpha)*(self.sigma+torch.einsum('ikl,ikm->klm', self.mu[None, :, :], self.mu[None, :, :])) - torch.einsum('ikl,ikm->klm', self.mu_new[None, :, :], self.mu_new[None, :, :])
            # print(f"self.sigma:{self.sigma.shape} {torch.mean(self.sigma)} self.mu:{self.mu.shape} {torch.mean(self.mu)}")
            # print(f"sigma:{self.sigma_new}")
            # print(f"mu:{self.mu} {torch.diag(self.mu)}")
            self.L_new = torch.linalg.cholesky(self.sigma_new + min_vals[None,:,:])
            

    def assign_param(self):
        self.phi = self.phi_new.detach()
        self.mu = self.mu_new.detach()
        self.sigma = self.sigma_new.detach()
        self.L = self.L_new.detach()

    def fit(self, z, gamma):
        """ fit data to GMM model

        Parameters
        ----------
        z : tf.Tensor, shape (n_samples, n_features)
            data fitted to GMM.
        gamma : tf.Tensor, shape (n_samples, n_comp)
            probability. each row is correspond to row of z.
        """

        # Calculate mu, sigma
        # i   : index of samples
        # k   : index of components
        # l,m : index of features
        gamma_sum = torch.sum(gamma, dim=0)
        phi = torch.mean(gamma, dim=0)
        # print(f"gmm fit z:{z.shape} {torch.mean(z)} gamma:{gamma.shape} {torch.mean(gamma)} phi:{phi.shape} {torch.mean(phi)} gamma_sum:{gamma_sum.shape}")
        mu = torch.einsum('ik,il->kl', gamma, z) / gamma_sum[:,None]
        # print(f"gamma:{torch.mean(gamma)} z:{torch.mean(z)} mu:{torch.mean(mu)} gamma[:,:,None]:{gamma[:,:,None]}")
        z_centered = torch.sqrt(gamma[:,:,None]) * (z[:,None,:] - mu[None,:,:])
        # print(f"z_centered:{z_centered.shape} {torch.mean(z_centered)}")
        sigma = torch.einsum('ikl,ikm->klm', z_centered, z_centered) / gamma_sum[:,None,None]
        # Calculate a cholesky decomposition of covariance in advance
        n_features = z.shape[1]
        min_vals = torch.diag(torch.ones(n_features, dtype=torch.float64, device=global_device)) * 1e-6
        # print(f"sigma:{sigma.shape}")
        # print(f"min_vals:{min_vals[None,:,:].shape}")
        # print(f"torch.mean(sigma + min_vals[None,:,:]): {torch.mean(sigma + min_vals[None,:,:])} {torch.mean(min_vals[None,:,:])} {torch.mean(sigma)}")
        # self.L = self.L + min_vals[None, :, :]
        # print(f"---min_vals: {min_vals}")
        self.update_param(phi, mu, sigma, gamma.size(0), min_vals)
        self.training = False

        # print(f"\nself.phi:{torch.mean(self.phi)} \nself.mu:{torch.mean(self.mu)} \nself.sigma:{torch.mean(self.sigma)} \nself.L:{torch.mean(self.L)}\n")

    def energy(self, z):
        """ calculate an energy of each row of z

        Parameters
        ----------
        z : tf.Tensor, shape (n_samples, n_features)
            data each row of which is calculated its energy.

        Returns
        -------
        energy : tf.Tensor, shape (n_samples)
            calculated energies
        """

        # if self.training and self.phi is None:
        #     self.phi, self.mu, self.sigma, self.L = self.create_variable(z.shape[1])

        # Instead of inverse covariance matrix, exploit cholesky decomposition
        # for stability of calculation.
        z_centered = z[:,None,:] - self.mu_new[None,:,:]  #ikl
        # print(f"z:{z.shape} self.mu:{self.mu.shape} z_centered:{torch.permute(z_centered, [1, 2, 0]).shape} self.L:{self.L.shape}")
        # v = torch.triangular_solve(self.L, torch.permute(z_centered, [1, 2, 0]))  # kli
        # tf传的是A,b torch传的是b,A
        v = torch.linalg.solve_triangular(self.L_new, torch.permute(z_centered, [1, 2, 0]), upper=True)  # kli
        # print(f"v:{v.shape} {torch.mean(v)}")
        # print(f"self.L:{self.L.shape} {torch.mean(self.L)} {self.L}")
        # print(f"torch.diagonal(self.L):{torch.diagonal(self.L).shape} {torch.mean(torch.diagonal(self.L, dim1=-2, dim2=-1))} {torch.diagonal(self.L, dim1=-2, dim2=-1)}")
        # print(f"torch.log(torch.diagonal(self.L)):{torch.log(torch.diagonal(self.L, dim1=-2, dim2=-1)).shape} {torch.log(torch.diagonal(self.L, dim1=-2, dim2=-1))}")

        # log(det(Sigma)) = 2 * sum[log(diag(L))]
        # print(f"self.L:{self.L.shape}")
        log_det_sigma = 2.0 * torch.sum(torch.log(torch.diagonal(self.L_new, dim1=-2, dim2=-1)), dim=1)
        # print(f"log_det_sigma:{log_det_sigma.shape} {torch.mean(log_det_sigma)}")
        # To calculate energies, use "log-sum-exp" (different from orginal paper)
        d = z.shape[1]
        logits = torch.log(self.phi_new[:,None]) - 0.5 * (torch.sum(torch.square(v), dim=1)
            + d * torch.log(2.0 * torch.tensor(torch.pi)) + log_det_sigma[:,None])
        energies = -torch.logsumexp(logits, dim=0)

        return energies

    def cov_diag_loss(self):
        # print(f"self.sigma:{self.sigma.shape}")
        return torch.sum(torch.divide(1, torch.diagonal(self.sigma_new, dim1=-2, dim2=-1)))
