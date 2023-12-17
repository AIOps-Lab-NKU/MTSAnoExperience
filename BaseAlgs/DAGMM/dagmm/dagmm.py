import pathlib
import torch
import torch.nn as nn

import numpy as np
from dagmm.compression_net import CompressionNet
from dagmm.estimation_net import EstimationNet
from dagmm.gmm import GMM
from dagmm.model import Model
from data_config import *

def freeze_a_layer(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = False
def unfreeze_a_layer(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = True

class DAGMM():
    """ Deep Autoencoding Gaussian Mixture Model.
    This implementation is based on the paper:
    Bo Zong+ (2018) Deep Autoencoding Gaussian Mixture Model
    for Unsupervised Anomaly Detection, ICLR 2018
    (this is UNOFFICIAL implementation)
    """

    MODEL_FILENAME = "DAGMM_model"
    SCALER_FILENAME = "DAGMM_scaler"

    def __init__(self, comp_hiddens, comp_activation,
            est_hiddens, est_activation, est_dropout_ratio,
            minibatch_size, epoch_size, x_dim, z_dim,
            learning_rate=0.0001, lambda1=0.1, lambda2=0.005, origin_samples=global_origin_samples
            ):
        self.model = Model(comp_hiddens, comp_activation, est_hiddens, est_activation, est_dropout_ratio, x_dim, z_dim, origin_samples=origin_samples)
        # print(f"model:{self.model}")
        self.minibatch_size = minibatch_size
        self.epoch_size = epoch_size
        self.learning_rate = learning_rate
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.valid_loss = []
        # self.init_param()
    
    def init_param(self,name):
        if name == 'vae_1':
            self.model.comp_net.encoder[0].reset_parameters()
            self.model.comp_net.decoder[4].reset_parameters()
        elif name == 'vae_12':
            self.model.comp_net.encoder[0].reset_parameters()
            self.model.comp_net.decoder[4].reset_parameters()
            self.model.comp_net.encoder[2].reset_parameters()
            self.model.comp_net.decoder[2].reset_parameters()
        elif name == 'vae_123':
            self.model.comp_net.encoder[0].reset_parameters()
            self.model.comp_net.decoder[0].reset_parameters()
            self.model.comp_net.encoder[2].reset_parameters()
            self.model.comp_net.decoder[2].reset_parameters()
            self.model.comp_net.encoder[4].reset_parameters()
            self.model.comp_net.decoder[4].reset_parameters()
        elif name == 'vae_23':
            self.model.comp_net.encoder[2].reset_parameters()
            self.model.comp_net.decoder[2].reset_parameters()
            self.model.comp_net.encoder[4].reset_parameters()
            self.model.comp_net.decoder[0].reset_parameters()

    def freeze_layers(self, name):
        # encoder 0, 2, 4
        # decoder 4, 2, 0
        if name == 'vae_1':
            freeze_a_layer(self.model.comp_net.encoder[0])
            freeze_a_layer(self.model.comp_net.decoder[4])
        elif name == 'vae_12':
            freeze_a_layer(self.model.comp_net.encoder[0])
            freeze_a_layer(self.model.comp_net.encoder[2])
            freeze_a_layer(self.model.comp_net.decoder[4])
            freeze_a_layer(self.model.comp_net.decoder[2])
        elif name == 'vae_123':
            freeze_a_layer(self.model.comp_net.encoder[0])
            freeze_a_layer(self.model.comp_net.encoder[2])
            freeze_a_layer(self.model.comp_net.encoder[4])
            freeze_a_layer(self.model.comp_net.decoder[4])                        
            freeze_a_layer(self.model.comp_net.decoder[2])                        
            freeze_a_layer(self.model.comp_net.decoder[0])
        elif name == 'est':
            # 0, 3
            freeze_a_layer(self.model.est_net.model[0])
            freeze_a_layer(self.model.est_net.model[3])

    def unfreeze_layers(self):
        unfreeze_a_layer(self.model.comp_net.encoder[0])
        unfreeze_a_layer(self.model.comp_net.encoder[2])
        unfreeze_a_layer(self.model.comp_net.encoder[4])
        unfreeze_a_layer(self.model.comp_net.decoder[4])                        
        unfreeze_a_layer(self.model.comp_net.decoder[2])                        
        unfreeze_a_layer(self.model.comp_net.decoder[0])
        unfreeze_a_layer(self.model.est_net.model[0])
        unfreeze_a_layer(self.model.est_net.model[3])

    def get_data(self, idx, valid_portition):
        if len(idx) < 200:
            train_idx = idx
            valid_idx = idx
        else:
            valid_idx = [idx[i] for i in range(0, len(idx), int(1//valid_portition))]
            train_idx = [i for i in idx if i not in valid_idx]
        return train_idx, valid_idx

    def valid_log(self, valid_idx, x, log_file, save_dir, epoch):
        valid_data = torch.tensor(x[valid_idx], dtype=torch.float64).to(global_device)
        # try:
        with torch.no_grad():
            z, x_dash, energy, gamma, z_c = self.model(valid_data)
            # Loss function
            recon_loss = self.model.comp_net.reconstruction_error(valid_data, x_dash)
            enrgy_loss = torch.mean(energy)
            diag_loss = self.model.gmm.cov_diag_loss()
            loss = recon_loss + self.lambda1 * enrgy_loss + self.lambda2 * diag_loss
            loss = loss.item() / valid_data.shape[0]

        if loss < self.best_loss:
            self.best_loss = loss
            self.save(save_dir=save_dir)
            print(f"***", end=" ")
            print(f"***", end=" ", file=log_file)
        print(f"epoch:{epoch} loss:{loss} recon_loss:{recon_loss.item()} enrgy_loss:{enrgy_loss.item()} diag_loss:{diag_loss.item()} z:{torch.mean(z)} z_c:{torch.mean(z_c)} x_dash:{torch.mean(x_dash)}")
        print(f"epoch:{epoch} loss:{loss} recon_loss:{recon_loss.item()} enrgy_loss:{enrgy_loss.item()} diag_loss:{diag_loss.item()} z:{torch.mean(z)} z_c:{torch.mean(z_c)} x_dash:{torch.mean(x_dash)}", file=log_file)
        del z, x_dash, energy, loss


    def fit(self, x, save_dir: pathlib.Path, valid_portition=0.3):
        print(f"x mean:{np.mean(x)} x std:{np.std(x)}\n")

        """ Fit the DAGMM model according to the given data.
        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Training data.
        """
        n_samples, n_features = x.shape
        # Number of batch
        # global_valid_epoch_freq = self.epoch_size // 20
        # global_valid_epoch_freq = 1
        # Training
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        train_idx, valid_idx = self.get_data(idx=idx, valid_portition=valid_portition)
        n_batch = (len(train_idx) - 1) // self.minibatch_size + 1
        self.step = 0
        log_path = save_dir / 'log.txt'
        if log_path.exists():
            os.remove(log_path)
        log_file = open(log_path, mode='a')
        self.best_loss = 99999999
        if_break = False
        if "finetune" in train_type or "freeze" in train_type:
            self.valid_log(valid_idx, x, log_file, save_dir, 0)
        for epoch in range(1, self.epoch_size+1):
            # print(f"epoch:{epoch}")
            for batch in range(n_batch):
                with torch.autograd.set_detect_anomaly(True):
                    self.step += 1
                    i_start = batch * self.minibatch_size
                    i_end = (batch + 1) * self.minibatch_size
                    x_batch = x[train_idx[i_start:i_end]]
                    x_batch = torch.tensor(x_batch, dtype=torch.float64).to(global_device)
                    try:
                        z, x_dash, energy, gamma, z_c = self.model(x_batch)
                        # Loss function
                        recon_loss = self.model.comp_net.reconstruction_error(x_batch, x_dash)
                        enrgy_loss = torch.mean(energy)
                        diag_loss = self.model.gmm.cov_diag_loss()
                        loss = recon_loss + self.lambda1 * enrgy_loss + self.lambda2 * diag_loss
                        self.optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
                        self.optimizer.step()
                        self.model.gmm.assign_param()
                    except Exception as e:
                        print(f"训练报错:{e}")
                        print(f"训练报错:{e}", file=log_file)
                        if_break = True
                        break

            if epoch % global_valid_epoch_freq == 0:
                self.valid_log(valid_idx, x, log_file, save_dir, epoch)
                
            del z, x_dash, energy, loss
            torch.cuda.empty_cache()
            if if_break:
                break
        self.valid_log(valid_idx, x, log_file, save_dir, epoch) 
        
    def predict(self, x, restore_dir: pathlib.Path):
        """ Calculate anormaly scores (sample energy) on samples in X.
        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Data for which anomaly scores are calculated.
            n_features must be equal to n_features of the fitted data.
        Returns
        -------
        energies : array-like, shape (n_samples)
            Calculated sample energies.
        """
        n_samples, n_features = x.shape
        # Number of batch
        n_batch = (n_samples - 1) // self.minibatch_size + 1  
        # Training
        idx = np.arange(x.shape[0])
        # self.restore(restore_dir)
        score = None
        recon = None
        z_res = None
        import time
        total_time = 0
        fw=open("/home/zhangshenglin/chenshiqi/DAGMM/0520/time.txt",'a')
        with torch.no_grad():
            for batch in range(n_batch):
                
                i_start = batch * self.minibatch_size
                i_end = (batch + 1) * self.minibatch_size
                x_batch = x[idx[i_start:i_end]]
                x_batch = torch.tensor(x_batch, dtype=torch.float64).to(global_device)
                start_time = time.time()
                z, x_dash, energy, gamma, z_c = self.model(x_batch, is_train=False)
                end_time = time.time()
                total_time+=end_time-start_time
                if score is None:
                    score = energy.cpu().numpy()
                    recon = x_dash.cpu().numpy()
                    z_res = z.cpu().numpy()
                else:
                    score = np.concatenate([score, energy.cpu().numpy()], axis=0)
                    recon = np.concatenate([recon, x_dash.cpu().numpy()], axis=0)
                    z_res = np.concatenate([z_res, z.cpu().numpy()], axis=0)
        fw.write(f'\n{dataset_type}:{total_time}\n')       
        fw.close()
        print(f"score:{score.shape} recon:{recon.shape} z_res:{z_res.shape},test_time:{total_time}")
        return score, recon, z_res

    def save(self, save_dir: pathlib.Path):
        """ Save trained model to designated directory.
        This method have to be called after training.
        (If not, throw an exception)
        Parameters
        ----------
        fdir : str
            Path of directory trained model is saved.
            If not exists, it is created automatically.
        """
        model_path = save_dir / 'model.pth'
        param_path = save_dir / 'param.pth'
        torch.save(self.model.state_dict(), model_path)
        torch.save({
            'phi': self.model.gmm.phi.detach().cpu(),
            'mu': self.model.gmm.mu.detach().cpu(),
            'sigma': self.model.gmm.sigma.detach().cpu(),
            'L': self.model.gmm.L.detach().cpu(),
        }, param_path)

    def restore(self, save_dir: pathlib.Path):
        """ Restore trained model from designated directory.
        Parameters
        ----------
        fdir : str
            Path of directory trained model is saved.
        """
        model_path = save_dir / 'model.pth'
        param_path = save_dir / 'param.pth'
        self.model.load_state_dict(torch.load(model_path))
        gmm_param_dict = torch.load(param_path)
        self.model.gmm.phi = gmm_param_dict['phi'].to(device=global_device)
        self.model.gmm.phi_new = gmm_param_dict['phi'].to(device=global_device)
        self.model.gmm.mu = gmm_param_dict['mu'].to(device=global_device)
        self.model.gmm.mu_new = gmm_param_dict['mu'].to(device=global_device)
        self.model.gmm.sigma = gmm_param_dict['sigma'].to(device=global_device)
        self.model.gmm.sigma_new = gmm_param_dict['sigma'].to(device=global_device)
        self.model.gmm.L = gmm_param_dict['L'].to(device=global_device)
        self.model.gmm.L_new = gmm_param_dict['L'].to(device=global_device)