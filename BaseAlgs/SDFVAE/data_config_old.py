from cgi import print_environ
import pathlib
import json
import numpy as np
import os
import argparse
import torch
# global_device = torch.device('cpu')
project_path = pathlib.Path(os.path.abspath(__file__)).parent

parser = argparse.ArgumentParser()
# GPU option
parser.add_argument('--gpu_id', type=int, default=0)
# dataset
parser.add_argument('--dataset_type', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--batch_size', type=int, default=250)
parser.add_argument('--base_model_dir', type=str, default=None)

# model
parser.add_argument('--T', type=int, default=5)
parser.add_argument('--s_dim', type=int, default=8)
parser.add_argument('--d_dim', type=int, default=10)
parser.add_argument('--model_dim', type=int, default=100)
# training
parser.add_argument('--epochs', type=int, default=50)
# parser.add_argument('--index_weight', type=int, default= 10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=None)  # 409，2021，2022
parser.add_argument('--train_type', type=str)
# parser.add_argument('--training_period', type=int, default=None) 
parser.add_argument('--valid_epoch', type=int, default=5) 

parser.add_argument('--min_std', type=float, default=-3) 
parser.add_argument('--global_window_size', type=int, default=60) 
# parser.add_argument('--dataset_path',type=str)
# parser.add_argument('--train_num',type=int,default=5)
# parser.add_argument('--index_weight_index',type=int,default=1)

args = parser.parse_args()



# train_num = args.train_num
min_std = args.min_std
# dataset_path = args.dataset_path
# index_weight_index = args.index_weight_index

single_score_th = 10000
out_dir = args.out_dir
# 注意202只有1块gpu，只能设置为0
GPU_index = str(args.gpu_id)
global_device = torch.device(f'cuda:{GPU_index}')
# print(f"global_device:{global_device}")
dataset_type = args.dataset_type
global_epochs = args.epochs
seed = args.seed
# training_period = args.training_period
global_valid_epoch = args.valid_epoch

# SDFVAE参数设置
global_s_dim = args.s_dim
global_d_dim = args.d_dim
global_conv_dim = args.model_dim
global_hidden_dim = args.model_dim
global_T = args.T

global_batch_size= args.batch_size
# learning rate
global_learning_rate = args.lr
# circle_loss_weight = args.index_weight

if_freeze_seq = True
train_type = args.train_type
global_window_size=args.global_window_size

exp_key = train_type
# exp_key += f"_{train_num}nodes"
# exp_key += f"_{index_weight_index}iwi"
exp_key += f"_{min_std}clip"
exp_key += f"_{seed}"
# exp_key +=f"_{training_period}daytrain"
exp_key += f"_model{args.model_dim}"
exp_key += f"_s{global_s_dim}"
exp_key += f"_d{global_d_dim}"
exp_key += f"_T{global_T}"
exp_key += f"_lr{global_learning_rate}"
# exp_key += f"_weight{circle_loss_weight}"
exp_key += f"_epoch{global_epochs}"
exp_key += f"_windows{60}"


exp_dir = project_path /out_dir/ dataset_type / exp_key
min_log_sigma = min_std

base_model_dir = args.base_model_dir

# 学习率衰减
learning_rate_decay_by_step = 100
learning_rate_decay_factor = 1
global_valid_step_freq = None

# dataset_root = pathlib.Path(f"../data/{dataset_type}")
dataset_root = pathlib.Path(f"/home/zhangshenglin/chenshiqi/Dataset/{dataset_type}")
train_data_json = dataset_root / f"{dataset_type}-train.json"
test_data_json = dataset_root / f"{dataset_type}-test.json"


def load_data_from_json(s):
    json_file = json.load(open(s))
    data = json_file['data']
    label = json_file['label']
    return data,label

# 实验参数
# day_points=96
# 学习率衰减
g_min_lr = 1e-4
learning_rate_decay_by_step = 5
learning_rate_decay_factor = 1

global_window_size = 60

bf_search_min = 0
bf_search_max = 1000
bf_search_step_size = 5
# eval_item_length = 96*7 - (global_window_size - 1)
# noshare_save_dir = project_path / base_model_dir


train_data,_ = load_data_from_json(train_data_json)
test_data,label = load_data_from_json(test_data_json)
# feature_dim = 22


# if dataset_type == 'soil-moisture-active-passive':
    # chosed_index = [5,7,9,10,11,12,15,19,21,25,30,31,35,37,38,41,44,46,49,51]
    # chosed_index = [30,31,35,37,40,44,49,51]

#yidong
if dataset_type=='yidong-22':
    chosed_index = [9]
    # # chosed_index = [1, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 23, 24, 25, 26, 27, 34, 36, 38, 42, 44, 45, 46, 48, 50, 51, 53, 56, 57, 60, 61, 62, 64, 65, 69, 71, 73, 74, 75, 77, 82, 83, 84, 89, 92, 95, 98, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 117, 118, 119, 120, 123, 126, 128, 129, 130, 132, 134, 135, 138, 139, 140, 141, 142, 145, 146, 147, 150, 154, 156, 157, 158, 160, 162, 163, 165, 167, 168, 169, 171, 173, 174, 175, 177, 178, 180, 186, 187, 190, 191, 192, 193, 197, 200]
    # chosed_index = [1, 3, 6, 8, 13, 14, 15, 18, 21, 24, 25, 26, 27, 29, 30, 31, 32, 36, 49, 50, 51, 58, 59, 60, 68, 70, 71, 72, 73, 75, 76, 77, 78, 79, 80, 83, 88, 89, 90, 94, 97, 101, 102, 103, 105]
    # train_data = [train_data[i] for i in range(len(train_data)) if sum(label[i])!=0]
    # test_data = [test_data[i] for i in range(len(test_data)) if sum(label[i])!=0]
    # label = [label[i] for i in range(len(label)) if sum(label[i])!=0]
#smd
if dataset_type == 'server-machine-dataset':
    chosed_index = [1]
#smap
if dataset_type == 'soil-moisture-active-passive':
    chosed_index = [19,31,35,38,51]
# msl
if dataset_type == 'mars-science-laboratory':
    chosed_index = [26]
# asd
if dataset_type == 'application-server-dataset':
    chosed_index = [5]
# ctf
if dataset_type == 'CTF_OmniClusterSelected_th48_26cluster':
    chosed_index = [5,11,12,13,16,17,19,23,25,26]

# SWaT
if dataset_type == 'secure-water-treatment':
    chosed_index = [1]
# WADI
if dataset_type == 'water-distribution':
    chosed_index = [1]


# 统一的数据预处理方式
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_minmax(df_train, df_test):
    """
    normalize raw data
    """
    print('minmax', end=' ')
    df_train = np.asarray(df_train, dtype=np.float32)
    df_test = np.asarray(df_test, dtype=np.float32)
    if len(df_train.shape) == 1 or len(df_test.shape) == 1:
        raise ValueError('Data must be a 2-D array')
    if np.any(sum(np.isnan(df_train)) != 0):
        print('train data contains null values. Will be replaced with 0')
        df_train = np.nan_to_num()
    if np.any(sum(np.isnan(df_test)) != 0):
        print('test data contains null values. Will be replaced with 0')
        df_test = np.nan_to_num()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(df_train)
    df_train = scaler.transform(df_train)
    df_test = scaler.transform(df_test)
    return df_train, df_test


def preprocess_meanstd(df_train, df_test):
    """returns normalized and standardized data.
    """
    # print('meanstd', end=' ')
    df_train = np.asarray(df_train, dtype=np.float32)

    if len(df_train.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df_train)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df_train = np.nan_to_num(df_train)

    k = 5
    
    e = 1e-3
    mean_array = np.mean(df_train, axis=0, keepdims=True)
    std_array = np.std(df_train, axis=0, keepdims=True)
    std_array[np.where(std_array==0)] = e
    df_train = np.where(df_train > mean_array + k * std_array, mean_array + k * std_array, df_train)
    df_train = np.where(df_train < mean_array - k * std_array, mean_array - k * std_array, df_train)
    
    train_mean_array = np.mean(df_train, axis=0, keepdims=True)
    train_std_array = np.std(df_train, axis=0, keepdims=True)
    train_std_array[np.where(train_std_array==0)] = e
    
    df_train_new = (df_train - train_mean_array) / train_std_array
    
    df_test = np.where(df_test > train_mean_array + k * train_std_array, train_mean_array + k * train_std_array, df_test)
    df_test = np.where(df_test < train_mean_array - k * train_std_array, train_mean_array - k * train_std_array, df_test)
    df_test_new = (df_test - train_mean_array) / train_std_array

    return df_train_new, df_test_new

def preprocess(df_train, df_test):
    return preprocess_meanstd(df_train, df_test)
    # return preprocess_minmax(df_train, df_test)

def global_loss_fn(original_seq, recon_seq_mu, recon_seq_logsigma, s_mean,
                s_logvar, d_post_mean, d_post_logvar, d_prior_mean, d_prior_logvar,cluster_id):
        batch_size = original_seq.size(0)

        # print(f"original_seq.shape:{original_seq.shape} recon_seq_mu:{recon_seq_mu.shape}")
        loglikelihood = -0.5 * torch.sum(
            (torch.pow(((original_seq.float() - recon_seq_mu.float()) / torch.exp(recon_seq_logsigma.float())), 2) + 
            2 * recon_seq_logsigma.float()+ np.log(np.pi * 2))
            )
        # print(f"平方项:{(torch.pow(((original_seq.float() - recon_seq_mu.float()) / torch.exp(recon_seq_logsigma.float())), 2)+ 2 * recon_seq_logsigma.float()+ np.log(np.pi * 2)).shape}")
        # See https://arxiv.org/pdf/1606.05908.pdf, Page 9, Section 2.2, Equation (7) for details.
        kld_s = -0.5 * torch.sum(1 + s_logvar - torch.pow(s_mean, 2) - torch.exp(s_logvar))
        # See https://arxiv.org/pdf/1606.05908.pdf, Page 9, Section 2.2, Equation (6) for details.
        d_post_var = torch.exp(d_post_logvar)
        d_prior_var = torch.exp(d_prior_logvar)
        kld_d = 0.5 * torch.sum(d_prior_logvar - d_post_logvar
                                + ((d_post_var + torch.pow(d_post_mean - d_prior_mean, 2)) / d_prior_var)
                                - 1)
        # loss, llh, kld_s, kld_d
        return (-loglikelihood + kld_s + kld_d) / batch_size, -loglikelihood / batch_size, kld_s / batch_size, kld_d / batch_size