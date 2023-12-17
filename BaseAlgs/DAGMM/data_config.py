import pathlib
import json
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
project_path = pathlib.Path(os.path.abspath(__file__)).parent
parser = argparse.ArgumentParser()
# GPU option
parser.add_argument('--gpu_id', type=int, default=1)
# dataset
parser.add_argument('--dataset_type', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--base_model_dir', type=str, default=None)

# model
parser.add_argument('--z_dim', type=int, default=5)
parser.add_argument('--origin_samples', type=int, default=None) # 与gmm的参数更新有关，是否与batch_size对应

# training
parser.add_argument('--epochs', type=int, default= 200)
# parser.add_argument('--index_weight', type=int, default= 10)
parser.add_argument('--lr', type=float, default= 5e-3)
parser.add_argument('--dropout_r', type=float, default=0.1) # est_dropout_ratio 减少隐藏层过拟合
parser.add_argument('--seed', type=int, default=None)  # 409，2021，2022
parser.add_argument('--train_type', type=str)
parser.add_argument('--freeze_index_list_encoder', type=int, nargs='+') # 找不到
parser.add_argument('--freeze_index_list_decoder', type=int, nargs='+')
# parser.add_argument('--training_period', type=int, default=None) 
# parser.add_argument('--dataset_path',type=str)
# parser.add_argument('--train_num',type=int,default=5)
parser.add_argument('--index_weight_index',type=int,default=1) # 找不到
parser.add_argument('--valid_epoch',type=int,default=1) # 控制日志打印
args = parser.parse_args()

single_score_th = 10000
out_dir = args.out_dir

GPU_index = str(args.gpu_id)
global_device = torch.device(f'cuda:{GPU_index}')
global_origin_samples = args.origin_samples
dataset_type = args.dataset_type
global_epochs = args.epochs
seed = args.seed

global_valid_epoch_freq = args.valid_epoch

dropout_r = args.dropout_r
global_batch_size= args.batch_size
# learning rate
global_learning_rate = args.lr
# circle_loss_weight = args.index_weight

train_type = args.train_type
# dataset_path = args.dataset_path

exp_key = train_type
exp_key += f"_{seed}"

# exp_key += f"_z{global_z_dim}"
exp_key += f"_epoch{global_epochs}"
exp_key += f"_lr{global_learning_rate}"
exp_key += f"_dropout_r{dropout_r}"
# exp_key += f"_weight{circle_loss_weight}"
exp_key += f"_originsample{global_origin_samples}"
# exp_dir = project_path / out_dir / dataset_type / exp_key
exp_dir = pathlib.Path(out_dir) / dataset_type / exp_key

comp_hidden = [15, 9, 2]
est_hidden = [16, 12, 2+3]
# est_hidden = [10, 2+3]
# comp_hidden = [15, 11, 8]
# est_hidden = [16, 8+5]
# comp_hidden = [11, 5, 2]
# est_hidden = [10, 7, 2+2]
global_z_dim= comp_hidden[-1]
# act_func = nn.Tanh()
act_func = nn.ReLU()
# act_func = nn.Sigmoid()


# if 'initr' in exp_key:
#     global_learning_rate = 2e-3
base_model_dir = args.base_model_dir


learning_rate_decay_by_epoch = 10
# learning_rate_decay_factor = 0.75
learning_rate_decay_factor = 1
global_valid_step_freq = None

dataset_root = pathlib.Path(f"../data/{dataset_type}")
dataset_root = pathlib.Path(f"/home/zhangshenglin/chenshiqi/Dataset/{dataset_type}")

train_data_json = dataset_root / f"{dataset_type}-train.json"
test_data_json = dataset_root / f"{dataset_type}-test.json"

def load_data_from_json(s):
    json_file = json.load(open(s))
    data = json_file['data']
    label = json_file['label']
    return data,label

train_data,_ = load_data_from_json(train_data_json)
test_data,label = load_data_from_json(test_data_json)

#yidong
if dataset_type=='yidong-22':
    # chosed_index = [9]
    chosed_index = [102]
    train_data = [train_data[i] for i in range(len(train_data)) if sum(label[i])!=0]
    test_data = [test_data[i] for i in range(len(test_data)) if sum(label[i])!=0]
    label = [label[i] for i in range(len(label)) if sum(label[i])!=0]
#smd
if dataset_type == 'server-machine-dataset':
    chosed_index = [1]
#smap
if dataset_type == 'soil-moisture-active-passive':
    chosed_index = [1]
# msl
if dataset_type == 'mars-science-laboratory':
    chosed_index = [1]
# asd
if dataset_type == 'application-server-dataset':
    chosed_index = [5]
# ctf
if dataset_type == 'CTF_OmniClusterSelected_th48_26cluster':
    chosed_index = [2]

# SWaT
if dataset_type == 'secure-water-treatment':
    chosed_index = [1]
# WADI
if dataset_type == 'water-distribution':
    chosed_index = [1]

global_window_size = 15
# feature_dim = 25
# label = np.load(label_path)
bf_search_min = 0
bf_search_max = 1000
bf_search_step_size = 1
# eval_item_length = 96*7 - (global_window_size - 1)
noshare_save_dir = project_path / base_model_dir


from sklearn.preprocessing import MinMaxScaler

k = 5
def preprocess_meanstd(df_train, df_test):
    """returns normalized and standardized data.
    """

    df_train = np.asarray(df_train, dtype=np.float32)

    if len(df_train.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df_train)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df_train = np.nan_to_num(df_train)

    # normalize data
    # df = MinMaxScaler().fit_transform(df)
    mean_array = np.mean(df_train, axis=0, keepdims=True)
    std_array = np.std(df_train, axis=0, keepdims=True)
    df_train = np.where(df_train > mean_array + k * std_array, mean_array + k * std_array, df_train)
    df_train = np.where(df_train < mean_array - k * std_array, mean_array - k * std_array, df_train)
    df_train_new = (df_train - np.mean(df_train, axis=0, keepdims=True)) / (np.std(df_train, axis=0, keepdims=True) + 1e-3)
    
    df_test = np.where(df_test > mean_array + k * std_array, mean_array + k * std_array, df_test)
    df_test = np.where(df_test < mean_array - k * std_array, mean_array - k * std_array, df_test)
    df_test_new = (df_test - np.mean(df_train, axis=0, keepdims=True)) / (np.std(df_train, axis=0, keepdims=True) + 1e-3)

    return df_train_new, df_test_new

def preprocess_minmax(df_train, df_test):
    scaler = MinMaxScaler().fit(df_train)
    train = scaler.transform(df_train)
    test = scaler.transform(df_test)
    test = np.clip(test, a_min=-3.0, a_max=3.0)
    return train, test

