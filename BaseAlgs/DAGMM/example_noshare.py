import os
import numpy as np
import torch
import torch.nn as nn
from dagmm.dagmm import DAGMM
import time, random
from data_config import *
# os.environ['CUDA_VISIBLE_DEVICES'] = GPU_index
# torch.cuda.set_device(int(GPU_index))

class Config:
    valid_step_freq = global_valid_step_freq
    exp_dir = exp_dir
    save_dir = exp_dir/ 'model'
    result_dir = exp_dir/'result'

def torch_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def end_to_end(i, config, train_data_item, test_data_item):
    feature_dim = train_data_item.shape[1]
    total_train_time = 0
    model = DAGMM(
        comp_hiddens=comp_hidden, comp_activation=act_func,
        est_hiddens=est_hidden, est_activation=act_func, est_dropout_ratio=dropout_r,
        minibatch_size=global_batch_size, epoch_size=global_epochs, z_dim=est_hidden[-1],
        x_dim=feature_dim, learning_rate=global_learning_rate
    )
    # get data
    x_train, x_test = preprocess_meanstd(train_data_item, test_data_item)
    
    # train
    print(f'-------training for cluster {i}---------')
    x_train_list = []
    train_id = i
    print(f'---------machine index: {train_id}---------')
    save_path = config.save_dir/ f'cluster_{i}'
    save_path.mkdir(parents=True, exist_ok=True)

    fw_time = open(exp_dir/'time.txt','w')
    fw_time.write("global_batch_size:{}\n\n".format(global_batch_size))
    train_start = time.time()
    model.fit(x_train, save_path)
    train_end = time.time()
    total_train_time += train_end - train_start
    train_time = train_end - train_start
    fw_time.write('train time: {}\n\n'.format(train_time))

    # test
    print(f'-------testing for cluster {i}---------')     

    test_model = DAGMM(
        comp_hiddens=comp_hidden, comp_activation=act_func,
        est_hiddens=est_hidden, est_activation=act_func, est_dropout_ratio=0,
        minibatch_size=global_batch_size, epoch_size=global_epochs, z_dim=est_hidden[-1],
        x_dim=feature_dim, learning_rate=global_learning_rate
    )
    save_path = config.save_dir/ f'cluster_{i}'
    test_model.restore(save_path)
    (config.result_dir/f'{i}').mkdir(parents=True, exist_ok=True)       

    test_start_time = time.time()
    score, recon, z = test_model.predict(x_test, save_path)
    test_end_time = time.time()
    test_time = test_end_time-test_start_time
    fw_time.write('test time: {}\n'.format(test_time))
    fw_time.close()

    np.save(config.result_dir/f'{i}/test_score.npy', score)
    np.save(config.result_dir/f'{i}/recon_mean.npy', recon)
    np.save(config.result_dir/f'{i}/z.npy', z)
    return total_train_time

def by_entity():
    total_train_time = 0

    # ts_list = Parallel(n_jobs=1)(delayed(end_to_end)(cluster, config, train_data[cluster['center']].T, test_data[cluster['center']].T) for cluster in clusters)
    # total_train_time += sum(ts_list)
    for i,(tr,te,lab) in enumerate(zip(train_data,test_data,label)):
        if i+1 not in chosed_index:
            continue
        tr=np.array(tr)
        te=np.array(te)
        torch_seed()
        train_time=end_to_end(i, config, tr.T, te.T)
        print(f"{i}--{train_time}s")
        total_train_time+=train_time
    print(f'exp:{dataset_type}{exp_key}total train time: {total_train_time}')

def by_dataset():
    total_train_time = 0

    # ts_list = Parallel(n_jobs=1)(delayed(end_to_end)(cluster, config, train_data[cluster['center']].T, test_data[cluster['center']].T) for cluster in clusters)
    # total_train_time += sum(ts_list)
    for i,(tr,te,lab) in enumerate(zip(train_data,test_data,label)):
        tr=np.array(tr)
        te=np.array(te)
        torch_seed()
        train_time=end_to_end(i, config, tr.T, te.T)
        print(f"{i}--{train_time}s")
        total_train_time+=train_time
    print(f'exp:{dataset_type}{exp_key}total train time: {total_train_time}')

def main():
    if train_type == "noshare":
        by_entity()
    else:
        by_entity()

if __name__ == '__main__':
    # get config
    config = Config()
    print(config.save_dir)
    main()

