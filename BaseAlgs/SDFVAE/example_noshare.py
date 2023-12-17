import os
import numpy as np
import torch
from sdfvae.model import SDFVAE
import time, random
from data_config_old import *
# os.environ['CUDA_VISIBLE_DEVICES'] = GPU_index
class Config:
    # max_epochs = global_epochs
    # exp_dir = exp_dir
    save_dir = exp_dir/ 'model'
    result_dir = exp_dir/'result'


def end_to_end(i, config: Config, train_data_item, test_data_item):
    total_train_time = 0
    feature_dim = train_data_item.shape[1]
    model = SDFVAE(
        s_dim=global_s_dim, d_dim=global_d_dim, conv_dim=global_conv_dim, 
        hidden_dim=global_hidden_dim, T=global_T, w=global_window_size, 
        n=feature_dim, enc_dec='CNN', nonlinearity=None, loss_fn=global_loss_fn)

    # train
    print(f'-------training for cluster {i}---------')
    x_train_list = []
    train_id = i
    print(f'---------machine index: {train_id}---------')
    x_train, x_test = preprocess(train_data_item, test_data_item)
    print(x_train.shape,x_test.shape)
    x_train_list.append(x_train)
    (config.save_dir/ f'data_{i}').mkdir(parents=True, exist_ok=True)
    save_path = config.save_dir/ f'data_{i}'/ 'model.pkl'

    fw_time = open(exp_dir/'time.txt','w')
    fw_time.write("global_batch_size:{}".format(global_batch_size))

    train_start = time.time()
    # 使用验证集默认比例
    model.fit(x_train_list, save_path, max_epoch=global_epochs, cluster_id=i)
    train_end = time.time()
    total_train_time += train_end - train_start

    train_time = train_end - train_start
    fw_time.write('train time: {}\n\n'.format(train_time))
    # test
    print(f'-------testing for cluster {i}---------')     
    # restore model 由于要选择最优的模型，因此需要重新load
    save_path = config.save_dir/ f'data_{i}'/ 'model.pkl'
    model.restore(save_path)
    (config.result_dir/f'{i}').mkdir(parents=True, exist_ok=True) 

    test_start_time = time.time()
    score, recon_mean, recon_std = model.predict(x_test)
    test_end_time = time.time()
    test_time = test_end_time-test_start_time
    fw_time.write('test time: {}\n'.format(test_time))
    fw_time.close()
    
    if score is not None:
        np.save(config.result_dir/f'{i}/test_score.npy', -score)
        np.save(config.result_dir/f'{i}/recon_mean.npy', recon_mean)
        np.save(config.result_dir/f'{i}/recon_std.npy', recon_std)
    return total_train_time


def torch_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def by_entity():
    total_train_time = 0
    # ts_list = Parallel(n_jobs=1)(delayed(end_to_end)(cluster, config, train_data[cluster['center']].T, test_data[cluster['center']].T) for cluster in clusters)
    # total_train_time += sum(ts_list)

    for i,(tr,te,lab) in enumerate(zip(train_data,test_data,label)):
        torch_seed()
        if i+1 not in chosed_index:
            continue
        tr=np.array(tr)
        te=np.array(te)
        print('tr.shape',tr.shape)
        print('te.shape',te.shape)
        # print(feature_dim)
        # train_data_item, test_data_item = get_data_by_index(dataset_type, cluster['center'], training_period)
        train_time=end_to_end(i, config, tr.T, te.T)
        print(i,train_time,'s')
        total_train_time+=train_time
        # break

    print(f'exp:{dataset_type}{exp_key}total train time: {total_train_time}')

def by_dataset():
    total_train_time = 0
    # ts_list = Parallel(n_jobs=1)(delayed(end_to_end)(cluster, config, train_data[cluster['center']].T, test_data[cluster['center']].T) for cluster in clusters)
    # total_train_time += sum(ts_list)

    for i,(tr,te,lab) in enumerate(zip(train_data,test_data,label)):
        torch_seed()
        tr=np.array(tr)
        te=np.array(te)
        print('tr.shape',tr.shape)
        print('te.shape',te.shape)
        # print(feature_dim)
        # train_data_item, test_data_item = get_data_by_index(dataset_type, cluster['center'], training_period)
        train_time=end_to_end(i, config, tr.T, te.T)
        print(i,train_time,'s')
        total_train_time+=train_time
        # break

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

