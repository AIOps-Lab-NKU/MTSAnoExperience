# -*- coding: utf-8 -*-
import tfsnippet as spt


class ExpConfig(spt.Config):
    # Data options
    index = 0
    dataset = 'SWT'
    noExp               = 1
    GPU_number          = '0'
    channels_last       = True
    datapathForTrain    = f"../{dataset}_dataset/publicDataset/train_data{index}"
    datapathForTest     = f"../{dataset}_dataset/publicDataset/test_data{index}"
    dataReadformat      = "each"                           # or all
    labelpath           = f"../{dataset}_dataset/publicDataset/test_label{index}/"
    interpret_filepath  = f"../{dataset}_dataset/publicDataset/interpretation_label.txt"
    result_dir          = f"results/{dataset}/results{index}"

    # model parameters
    n_c = 8
    strides1 = 1
    strides2 = 1
    kernel_size1 = 3
    kernel_size2 = 3
    timeLength = 9
    metricNumber = 51
    x_dim = timeLength * metricNumber
    z_dim = 10
    norm = False
    VALID_PORTION = 0.1
    act_norm = True
    l2_reg = 0.0001
    shortcut_kernel_size = 1

    # Training parameters
    batch_size = 32  # 32
    initial_lr = 0.001  # 0.0005, 0.001
    lr_anneal_factor = 0.5  # 0.5, 0.75
    lr_anneal_epoch_freq = 2  # 20
    max_epoch =  12  # 50, 100, 200
    lr_anneal_step_freq = None
    max_step = None
    write_summary = False
    grad_clip_norm = 1.0
    check_numerics = True
    std_epsilon = 1e-10
    eff_len = 4320
    # Evaluation parameters
    test_batch_size = 32  # 64, 128, 256
    batchTest = True
    test_n_z = 20  # 5000, 1000
    train_n_samples = None
    savetrainDS = True
    savetestDS = True
    savetestTS = True
    evaluation = False
    saveMetricInfo = True

    # Test
    q = 1e-4
    level = 0.2
    eff_len = 449919

    @property
    def x_shape(self):
        return (self.timeLength, self.metricNumber, 1) if self.channels_last else (1, self.timeLength, self.metricNumber)
