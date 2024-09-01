import argparse

configs = argparse.ArgumentParser()

configs.scenario = 'mobile'
configs.experiment_name = '2_3'

configs.base_dir = 'KVC/'

configs.model_name = configs.scenario + '_' + configs.experiment_name

configs.experiment_dir = configs.base_dir + configs.model_name + '/'
configs.model_dir = configs.experiment_dir + 'models/'
configs.results_dir = configs.experiment_dir + 'results/'

configs.preprocessed_dir = configs.experiment_dir + 'preprocessed_data/'

configs.log_dir = configs.experiment_dir + 'logs/'
configs.config_log_dir = configs.experiment_dir
configs.data_dir = '../../databases/KVC_data/'
configs.dev_data_dir = configs.data_dir + '{}/{}_dev_set.npy'.format(configs.scenario, configs.scenario)
configs.comparison_file = configs.data_dir + '{}_comparisons.txt'.format(configs.scenario)


configs.log_filename = configs.log_dir + configs.model_name + '_log.txt'
configs.model_filename = configs.model_dir + configs.model_name + '.pt'

configs.sequence_length = 50
configs.batch_size_train = 64
configs.batch_size_val = 64
configs.dimensionality = 5
configs.output_dim = 64
configs.batches_per_epoch = 100
configs.val_batches_per_epoch = 10
configs.epochs = 1000


configs.K = 20  # number of Gaussian distributions
configs.hlayers = 9
configs.hlayers_rec = 2
configs.hlayers_pos = 1
configs.hheads = 10
configs.vlayers = 1
configs.vheads = 5

configs.decimals = 4


configs.test_data_dir = configs.data_dir + '{}/{}_test_sessions.npy'.format(configs.scenario, configs.scenario)
configs.test_batch_size = 5000
