import argparse

configs = argparse.ArgumentParser()

configs.model_name = 'latest_experiment'
configs.base_dir = configs.model_name + '/'
configs.model_dir = configs.base_dir
configs.log_dir = configs.base_dir
configs.data_dir = 'data/'
configs.log_filename = configs.log_dir + configs.model_name + '_log.txt'
configs.model_filename = configs.model_dir + configs.model_name + '.pt'


configs.main_db = configs.data_dir + 'Mobile_keys_db_6_features.npy'
configs.total_users = 60000
configs.num_training_subjects = 30000
configs.num_validation_subjects = 400
configs.sequence_length = 50

configs.batch_size_train = 64
configs.batch_size_val = 400
configs.dimensionality = 5
configs.output_dim = 64
configs.batches_per_epoch = 29
configs.val_batches_per_epoch = 1
configs.epochs = 1000


configs.K = 20  # number of Gaussian distributions
configs.hlayers = 9
configs.hlayers_rec = 2
configs.hlayers_pos = 1
configs.hheads = 10
configs.vlayers = 1
configs.vheads = 5


