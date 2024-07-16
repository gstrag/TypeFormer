import argparse
from utils.train_config import configs


test_configs = argparse.ArgumentParser()

if test_configs.db == 'Aalto_mobile':
    test_configs.db_filename = configs.main_db
    test_configs.results_dir = configs.base_dir + '{}_results/'.format(test_configs.db)
    test_configs.num_test_subjects = 1000
    test_configs.num_validation_subjects = configs.num_validation_subjects
    test_configs.total_num_sessions = 15
    test_configs.enrolment_samples = [1, 2, 5, 7, 10][2]
    test_configs.test_samples = 5
    test_configs.impostor_test_samples = 1

