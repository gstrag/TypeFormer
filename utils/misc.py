import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve
import random

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


def compute_eer(labels, scores):
    """
    labels: 1D np.array, 0 = impostor comparison, 1 = genuine comparison
    scores: 1D np.array
    """
    fmr, tmr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnmr = 1 - tmr
    eer_index = np.nanargmin(np.abs((fnmr - fmr)))
    eer_threshold = thresholds[eer_index]
    eer = np.mean((fmr[eer_index], fnmr[eer_index]))
    return eer, eer_threshold


class KeystrokeSessionTriplet():
    def __init__(self, Dataset, data_length=50, dimension=5, max_num_sample_per_user=15, length=1024, db='Aalto_mobile'):
        self.Dataset = Dataset
        print('loaded dataset')
        self.data_length = data_length
        self.dimension = dimension
        self.len = length
        self.db = db
        self.num_users = len(Dataset)
        self.user_idx_list = []
        self.user_idx_n_list = []
        self.sample_idx_p_list = []
        self.sample_idx_n_list = []
        self.sample_idx_list = []
        self.anchor_list = []
        self.positive_list = []
        self.negative_list = []
        self.max_num_sample_per_user = max_num_sample_per_user
        for i in range(len(self.Dataset)):
            self.Dataset[i] = self.Dataset[i][:self.max_num_sample_per_user]
        for i in range(len(self.Dataset)):
            # print(i)
            for j in range(len(self.Dataset[i])):
                if self.db == 'Aalto_mobile':
                    self.Dataset[i][j] = np.delete(self.Dataset[i][j], -1, 1)
                # self.Dataset[i][j] = np.delete(self.Dataset[i][j], [0, 2, 3, -1], 1) # not for aalto db
        if self.db == 'Clarkson_II':
            self.Dataset = np.delete(self.Dataset, [x for x in range(50, 150)], axis=2) # only for clarkson II
        for i in range(len(self.Dataset)):
            # print(i)
            for j in range(len(self.Dataset[i])):
                self.Dataset[i][j] = np.concatenate((self.Dataset[i][j], np.zeros((self.data_length, dimension))))[
                                     :self.data_length]

    def __getitem__(self, index):
        user_idx = random.randint(0, self.num_users - 1)
        sample_idx = random.randint(0, self.max_num_sample_per_user - 1)
        self.user_idx_list.append(user_idx)
        self.sample_idx_list.append(sample_idx)
        anchor_segment = self.Dataset[user_idx][sample_idx].astype(np.double)
        self.anchor_list.append((user_idx, sample_idx))

        sample_idx_p = random.randint(0, self.max_num_sample_per_user - 1)
        while sample_idx_p == sample_idx:
            sample_idx_p = random.randint(0, self.max_num_sample_per_user - 1)
        self.sample_idx_p_list.append(sample_idx_p)
        positive_segment = self.Dataset[user_idx][sample_idx_p].astype(np.double)
        self.positive_list.append((user_idx, sample_idx_p))

        sample_idx_n = random.randint(0, self.max_num_sample_per_user - 1)
        user_idx_n = random.randint(0, self.num_users - 1)
        while user_idx_n == user_idx:
            user_idx_n = random.randint(0, self.num_users - 1)
        negative_segment = self.Dataset[user_idx_n][sample_idx_n].astype(np.double)
        self.user_idx_n_list.append(user_idx_n)
        self.sample_idx_n_list.append(sample_idx_n)
        self.negative_list.append((user_idx_n, sample_idx_n))

        return anchor_segment, positive_segment, negative_segment  # , anchor_label

    def __len__(self):
        return self.len


def extract_test_keystroke_features(session_1, sequence_length=100):
    hold_time_1 = np.reshape((session_1[:, 1] - session_1[:, 0]) / 1e3, (np.shape(session_1)[0], 1)).astype(
        np.float32)
    inter_press_1 = np.reshape(np.append(0, np.diff(session_1[:, 0])) / 1e3,
                               (np.shape(session_1)[0], 1)).astype(np.float32)
    inter_release_1 = np.reshape(np.append(0, np.diff(session_1[:, 1])) / 1e3,
                                 (np.shape(session_1)[0], 1)).astype(np.float32)
    inter_key_1 = np.reshape(np.append(0, session_1[:-1, 1] - session_1[1:, 0]) / 1e3,
                             (np.shape(session_1)[0], 1)).astype(np.float32)
    # ascii_1 = np.reshape(session_1[:, 2] / 256, (np.shape(session_1)[0], 1)).astype(np.float32)
    # session_1_processed = np.concatenate(
   #     (hold_time_1, inter_press_1, inter_release_1, inter_key_1, ascii_1), axis=1)
    session_1_processed = np.concatenate(
        (hold_time_1, inter_press_1, inter_release_1, inter_key_1), axis=1)
    session_1_processed = np.concatenate(
        (session_1_processed, np.zeros((sequence_length, np.shape(session_1_processed)[1]))))[
                          :sequence_length]
    return session_1_processed
