import numpy as np
import random

def preprocess(file_loc, scenario, name, loc, sequence_length = 100):
    data = np.load(file_loc, allow_pickle=True).item()
    data_processed = {}
    i = 0
    L = len(data)
    for user_idx in list(data.keys()):
        data_processed[user_idx] = {}
        print('Preprocessing data: {}%'.format(str(100 * (i / L))[:6]), end='\r')
        for session_idx in list(data[user_idx].keys()):
            sl2 = len(data[user_idx][session_idx]) // 2

            session_1 = data[user_idx][session_idx]
            hold_time_1 = np.expand_dims((session_1[:, 1] - session_1[:, 0]) / 1E3, axis=1).astype(np.float32)
            std_len = len(hold_time_1)
            inter_press_1 = np.expand_dims(np.append(0, np.diff(session_1[:, 0])) / 1E3, axis=1).astype(np.float32)
            inter_release_1 = np.expand_dims(np.append(0, np.diff(session_1[:, 1])) / 1E3, axis=1).astype(np.float32)
            inter_key_1 = np.expand_dims(np.append(0, session_1[1:, 0] - session_1[:-1, 1]) / 1E3, axis=1).astype(np.float32)

            inter_press_2 = np.expand_dims(np.concatenate((np.array([0, 0]), np.diff(session_1[:, 0], n=2))) / 1E3, axis=1).astype(np.float32)[:std_len]
            inter_release_2 = np.expand_dims(np.concatenate((np.array([0, 0]), np.diff(session_1[:, 1], n=2))) / 1E3, axis=1).astype(np.float32)[:std_len]
            inter_key_2 = np.expand_dims(np.concatenate((np.array([0, 0]), session_1[2:, 0] - session_1[:-2, 1])) / 1E3, axis=1).astype(np.float32)[:std_len]

            inter_press_3 = np.expand_dims(np.concatenate((np.array([0, 0, 0]), np.diff(session_1[:, 0], n=3))) / 1E3, axis=1).astype(np.float32)[:std_len]
            inter_release_3 = np.expand_dims(np.concatenate((np.array([0, 0, 0]), np.diff(session_1[:, 1], n=3))) / 1E3, axis=1).astype(np.float32)[:std_len]
            inter_key_3 = np.expand_dims(np.concatenate((np.array([0, 0, 0]), session_1[3:, 0] - session_1[:-3, 1])) / 1E3, axis=1).astype(np.float32)[:std_len]

            # session_1_processed = np.concatenate((hold_time_1, inter_press_1, inter_release_1, inter_key_1,
            #                                  inter_press_2, inter_release_2, inter_key_2,
            #                                  inter_press_3, inter_release_3, inter_key_3), axis=1)

            ascii_1 = np.reshape(session_1[:, 2] / 256, (np.shape(session_1)[0], 1)).astype(np.float32)

            session_1_processed = np.concatenate((hold_time_1, inter_press_1, inter_release_1, inter_key_1,
                                             inter_press_2, inter_release_2, inter_key_2,
                                             inter_press_3, inter_release_3, inter_key_3, ascii_1), axis=1)


            # fft_hold_time_1 = np.fft.fft(np.ravel(hold_time_1))
            # fft_inter_press_1 = np.fft.fft(np.ravel(inter_press_1))
            # fft_inter_release_1 = np.fft.fft(np.ravel(inter_release_1))
            # fft_inter_key_1 = np.fft.fft(np.ravel(inter_key_1))
            #
            # fft_hold_time_1 = np.expand_dims(np.concatenate((np.abs(np.real(fft_hold_time_1)[:sl2]), np.imag(fft_hold_time_1)[:sl2])), axis=1)
            # fft_inter_press_1 = np.expand_dims(np.concatenate((np.abs(np.real(fft_inter_press_1)[:sl2]), np.imag(fft_inter_press_1)[:sl2])), axis=1)
            # fft_inter_release_1 = np.expand_dims(np.concatenate((np.abs(np.real(fft_inter_release_1)[:sl2]), np.imag(fft_inter_release_1)[:sl2])), axis=1)
            # fft_inter_key_1 = np.expand_dims(np.concatenate((np.abs(np.real(fft_inter_key_1)[:sl2]), np.imag(fft_inter_key_1)[:sl2])), axis=1)
            #
            # try:
            #     fft_hold_time_1[0] = fft_hold_time_1[0] / sl2
            #     fft_inter_press_1[0] = fft_inter_press_1[0] / sl2
            #     fft_inter_release_1[0] = fft_inter_release_1[0] / sl2
            #     fft_inter_key_1[0] = fft_inter_key_1[0] / sl2
            # except:
            #     pass

            # session_1_processed = np.concatenate((hold_time_1[:sl], inter_press_1[:sl], inter_release_1[:sl], inter_key_1[:sl]), axis=1)
            #                                       fft_hold_time_1, fft_inter_press_1, fft_inter_release_1, fft_inter_key_1), axis=1)


            # session_1_processed = np.concatenate((hold_time_1, inter_press_1, inter_release_1, inter_key_1, ascii_1), axis=1)


            # session_1_processed = np.concatenate((hold_time_1, inter_press_1, inter_release_1, inter_key_1), axis=1)
            session_1_processed = np.concatenate((session_1_processed, np.zeros((sequence_length, np.shape(session_1_processed)[1]))))[
                        :sequence_length]
            data_processed[user_idx][session_idx] = session_1_processed
        i = i + 1
    np.save(loc + '{}_processed_{}.npy'.format(scenario, name), data_processed)


class KeystrokeSessionTriplet():
    def __init__(self, Dataset, users_list, data_length=50, dimension=5, max_num_sample_per_user=15, offset=0, samples_considered_per_epoch=2048):
        self.Dataset = Dataset
        self.data_length = data_length
        self.dimension = dimension
        self.len = samples_considered_per_epoch
        self.users_list = users_list
        self.num_users_in_set = len(users_list)
        self.user_idx_list = []
        self.user_idx_n_list = []
        self.sample_idx_p_list = []
        self.sample_idx_n_list = []
        self.sample_idx_list = []
        self.anchor_list = []
        self.positive_list = []
        self.negative_list = []
        self.offset = offset
        self.max_num_sample_per_user = max_num_sample_per_user
        for i in self.users_list:
            session_list = list(self.Dataset[i].keys())[self.offset:self.max_num_sample_per_user]
            for j in list(self.Dataset[i].keys()):
                if j not in session_list:
                    del self.Dataset[i][j]
        for i in list(self.Dataset.keys()):
            if i not in self.users_list:
                del self.Dataset[i]

        for i in self.users_list:
            for j in list(self.Dataset[i].keys()):
                self.Dataset[i][j] = self.Dataset[i][j][:, :dimension]

        for i in self.users_list:
            for j in list(self.Dataset[i].keys()):
                self.Dataset[i][j] = np.concatenate((self.Dataset[i][j], np.zeros((self.data_length, dimension))))[:self.data_length]

    def __getitem__(self, index):
        user_idx_int = random.randint(0, self.num_users_in_set - 1)
        user_idx = self.users_list[user_idx_int]

        max_num_sample_per_user_a_p = self.max_num_sample_per_user-self.offset  # len(list(self.Dataset[user_idx].keys()))
        sample_idx = list(self.Dataset[user_idx].keys())[random.randint(0, max_num_sample_per_user_a_p - 1)]
        self.user_idx_list.append(user_idx)
        self.sample_idx_list.append(sample_idx)
        anchor_segment = self.Dataset[user_idx][sample_idx]
        self.anchor_list.append((user_idx, sample_idx))

        sample_idx_p = list(self.Dataset[user_idx].keys())[random.randint(0, max_num_sample_per_user_a_p - 1)]
        while sample_idx_p == sample_idx:
            sample_idx_p = list(self.Dataset[user_idx].keys())[random.randint(0, max_num_sample_per_user_a_p - 1)]
        self.sample_idx_p_list.append(sample_idx_p)
        positive_segment = self.Dataset[user_idx][sample_idx_p]
        self.positive_list.append((user_idx, sample_idx_p))

        user_idx_n_int = random.randint(0, self.num_users_in_set - 1)
        while user_idx_n_int == user_idx_int:
            user_idx_n_int = random.randint(0, self.num_users_in_set - 1)
        user_idx_n = self.users_list[user_idx_n_int]

        max_num_sample_per_user_n = self.max_num_sample_per_user-self.offset  # len(list(self.Dataset[user_idx_n].keys()))
        sample_idx_n = list(self.Dataset[user_idx_n].keys())[random.randint(0, max_num_sample_per_user_n - 1)]
        negative_segment = self.Dataset[user_idx_n][sample_idx_n]
        self.user_idx_n_list.append(user_idx_n)
        self.sample_idx_n_list.append(sample_idx_n)
        self.negative_list.append((user_idx_n, sample_idx_n))

        # return [np.nan_to_num(anchor_segment), int(float(user_idx))], [np.nan_to_num(positive_segment), int(float(user_idx))], [np.nan_to_num(negative_segment), int(float(user_idx_n))]  # , anchor_label
        return np.nan_to_num(anchor_segment), np.nan_to_num(positive_segment), np.nan_to_num(negative_segment)  # , anchor_label

    def __len__(self):
        return self.len


