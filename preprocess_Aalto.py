import numpy as np
import pandas as pd
import time

start = time.time()



file_raw = '../DBs/Aalto_mobile/Data_Raw/keystrokes.csv'
file_users = '../DBs/Aalto_mobile/Data_Raw/test_sections.csv'

def extract_keys_features(session_key):
    Press = np.asarray(session_key.PRESS_TIME)
    Release = np.asarray(session_key.RELEASE_TIME)
    key_code = np.asarray(session_key.KEYCODE) / 255
    key_name = np.asarray(session_key.LETTER)
    hold_time = (Release - Press) / 1000
    inter_press_time = np.append(0, np.diff(Press) / 1000)
    inter_release_time = np.append(0, np.diff(Release) / 1000)
    inter_key_time = np.append(0, (Release[:-1] - Press[1:]) / 1000)
    keys_features = np.array(
        [hold_time.astype(np.float32), inter_press_time.astype(np.float32), inter_release_time.astype(np.float32),
         inter_key_time.astype(np.float32), key_code.astype(np.float32), key_name])
    return keys_features.T

# rows = 10000 # 4*500000
# sessions = 150 # 4*8000

NUM_SESSIONS = 15
keys_db = pd.read_csv(file_raw, sep=",", index_col=False, header=None, encoding_errors='replace',
                      names = ['KEYSTROKE_ID', 'PRESS_TIME', 'RELEASE_TIME', 'LETTER', 'TEST_SECTION_ID', 'KEYCODE', 'IKI'])  #, nrows=rows)

other_db = pd.read_csv(file_users, sep=",", index_col=False, header=None, encoding_errors='replace',
                      names = ['TEST_SECTION_ID', 'SENTENCE_ID', 'PARTICIPANT_ID', 'USER_INPUT', 'INPUT_TIME', 'EDIT_DISTANCE',
                               'ERROR_RATE', 'WPM', 'INPUT_LENGTH', 'ERROR_LEN', 'POTENTIAL_WPM', 'POTENTIAL_LENGTH', 'DEVICE'])  # , nrows=sessions)

dummy_column = [0 for x in range(len(keys_db))]
keys_db['PARTICIPANT_ID'] = dummy_column
for test_section_id in other_db['TEST_SECTION_ID']:
    participant_id = int(other_db.loc[other_db['TEST_SECTION_ID'] == test_section_id]['PARTICIPANT_ID'])
    keys_db.loc[keys_db.TEST_SECTION_ID == test_section_id, 'PARTICIPANT_ID'] = (keys_db.PARTICIPANT_ID + participant_id).astype(int)
    print("test_section_id", test_section_id)
del dummy_column




keys_db = keys_db[(keys_db.PARTICIPANT_ID != 0)]

for participant_id in set(keys_db['PARTICIPANT_ID']):
    if keys_db[keys_db['PARTICIPANT_ID'] == participant_id]['TEST_SECTION_ID'].nunique() < NUM_SESSIONS:
        keys_db = keys_db[keys_db.PARTICIPANT_ID != participant_id]
    print("participant_id", participant_id)

keys_db = keys_db.set_index(['PARTICIPANT_ID', 'TEST_SECTION_ID'])
keys_db = keys_db.sort_index()

users_ids = list(keys_db.groupby(level=0).first().index)
users_len_list = [len(keys_db.loc[x].groupby(level=[0]).size()) for x in users_ids]


keys_features_db = []
keys_features_db_users_ids = []
indexes = np.unique(keys_db.index.values)

current_user = indexes[0][0]

end = time.time()

time_elapsed = (end-start)/60
print("time_elapsed:", time_elapsed)


keys_feature_session = []
for index in indexes:
    session_key = keys_db.loc[index]
    next_user = index[0]
    keys_features = extract_keys_features(session_key)
    if current_user != next_user:
        if len(keys_feature_session) >= NUM_SESSIONS:
            keys_features_db.append(keys_feature_session)
            keys_features_db_users_ids.append(current_user)
        current_user = next_user
        keys_feature_session = []
    keys_feature_session.append(keys_features)
keys_features_db.append(keys_feature_session)
keys_features_db_users_ids.append(current_user)

np.save('keystroke_all_list.npy', keys_features_db)

keys_features_db_dict = {}
current_user = indexes[0][0]
keys_feature_session_dict = {}
for index in indexes:
    session_key = keys_db.loc[index]
    next_user = index[0]
    keys_features = extract_keys_features(session_key)
    if current_user != next_user:
        if len(keys_feature_session) >= NUM_SESSIONS:
            keys_features_db_dict[str(current_user)] = keys_feature_session_dict
        current_user = next_user
        keys_feature_session_dict = {}
    keys_feature_session_dict[str(index[1])] = keys_features
keys_features_db_dict[str(next_user)] = keys_feature_session_dict

np.save('keystroke_all_dict.npy', keys_features_db_dict)


# file_path = 'D:/Giuseppe/DBs/Mobile_keys_db_6_features.npy'
# keystroke_dataset = list(np.load(file_path, allow_pickle=True))

#
# problematic_users = []
# problems = []
# no_problems = []
# for i in range(len(keys_features_db)):
#     if not(len(keys_features_db[i]) == (len(keystroke_dataset[i]))):
#         problems.append(['dif_num_sess', i, keys_features_db_users_ids[i], np.nan])
#         problematic_users.append(keys_features_db_users_ids[i])
#     else:
#         for j in range(len(keys_features_db[i])):
#             try:
#                 comparison = (keys_features_db[i][j][:, :-1] == keystroke_dataset[i][j][:, :-1])
#                 if np.sum(comparison) / (np.shape(comparison)[0] * np.shape(comparison)[1]) != 1.0:
#                     problems.append(['dif_val', i, keys_features_db_users_ids[i], j])
#                     problematic_users.append(keys_features_db_users_ids[i])
#             except:
#                 problems.append(['dif_ses_len', i, keys_features_db_users_ids[i], j])
#                 problematic_users.append(keys_features_db_users_ids[i])
#             else:
#                 no_problems.append([i, j])
# problematic_users = sorted(list(set(problematic_users)))


# for element in problems:
#     try:
#         print('new ' + str(element[1]) + ' ' + str(element[2]) + ' ' + ''.join(list(keys_features_db[element[1]][element[2]][:, -1])))
#     except Exception as e:
#         print('new ' + str(element[1]) + ' ' + str(element[2]) + ' ' + str(e))
#     try:
#         print('old ' + str(element[1]) + ' ' + str(element[2]) + ' ' + ''.join(list(keystroke_dataset[element[1]][element[2]][:, -1])))
#     except Exception as e:
#         print('old ' + str(element[1]) + ' ' + str(element[2]) + ' ' + str(e))
#     print('\n')
