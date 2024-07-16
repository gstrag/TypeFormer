import torch
import numpy as np
from utils.misc import KeystrokeSessionTriplet, compute_eer
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from utils.train_config import configs
from utils.test_config import test_configs
from model.Model import HARTrans
from sklearn.metrics.pairwise import euclidean_distances


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.makedirs(test_configs.results_dir, exist_ok=True)


TransformerModel = HARTrans(configs).double()

keystroke_dataset = list(np.load(test_configs.db_filename, allow_pickle=True))

TransformerModel.load_state_dict(torch.load(configs.model_filename))
TransformerModel.eval()

ds_e = KeystrokeSessionTriplet(keystroke_dataset[test_configs.num_validation_subjects:test_configs.num_validation_subjects+test_configs.num_test_subjects], length=test_configs.num_test_subjects, db=test_configs.db)
testing_dataloader = DataLoader(ds_e, batch_size=1, shuffle=False)

TransformerModel = TransformerModel.to(device)


embeddings = {}
for user in range(len(testing_dataloader)):
    embeddings[str(user)] = {}
    print("Computing embeddings for user " + str(user))
    for enrolment_session in range(test_configs.total_num_sessions):
        input_data = Variable(torch.from_numpy(np.reshape(testing_dataloader.dataset.Dataset[user][enrolment_session], (1, configs.sequence_length, configs.dimensionality)).astype(np.float64))).double()
        input_data = input_data.to(device)
        embedding = TransformerModel(input_data)
        embeddings[str(user)][str(enrolment_session)] = np.ravel(embedding.cpu().detach().numpy())

np.save(test_configs.results_dir + "test_embeddings_all_users.npy", embeddings)
embeddings = np.load(test_configs.results_dir + "test_embeddings_all_users.npy", allow_pickle=True).item()
#
genuine_distances = []
impostor_distances = []
for user in list(embeddings.keys()):
    enrolment_embs = list(embeddings[user].values())[:test_configs.enrolment_samples]
    test_embs = list(embeddings[user].values())[-test_configs.test_samples:]
    genuine_distances.append([np.mean(euclidean_distances(enrolment_embs, test_embs), axis = 0)])
    print("Testing user " + str(user))
    for impostor_user in [x for x in list(embeddings.keys()) if x != user]:
        impostor_test_embs = list(embeddings[impostor_user].values())[-test_configs.impostor_test_samples:]
        impostor_distances.append([np.mean(euclidean_distances(enrolment_embs, impostor_test_embs), axis=0)])

np.save(test_configs.results_dir + 'genuine_distances_{}.npy'.format(test_configs.enrolment_samples), np.array(genuine_distances, dtype = float))
np.save(test_configs.results_dir + 'impostor_distances_{}.npy'.format(test_configs.enrolment_samples), np.array(impostor_distances, dtype = float))

genuine_distances = np.ravel(np.load(test_configs.results_dir + 'genuine_distances_{}.npy'.format(test_configs.enrolment_samples), allow_pickle=True))
impostor_distances = np.ravel(np.load(test_configs.results_dir + 'impostor_distances_{}.npy'.format(test_configs.enrolment_samples), allow_pickle=True))

labels = np.array([0 for x in genuine_distances] + [1 for x in impostor_distances])
scores = np.concatenate((genuine_distances, impostor_distances))
eer = np.round(100*compute_eer(labels, scores)[0], 2)
print("Global EER (%):", eer)

eers_per_user = []
for user in range(test_configs.num_test_subjects):
    scores = np.concatenate((np.ravel(genuine_distances[user * test_configs.test_samples:test_configs.test_samples * (user + 1)]), np.ravel(impostor_distances[user * (test_configs.num_test_subjects-1):(test_configs.num_test_subjects-1) * (user + 1)])))
    labels = np.array([0 for x in np.ravel(genuine_distances[user * test_configs.test_samples:test_configs.test_samples * (user + 1)])] + [1 for x in np.ravel(impostor_distances[user * (test_configs.num_test_subjects-1):(test_configs.num_test_subjects-1) * (user + 1)])])
    eer = np.round(100*compute_eer(labels, scores)[0], 2)
    eers_per_user.append(eer)
mean_eer_per_user = np.mean(eers_per_user)
print("Mean Per-Subject EER (%):", mean_eer_per_user)
