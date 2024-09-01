import torch
import numpy as np
from utils.misc import extract_test_keystroke_features
from torch.autograd import Variable
import os
from utils.KVC_config import configs
from model.Model import HARTrans
import shutil
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.makedirs(configs.results_dir, exist_ok=True)


TransformerModel = HARTrans(configs).double()


TransformerModel.load_state_dict(torch.load(configs.model_filename))
TransformerModel.eval()

TransformerModel = TransformerModel.to(device)

try:
    preproc_data = np.load(configs.preprocessed_dir + 'test_sessions_{}_processed_{}.npy'.format(configs.scenario, configs.experiment_name)).item()
except:
    test_dataset = np.load(configs.test_data_dir, allow_pickle=True).item()
    preproc_data = {}
    i = 0
    for session in list(test_dataset.keys()):
        if i % 100 == 0:
            print(str(np.round(100*(i/len(list(test_dataset.keys()))), configs.decimals)) + '% raw data preprocessed', end='\r')
        preproc_data[session] = extract_test_keystroke_features(test_dataset[session], sequence_length=configs.sequence_length)
        i = i + 1
    np.save(configs.preprocessed_dir + 'test_sessions_{}_processed_{}.npy'.format(configs.scenario, configs.experiment_name), preproc_data)
    del test_dataset


embeddings_list = []
TransformerModel.eval()
with torch.no_grad():
    all_sessions = np.array(list(preproc_data.values()))
    for i in range(int(len(all_sessions)/configs.test_batch_size)):
        input_data = np.reshape(all_sessions[i*configs.test_batch_size:(i+1)*configs.test_batch_size],
                                (configs.test_batch_size, configs.sequence_length, configs.dimensionality))
        input_data = Variable(torch.from_numpy(input_data)).double().to(device)
        embeddings_list.append(TransformerModel(input_data).cpu())
        print(str(np.round(100 * ((i+1) / int(len(all_sessions)/configs.test_batch_size)), configs.decimals)) +
              '% embeddings computed', end='\r')

embeddings_list = [item for sublist in embeddings_list for item in sublist]
embeddings = {}


tmp_list = list(preproc_data.keys())
for i in range(len(tmp_list)):
    embeddings[tmp_list[i]] = ''
embeddings.update(zip(embeddings, embeddings_list))

with open(configs.comparison_file, "r") as file:
    comps = eval(file.readline())


distances = {}
i = 0
for comp in comps:
    distance = nn.functional.pairwise_distance(embeddings[comp[0]], embeddings[comp[1]]).item()
    distances[str(comp)] = distance
    if i % 1000 == 0:
        print(str(np.round(100 * ((i + 1) / len(comps)), configs.decimals)) + '% distances computed', end='\r')
    i = i + 1

distances_list = list(distances.values())
max_dist = max(distances_list)
distances_list = [1-(x/max_dist) for x in distances_list]

with open(configs.result_filename, "w") as file:
    file.write(str(distances_list))

shutil.make_archive(configs.experiment_name, 'zip', configs.result_dir)


