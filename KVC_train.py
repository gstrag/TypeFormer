import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.KVC_training import KeystrokeSessionTriplet
import random
import time
from model.Model import HARTrans

from utils.misc import compute_eer, TripletLoss
from utils.KVC_config import configs

from utils.KVC_training import preprocess


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

os.makedirs(configs.base_dir, exist_ok=True)
os.makedirs(configs.preprocessed_dir, exist_ok=True)
os.makedirs(configs.model_dir, exist_ok=True)
os.makedirs(configs.log_dir, exist_ok=True)

# Saving specific config file for reproducibility
with open('utils/KVC_config.py') as f:
    data = f.read()
    f.close()
with open(configs.config_log_dir + "experimental_config.txt", mode="w") as f:
    f.write(data)
    f.close()

try:
    keystroke_dataset = np.load(configs.preprocessed_dir + '{}_processed_{}.npy'.format(configs.scenario, configs.experiment_name), allow_pickle=True).item()
except:
    preprocess(configs.dev_data_dir, configs.scenario, configs.experiment_name, configs.preprocessed_dir, sequence_length=configs.sequence_length)
    keystroke_dataset = np.load(configs.preprocessed_dir + '{}_processed_{}.npy'.format(configs.scenario, configs.experiment_name), allow_pickle=True).item()


total_users = len(list(keystroke_dataset.keys()))
validation_users = int(total_users*0.2)

val_dataset_keys = random.sample(list(keystroke_dataset.keys()), validation_users)
train_dataset_keys = [x for x in list(keystroke_dataset.keys()) if x not in val_dataset_keys]

train_dataset = {x:keystroke_dataset[x] for x in train_dataset_keys}
val_dataset = {x:keystroke_dataset[x] for x in val_dataset_keys}

del keystroke_dataset

ds_t = KeystrokeSessionTriplet(train_dataset, train_dataset_keys, data_length=configs.sequence_length, dimension=configs.dimensionality, samples_considered_per_epoch=configs.batches_per_epoch*configs.batch_size_train)
ds_v = KeystrokeSessionTriplet(val_dataset, val_dataset_keys, data_length=configs.sequence_length, dimension=configs.dimensionality, samples_considered_per_epoch=configs.val_batches_per_epoch*configs.batch_size_val)

train_dataloader = DataLoader(ds_t, batch_size=configs.batch_size_train, shuffle=True)
val_dataloader = DataLoader(ds_v, batch_size=configs.batch_size_val, shuffle=True)


TransformerModel = HARTrans(configs).double()


optimizer = torch.optim.Adam(TransformerModel.parameters(), lr=0.001, betas=(0.9, 0.999))
TransformerModel = TransformerModel.to(device)
criterion = torch.jit.script(TripletLoss())


def inner_ops(input_, mode='train'):
    if mode == 'train':
        optimizer.zero_grad()
    anchor_sgm, positive_sgm, negative_sgm = (Variable(input_[0]).to(device),
                                              Variable(input_[1]).to(device),
                                              Variable(input_[2]).to(device))
    anchor_out, positive_out, negative_out = (TransformerModel(anchor_sgm),
                                              TransformerModel(positive_sgm),
                                              TransformerModel(negative_sgm))
    loss = criterion(anchor_out, positive_out, negative_out)
    if mode == 'train':
        loss.backward(retain_graph=True)
        optimizer.step()
    running_loss = np.round(loss.item(), configs.decimals)
    pred_a, pred_p, pred_n = (np.round(anchor_out.cpu().detach().numpy(), configs.decimals),
                              np.round(positive_out.cpu().detach().numpy(), configs.decimals),
                              np.round(negative_out.cpu().detach().numpy(), configs.decimals))
    scores_g = np.sqrt(np.add.reduce(np.square(pred_a - pred_p), 1))
    scores_i = np.sqrt(np.add.reduce(np.square(pred_a - pred_n), 1))
    labels = np.array([0 for x in range(len(scores_g))] + [1 for x in range(len(scores_i))])
    eer = np.round(compute_eer(labels, np.concatenate((scores_g, scores_i)))[0], configs.decimals)

    return eer, running_loss


def train_one_epoch():
    # Make sure gradient tracking is on, and do a pass over the data
    TransformerModel.train()
    epoch_eers = []
    total_loss_per_epoch = 0.
    for i, (anchor_sgm, positive_sgm, negative_sgm) in enumerate(train_dataloader, 0):
        eer_, running_loss_ = inner_ops((anchor_sgm, positive_sgm, negative_sgm))
        epoch_eers.append(eer_)
        total_loss_per_epoch = total_loss_per_epoch + running_loss_
    last_batch_eer = np.round(epoch_eers[-1], configs.decimals)
    return total_loss_per_epoch, last_batch_eer

def eval_one_epoch():
    epoch_eers = []
    total_loss_per_epoch = 0.
    TransformerModel.eval()
    for i, (anchor_sgm, positive_sgm, negative_sgm) in enumerate(val_dataloader, 0):
        eer_, running_loss_ = inner_ops((anchor_sgm, positive_sgm, negative_sgm), mode='eval')
        epoch_eers.append(eer_)
        total_loss_per_epoch = total_loss_per_epoch + running_loss_
    mean_eer = np.round(np.mean(epoch_eers), configs.decimals)
    return total_loss_per_epoch, mean_eer



best_vloss = 1_000_000.
best_eer_v = 100.
best_eer_v = 100.
best_epoch, new_best_epoch = 0, False

loss_t_list, eer_t_list = [], []
loss_v_list, eer_v_list = [], []


for epoch in range(configs.epochs):
    start = time.time()


    loss_t, eer_t = train_one_epoch()
    loss_t_list.append(loss_t)
    eer_t_list.append(eer_t)

    loss_v, eer_v = eval_one_epoch()
    loss_v_list.append(loss_v)
    eer_v_list.append(eer_v)

    end = time.time()
    if eer_v_list[-1] < best_eer_v:
        new_best_epoch, best_eer_v, best_epoch = True, eer_v_list[-1], epoch
        torch.save(TransformerModel.state_dict(), configs.model_filename)
    else:
        new_best_epoch = False
    print('Epoch: %d. Training set: Loss: %.2f, EER [%%]: %.2f%%. Validation set: Loss: %.2f, EER [%%]: %.2f%%. '
          'Time for last epoch [min]: %.2f. New best EER on val set: %.d'
          % (epoch, loss_t_list[-1], 100*eer_t_list[-1], loss_v_list[-1], 100*eer_v_list[-1],
             np.round((end-start)/60, configs.decimals), new_best_epoch))
    log_list = [loss_t_list, loss_v_list, eer_t_list, eer_v_list]
    with open(configs.log_filename, "w") as output:
        output.write(str(log_list))

print('\nBest Validation EER: %.2f%%, in epoch: %.d' % (best_eer_v, best_epoch))

