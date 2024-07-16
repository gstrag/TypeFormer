import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.misc import KeystrokeSessionTriplet

from sklearn.metrics.pairwise import euclidean_distances
from utils.train_config import configs
from utils.misc import compute_eer, TripletLoss

import time


from model.Preliminary import HARTrans


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

os.makedirs(configs.base_dir, exist_ok=True)

# Saving specific config file for reproducibility
with open('utils/train_config.py') as f:
    data = f.read()
    f.close()
with open(configs.base_dir + "experimental_config.txt", mode="w") as f:
    f.write(data)
    f.close()

keystroke_dataset = list(np.load(configs.main_db, allow_pickle=True))

ds_t = KeystrokeSessionTriplet(keystroke_dataset[configs.num_training_subjects:2*configs.num_training_subjects], data_length=configs.sequence_length, length=len(keystroke_dataset))
ds_v = KeystrokeSessionTriplet(keystroke_dataset[:configs.num_validation_subjects], data_length=configs.sequence_length, length=len(keystroke_dataset))

train_dataloader = DataLoader(ds_t, batch_size=configs.batch_size_train, shuffle=True)
val_dataloader = DataLoader(ds_v, batch_size=configs.batch_size_val, shuffle=True)

TransformerModel = HARTrans(configs).double()

optimizer = torch.optim.Adam(TransformerModel.parameters(), lr=0.001, betas=(0.9, 0.999))
TransformerModel = TransformerModel.to(device)
loss_fn = TripletLoss()

def train_one_epoch(epoch_index):
    running_loss = 0.
    epoch_eer = []
    total_loss_per_epoch = 0.
    for i, (anchor_sgm, positive_sgm, negative_sgm) in enumerate(train_dataloader, 0):#, anchor_label) in enumerate(train_dataloader, 0):
        # print("Batch " + str(i) + " of " + str(configs.batches_per_epoch) + " batches per epoch")
        if i == configs.batches_per_epoch:
            break
        optimizer.zero_grad()
        anchor_sgm, positive_sgm, negative_sgm = Variable(anchor_sgm).double().to(device), Variable(positive_sgm).double().to(device), Variable(negative_sgm).double().to(device)
        anchor_out, positive_out, negative_out = TransformerModel(anchor_sgm), TransformerModel(positive_sgm), TransformerModel(negative_sgm)
        criterion = torch.jit.script(TripletLoss())
        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        pred_a, pred_p, pred_n = anchor_out.cpu().detach().numpy(), positive_out.cpu().detach().numpy(), negative_out.cpu().detach().numpy()
        scores_g, scores_i = [], []
        for j in range(configs.batch_size_train):
            scores_g.append(euclidean_distances(pred_a[j].reshape(1, -1), pred_p[j].reshape(1, -1)))
            scores_i.append(euclidean_distances(pred_a[j].reshape(1, -1), pred_n[j].reshape(1, -1)))
        labels = np.array([0 for x in range(configs.batch_size_train)] + [1 for x in range(configs.batch_size_train)])
        eer, _ = compute_eer(labels, np.squeeze(np.array(scores_g+scores_i)))
        epoch_eer.append(eer)
        total_loss_per_epoch = total_loss_per_epoch + running_loss
    return total_loss_per_epoch, np.mean(epoch_eer)



best_vloss = 1_000_000.
best_eer_v = 100.
best_eer_v = 100.
best_epoch = 0

loss_t_list, eer_t_list = [], []
loss_v_list, eer_v_list = [], []


for epoch in range(configs.epochs):
    print('EPOCH:', epoch)
    start = time.time()

    # Make sure gradient tracking is on, and do a pass over the data
    TransformerModel.train()

    loss, eer = train_one_epoch(epoch)
    loss_t_list.append(loss)
    eer_t_list.append(eer)
    print("Loss " + str(loss))
    print("EER [%] "+ str(eer))


    running_loss_v = 0.
    epoch_eer_v = []
    total_loss_per_epoch_v = 0.
    for k, (anchor_sgm, positive_sgm, negative_sgm) in enumerate(val_dataloader, 0):
        # print("Batch " + str(k) + " of " + str(configs.val_batches_per_epoch) + " batches per epoch")
        if k == configs.val_batches_per_epoch:
            break
        # zero gradients
        anchor_sgm, positive_sgm, negative_sgm = Variable(anchor_sgm).double().to(device), Variable(positive_sgm).double().to(device), Variable(negative_sgm).double().to(device)
        anchor_out, positive_out, negative_out = TransformerModel(anchor_sgm), TransformerModel(positive_sgm), TransformerModel(negative_sgm)
        criterion = torch.jit.script(TripletLoss())
        loss_v = criterion(anchor_out, positive_out, negative_out)
        running_loss_v += loss_v.item()
        pred_a_v, pred_p_v, pred_n_v = anchor_out.cpu().detach().numpy(), positive_out.cpu().detach().numpy(), negative_out.cpu().detach().numpy()
        scores_g_v, scores_i_v = [], []
        for j in range(configs.batch_size_val):
            scores_g_v.append(euclidean_distances(pred_a_v[j].reshape(1, -1), pred_p_v[j].reshape(1, -1)))
            scores_i_v.append(euclidean_distances(pred_a_v[j].reshape(1, -1), pred_n_v[j].reshape(1, -1)))
        labels_v = np.array([0 for x in range(configs.batch_size_val)] + [1 for x in range(configs.batch_size_val)])
        eer, _ = compute_eer(labels_v, np.squeeze(np.array(scores_g_v+scores_i_v)))
        epoch_eer_v.append(eer)
        total_loss_per_epoch_v = total_loss_per_epoch_v + running_loss_v
    loss_v_list.append(total_loss_per_epoch_v)
    eer_v_list.append(np.mean(epoch_eer_v))
    print("Val Loss " + str(total_loss_per_epoch_v))
    print("Val EER [%] "+ str(np.mean(epoch_eer_v)))
    end = time.time()
    print("Time for last epoch [min]: " + str((end-start)/60))
    if np.mean(epoch_eer_v) < best_eer_v:
        best_eer_v = np.mean(epoch_eer_v)
        best_epoch = epoch
        print('Best EER validation achieved: ' + str(best_eer_v) + '...')
        torch.save(TransformerModel.state_dict(), configs.model_filename)
    log_list = [loss_t_list, loss_v_list, eer_t_list, eer_v_list]
    with open(configs.log_filename, "w") as output:
        output.write(str(log_list))

print('\nBest Validation EER: ' + str(best_eer_v)[:6] + ', in epoch: ' + str(best_epoch))

