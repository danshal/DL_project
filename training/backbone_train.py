#region imports
#General imports
import argparse
import glob
import os
import os.path as osp
import pprint
import soundfile as sf
#facebook team framework
import fairseq
#pytorch
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import torchaudio
torchaudio.set_audio_backend("sox_io")
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    walk_files,
)
from multiprocessing import set_start_method
import multiprocessing
import matplotlib.pyplot as plt
#numpy
import numpy as np
#pandas
import pandas as pd
try:
    import tqdm
except:
    print("Install tqdm to use --log-format=tqdm")

from helper_functions import sv_helper
from sv_dataset import SV_LIBRISPEECH
from sv_dataset import SV_LIBRISPEECH_PAIRS
import Models
import time
from pytorch_metric_learning import losses, miners, distances, reducers, testers, samplers
import pytorch_metric_learning.utils.loss_and_miner_utils as lmu
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import faiss
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from pytorchtools import EarlyStopping
import sklearn.metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from datetime import datetime
#endregion


#region compute EER
def compute_eer(miner):
    """Compute the Equal Error Rate (EER) from the predictions and scores.
    Args:
        labels (list[int]): values indicating whether the ground truth
            value is positive (1) or negative (0).
        scores (list[float]): the confidence of the prediction that the
            given sample is a positive.
    Return:
        (float, thresh): the Equal Error Rate and the corresponding threshold
    NOTES:
       The EER corresponds to the point on the ROC curve that intersects
       the line given by the equation 1 = FPR + TPR.
       The implementation of the function was taken from here:
       https://yangcha.github.io/EER-ROC/
    """
    dist_labels =torch.cat([torch.zeros(miner.neg_pair_all_dist.shape),torch.ones(miner.pos_pair_all_dist.shape)],dim=0)
    dists =  torch.cat([miner.neg_pair_all_dist,miner.pos_pair_all_dist],dim=0)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Tensor.cpu(dist_labels), Tensor.cpu(dists), pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh
#endregion


#region Dataset Constants
rootDir = '/home/Daniel/DeepProject/dataset'
URL = "train-clean-100"
FOLDER_IN_ARCHIVE_THREE_SEC_REPR = "cut_train_data_360_full_repr"
FOLDER_IN_ARCHIVE_THREE_SEC_REPR_AVG = "cut_train_data_360_repr"
FOLDER_IN_ARCHIVE_THREE_SEC_AUDIO = "cut_train_data_360"
FOLDER_IN_ARCHIVE_THREE_SEC_REPR_TEST = "cut_test_full_repr"
FOLDER_IN_ARCHIVE_THREE_SEC_REPR_AVG_TEST = "cut_test_repr"
FOLDER_IN_ARCHIVE_THREE_SEC_AUDIO_TEST = "cut_test-clean"
FOLDER_IN_ARCHIVE_ORIGINAL_LIBRI = "LibriSpeech"
_CHECKSUMS = {
    "http://www.openslr.org/resources/12/dev-clean.tar.gz":
    "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3",
    "http://www.openslr.org/resources/12/dev-other.tar.gz":
    "12661c48e8c3fe1de2c1caa4c3e135193bfb1811584f11f569dd12645aa84365",
    "http://www.openslr.org/resources/12/test-clean.tar.gz":
    "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",
    "http://www.openslr.org/resources/12/test-other.tar.gz":
    "d09c181bba5cf717b3dee7d4d592af11a3ee3a09e08ae025c5506f6ebe961c29",
    "http://www.openslr.org/resources/12/train-clean-100.tar.gz":
    "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2",
    "http://www.openslr.org/resources/12/train-clean-360.tar.gz":
    "146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf",
    "http://www.openslr.org/resources/12/train-other-500.tar.gz":
    "ddb22f27f96ec163645d53215559df6aa36515f26e01dd70798188350adcb6d2"
}
#endregion

#set for REPRODUCIBILITY
torch.manual_seed(42)
np.random.seed(42)

cur_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
base_summary_path = '/home/Daniel/DeepProject/'
writer_path = f'{base_summary_path}/summary_events_{cur_time}'
writer = SummaryWriter(writer_path)



def balance_pairs_amount(indices_tuple):
    list_indices_tuple = list(indices_tuple)
    pos_exmp_num = torch.tensor(list_indices_tuple[0].shape).item()
    list_indices_tuple[2] = list_indices_tuple[2][:pos_exmp_num]
    list_indices_tuple[3] = list_indices_tuple[3][:pos_exmp_num]
    return tuple(list_indices_tuple)


def train(model,pos_margin, neg_margin , loss_func , reducer, device, train_loader, optimizer, epoch, update_flag):
  margin_mining_func = miners.PairMarginMiner(pos_margin=pos_margin, neg_margin=neg_margin)
  no_constraint_mining_func = miners.PairMarginMiner(pos_margin=0, neg_margin=100)
  model.to(device)
  model.train()
  num_of_pos_pairs_over_margin = 0
  total_num_of_pos_pairs = 0
  num_of_neg_pairs_under_margin = 0
  total_num_of_neg_pairs = 0
  
  for batch_idx, (data, labels) in enumerate(train_loader):
     data, labels = data.to(device), labels.to(device)
     optimizer.zero_grad()
     embeddings = model(data)
     indices_tuple = margin_mining_func(embeddings, labels)
     _ = no_constraint_mining_func(embeddings, labels)
     num_of_pos_pairs_over_margin += margin_mining_func.num_pos_pairs
     total_num_of_pos_pairs += no_constraint_mining_func.num_pos_pairs
     num_of_neg_pairs_under_margin += margin_mining_func.num_neg_pairs
     total_num_of_neg_pairs += no_constraint_mining_func.num_neg_pairs
    #  if indices_tuple[0].shape < indices_tuple[2].shape:
    #    indices_tuple = balance_pairs_amount(indices_tuple)  #force positive and negative pairs to be with equal amount
     loss = loss_func(embeddings, labels, indices_tuple)
     writer.add_scalar(f'Train_Loss/train_{epoch}_epoch', loss, batch_idx)
     #mean summary
     writer.add_scalars(f'Mean_Distance/train_epoch_{epoch}', {'mean_pos_pair_dist': margin_mining_func.pos_pair_dist, 'mean_neg_pair_dist': margin_mining_func.neg_pair_dist}, batch_idx)
     #std summary
     writer.add_scalars(f'Std_Distance/train_epoch_{epoch}', {'pos_pair_dist_std': margin_mining_func.pos_pair_dist_std, 'neg_pair_dist_std': margin_mining_func.neg_pair_dist_std}, batch_idx)
     #min & max summary
     writer.add_scalars(f'Min_Max_Distance/train_epoch_{epoch}', {'pos_pair_min_dist': margin_mining_func.pos_pair_min_dist, 'neg_pair_max_dist': margin_mining_func.neg_pair_max_dist}, batch_idx)
     loss.backward()
     optimizer.step()
     if batch_idx % 200 == 0:
         print(f'Finished {200 * batch_idx} batches')
  if update_flag:
    if num_of_pos_pairs_over_margin/total_num_of_pos_pairs > 0.7:
            pos_margin = margin_mining_func.pos_margin * 1.1
    elif num_of_pos_pairs_over_margin/total_num_of_pos_pairs < 0.3:
            pos_margin = margin_mining_func.pos_margin * 0.9
    if num_of_neg_pairs_under_margin/total_num_of_neg_pairs > 0.95:
            neg_margin = margin_mining_func.neg_margin * 0.9
    elif num_of_neg_pairs_under_margin/total_num_of_neg_pairs < 0.05:
            neg_margin = margin_mining_func.neg_margin * 1.1
    print("Updated negative margin to : {} ".format(neg_margin))
    print("Updated positive margin to : {}".format(pos_margin))
  else:
      print('No margin updates were made...')
  print(f"*********************Finished Training EPOCH number {epoch}*********************")
  print("Updated negative margin to : {} ".format(neg_margin))      
  print("Updated positive margin to : {}".format(pos_margin))
  print(f"Train negative pairs under margin:{num_of_neg_pairs_under_margin}, percentage : {100 * num_of_neg_pairs_under_margin/total_num_of_neg_pairs}%")
  print(f"Train positive pairs over margin:{num_of_pos_pairs_over_margin} , percentage : {100 * num_of_pos_pairs_over_margin/total_num_of_pos_pairs}%")
  return pos_margin , neg_margin



### compute accuracy ###
def evaluation(data_loader, model, mining_func_no_Constraint, device, epoch, early_stopping, is_val):
  model.to(device)
  model.eval()
  batch_counter = 0
  sum_eer = 0
  sum_thresh = 0
  total_pos_pairs_dists = torch.tensor([])
  total_neg_pairs_dists = torch.tensor([])
  with torch.no_grad():
    start_time = time.time()
    for batch_idx, (data, labels) in enumerate(data_loader, 0):
        batch_counter += 1
        data, labels = data.to(device), labels.to(device)
        embeddings = model(data)
        _ = mining_func_no_Constraint(embeddings, labels)
        batch_eer, batch_thresh = compute_eer(mining_func_no_Constraint)
        sum_eer = sum_eer + batch_eer
        sum_thresh = sum_thresh + batch_thresh
        if batch_idx == 0:
            total_pos_pairs_dists = mining_func_no_Constraint.pos_pair_all_dist
            total_neg_pairs_dists = mining_func_no_Constraint.neg_pair_all_dist
        else:
            total_pos_pairs_dists = torch.cat([total_pos_pairs_dists.to(device) , mining_func_no_Constraint.pos_pair_all_dist])
            total_neg_pairs_dists = torch.cat([total_neg_pairs_dists.to(device) , mining_func_no_Constraint.neg_pair_all_dist])

        if batch_idx % 200 == 0:
            print("Finished evaluate {} batches".format(batch_idx))
    eer = sum_eer / (batch_idx + 1)
    thresh = sum_thresh / (batch_idx + 1)
    print("Validation negative average distance:{}".format(mining_func_no_Constraint.neg_pair_dist))
    print("Validation positive average distance:{}".format(mining_func_no_Constraint.pos_pair_dist))
    print(f'Validation THRESHOLD = {thresh}')
    print(f"Validation proccessing took: {(time.time()-start_time):.2f} seconds")
    writer.add_scalars(f'EER Vs Epoch/test', {'eer': eer}, epoch)
    writer.add_scalars(f'THRESHOLD Vs Epoch/test', {'threshold': thresh}, epoch)
    writer.add_histogram(f'epoch {epoch} pos distance histogram',total_pos_pairs_dists)
    writer.add_histogram(f'epoch {epoch} neg distance histogram',total_neg_pairs_dists)
    if is_val == True:
        early_stopping((1 - eer) * 100, model)
        if early_stopping.early_stop:
            return -1, thresh
    return (1 - eer) * 100, thresh


def main():
  #Declaring GPU device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f'You are using {device} device')
  #resnet18 = models.resnet18(pretrained=True)
  #Hyperparametes
  batch_size = 64
  epochs = 40
  learning_rate = 0.000005

  #region Get train & test datasets and DataLoaders
  train_data = SV_LIBRISPEECH_PAIRS('/home/Daniel/DeepProject/',
                                 folder_in_archive = FOLDER_IN_ARCHIVE_THREE_SEC_AUDIO, download=False, file_ext='.flac')
  print(f'Number of training examples(utterances): {len(train_data)}')

  #my_train_loader = torch.utils.data.DataLoader(my_train_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

  test_data = SV_LIBRISPEECH_PAIRS('/home/Daniel/DeepProject/',
                                url = "test-clean",
                                folder_in_archive = FOLDER_IN_ARCHIVE_THREE_SEC_AUDIO_TEST,
                                download = False, file_ext='.flac')
  print(f'Number of test examples(utterances): {len(test_data)}')
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
  #endregion


  on_top_net = Models.FC_SV()
  on_top_net.load_state_dict(torch.load('/home/Daniel/DeepProject/checkpoints/feature_extractor/eer_3_02_p05_n083.pt'))
  net = Models.Wav2vecTuning(on_top_net)
  optimizer = optim.Adam(net.parameters(), lr=learning_rate)
  lmbda = lambda epoch: 1.05
#   lmbda2 = lambda epoch: 1.2
  scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda = lmbda)

  ### pytorch-metric-learning stuff ###
  reducer = reducers.ThresholdReducer(low = 0)
  ### train proccess
  pos_margin = 0.5
  neg_margin = 0.83
  loss_func = losses.ContrastiveLoss(pos_margin=0, neg_margin=neg_margin,reducer = reducer)
  ### test proccess
  #test_err_mining_func = miners.PairMarginMiner(pos_margin=threshold, neg_margin=threshold)
  test_no_constraint_mining_func = miners.PairMarginMiner(collect_stats=True,pos_margin=0., neg_margin=100.)
  ### pytorch-metric-learning stuff ###

  patience = 10
  early_stopping = EarlyStopping(patience=patience, verbose=True,
   path = f'/home/Daniel/DeepProject/checkpoints/fine_tuning/checkpoint_{cur_time}.pt',
   pos_margin=pos_margin, neg_margin=neg_margin, lr=learning_rate, batch_size=batch_size)
  last_eer = 100
  update_flag = True
  thresh_list = []
  for epoch in range(epochs):
      start_train_time = time.time()
      #optimizer = optim.AdamW(net.parameters(), lr=i)
      pos_margin , neg_margin  = train(net,pos_margin , neg_margin,loss_func, reducer, device, train_loader, optimizer, epoch,update_flag)
      print(f'Finished train epoch in {(time.time() - start_train_time):.2f} seconds')
      train_eer, train_thresh = evaluation(train_loader, net, test_no_constraint_mining_func, device, epoch, early_stopping, False)
      val_eer, val_thresh = evaluation(test_loader, net, test_no_constraint_mining_func, device, epoch, early_stopping, True)
      thresh_list.append(val_thresh)
      if val_eer == -1:
          print('Early stopping')
          break
      print(f"Train EER = {train_eer}")
      print(f"Val EER = {val_eer}")
      if val_eer < last_eer:
        update_flag = True
        last_eer = val_eer
      else:
        update_flag = False
      scheduler.step()
      print(f"current learning rate:{scheduler.get_last_lr()[0]}")
  print(f'Saved model with lowest validation EER = {(early_stopping.val_loss_min):.2f}%')
  writer.close()
  if val_eer == -1:
      print(f'Your Threshold should be {thresh_list[len(thresh_list) - patience - 1]}')
  else:
      print(f'Your Threshold should be {thresh_list[len(thresh_list) - 1]}')
  with open(f'thresh_for_{cur_time}', 'w') as f:
      f.write(str(thresh_list[len(thresh_list) - 1]))



if __name__ == '__main__':
  main()
