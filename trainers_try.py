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
#torchaudio.set_audio_backend("sox_io")
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
import Models
import time
from pytorch_metric_learning import losses, miners, distances, reducers, testers, samplers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import faiss


#Dataset Constants:
rootDir = '/home/Daniel/DeepProject/dataset'
URL = "train-clean-100"
FOLDER_IN_ARCHIVE_THREE_SEC_REPR = "cut_train_data_360_full_repr"
FOLDER_IN_ARCHIVE_THREE_SEC_REPR_AVG = "cut_train_data_360_repr"
FOLDER_IN_ARCHIVE_THREE_SEC_REPR_AVG_SMALL = "cut_train_data_360_small_repr"
FOLDER_IN_ARCHIVE_THREE_SEC_REPR_MAX = "cut_train_data_360_max_pool_repr"
FOLDER_IN_ARCHIVE_THREE_SEC_REPR_MAX_TEST = "cut_test_max_pool_repr"
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

# Create the tester
# tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook = hooks.end_of_testing_hook, 
#                                             visualizer = umap.UMAP(), 
#                                             visualizer_hook = visualizer_hook,
#                                             dataloader_num_workers = 4)
### pytorch-metric-learning stuff ###


def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
  model.to(device)
  model.train()
  start_time = time.time()
  for batch_idx, (data, labels) in enumerate(train_loader):
     data, labels = data.to(device), labels.to(device)
     optimizer.zero_grad()
     embeddings = model(data)
     indices_tuple = mining_func(embeddings, labels)
     if indices_tuple[0].shape < indices_tuple[2].shape:
       list_indices_tuple = list(indices_tuple)
       pos_exmp_num = torch.tensor(list_indices_tuple[0].shape).item()
       list_indices_tuple[2] = list_indices_tuple[2][:pos_exmp_num]
       list_indices_tuple[3] = list_indices_tuple[3][:pos_exmp_num]
       indices_tuple = tuple(list_indices_tuple)
     loss = loss_func(embeddings, labels, indices_tuple)
     loss.backward()
     optimizer.step()
     if batch_idx % 50 == 0:
       print(" average positive distance:{} +- {} average negative distance:{} +- {}".format(mining_func.pos_pair_dist,mining_func.pos_pair_dist_std,mining_func.neg_pair_dist,mining_func.neg_pair_dist_std))
       print(" minimum positive distance:{} maximum negative distance:{}".format(mining_func.pos_pair_min_dist,mining_func.neg_pair_max_dist))

       print("Epoch {} Iteration {}: Loss = {}, Number of mined pairs = {} , 50 batches training took:{}".format(epoch, batch_idx, loss, mining_func.num_pos_pairs,time.time()-start_time))
       start_time = time.time()



### compute accuracy ###
def evaluation(data_loader, model, mining_func_err,mining_func_no_Constraint,device,epoch):
  model.to(device)
  model.eval()
  batch_counter = 0
  with torch.no_grad():
    start_time = time.time()
    sum_pos_err = 0
    sum_neg_err = 0
    for batch_idx, (data, labels) in enumerate(data_loader):
        batch_counter+=1
        data, labels = data.to(device), labels.to(device)
        embeddings = model(data)
        _ = mining_func_err(embeddings, labels)
        _ = mining_func_no_Constraint(embeddings, labels)
       
        pos_err = mining_func_err.num_pos_pairs/mining_func_no_Constraint.num_pos_pairs
        neg_err = mining_func_err.num_neg_pairs/mining_func_no_Constraint.num_neg_pairs
        sum_pos_err+= pos_err
        sum_neg_err+=neg_err
        # loss = loss_func(embeddings, labels, indices_tuple)
        # pos_err = torch.tensor( loss['pos_loss']['losses'][loss['pos_loss']['losses']>0.5].shape).item()/torch.tensor(loss['pos_loss']['losses'].shape).item()
        # neg_err = torch.tensor( loss['neg_loss']['losses'][loss['neg_loss']['losses']>0.5].shape).item()/torch.tensor(loss['neg_loss']['losses'].shape).item()
        if batch_idx % 200 == 0:
            print("done {} batches".format(batch_idx))
    dist_pos = mining_func_no_Constraint.pos_pair_dist
    dist_neg = mining_func_no_Constraint.neg_pair_dist
    print("negative average distance:{} +- {}".format(dist_neg,mining_func_no_Constraint.neg_pair_dist_std))
    print("positive average distance:{} +- {}".format(dist_pos,mining_func_no_Constraint.pos_pair_dist_std))
    print("proccessing took: {} seconds".format(time.time()-start_time))
    print("EVAL : Epoch {} Iteration {}: positives_acc = {}, Number of mined positives = {}".format(epoch, batch_idx, 1-sum_pos_err/(batch_counter), mining_func_err.num_pos_pairs))
    print("EVAL : Epoch {} Iteration {}: negative_acc = {}, Number of mined negatives = {}".format(epoch, batch_idx, 1-sum_neg_err/(batch_counter), mining_func_err.num_neg_pairs))
    return 1-sum_pos_err/batch_counter , 1-sum_neg_err/batch_counter

def main():
  #Declaring GPU device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f'You are using {device} device')

  #Hyperparametes
  batch_size = 2048
  epochs = 20
  learning_rate = 0.003
  learning_rate_list = [0.0001, 0.0005, 0.001, 0.003, 0.005, 0.01]
  optimizer_type = "Adam"
  waveform_length_in_seconds = 3
  sample_rate = 16000
  threshold = 0.058

  #Get fairseq wav2vec model
  cp_path = '/home/Daniel/DeepProject/wav2vec/wav2vec_large.pt'
  model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
  model = model[0].to(device)
  model.eval()

  helper = sv_helper(model)
  #Get train & test datasets and DataLoaders
  train_data = SV_LIBRISPEECH('/home/Daniel/DeepProject/',
                                 folder_in_archive = FOLDER_IN_ARCHIVE_THREE_SEC_REPR_AVG, download=False)
  print(f'Number of training examples(utterances): {len(train_data)}')
  #my_train_loader = torch.utils.data.DataLoader(my_train_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

  test_data = SV_LIBRISPEECH('/home/Daniel/DeepProject/',
                                url = "test-clean",
                                folder_in_archive = FOLDER_IN_ARCHIVE_THREE_SEC_REPR_AVG_TEST,
                                download = False)
  print(f'Number of test examples(utterances): {len(test_data)}')
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

  net = Models.FC_SV()
  optimizer = optim.AdamW(net.parameters(), lr=learning_rate)  #need to check what the wav2vec2 paper did with the learning rate (i think it was frozen for big amount of steps and afterwards updated each step)
  print(net)
  ### pytorch-metric-learning stuff ###
  distance = distances.LpDistance(normalize_embeddings=True)
  reducer = reducers.ThresholdReducer(low = 0)
  #loss_func = losses.ContrastiveLoss(pos_margin=0, neg_margin=0.5)
  #loss_func = losses.TripletMarginLoss(margin = 0.2, distance = distance, reducer = reducer)
  #mining_func = miners.TripletMarginMiner(margin = 0.2, distance = distance, type_of_triplets = "semihard")
  #mining_func = miners.PairMarginMiner(pos_margin=0.2, neg_margin=0.8)
  #loss_func =  losses.ContrastiveLoss(pos_margin=0.2, neg_margin=0.8, distance = distance, reducer = reducer)
  ### train proccess
  loss_func = losses.ContrastiveLoss(pos_margin=0.01, neg_margin=0.05,reducer = reducer)
  train_mining_func = miners.PairMarginMiner(pos_margin=0.01, neg_margin=0.05)
  ### test proccess
  test_err_mining_func = miners.PairMarginMiner(pos_margin=threshold, neg_margin=threshold)
  test_no_constraint_mining_func = miners.PairMarginMiner(collect_stats=True,pos_margin=0., neg_margin=100.)
  ### pytorch-metric-learning stuff ###

  #TO DO: loop over different hyperparameters!!
  for epoch in range(epochs):
    start_train_time = time.time()
    train(net, loss_func, train_mining_func, device, train_loader, optimizer, epoch)
    print(f'Finished train epoch in {(time.time() - start_train_time):.2f}')
    pos_acc , neg_acc = evaluation(test_loader, net, test_err_mining_func, test_no_constraint_mining_func, device, epoch)
    print("test accuracy={}".format(0.5*(pos_acc + neg_acc)))


if __name__ == '__main__':
  main()
