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


def train(sv_model,backbone_model, loss_func, mining_func, device, train_loader, sv_optimizer, wav2vec_optimizer, epoch):
  sv_model.to(device)
  backbone_model.to(device)
  backbone_model.train()
  sv_model.train()
  max_pool = nn.MaxPool1d(wav2vec_output_dims[1])
  for batch_idx, (data, labels) in enumerate(train_loader):
     data, labels = data.to(device), labels.to(device)
     optimizer.zero_grad()
     back_bone_embeddings = backbone_model(data)
     embeddings = max_pool(back_bone_embeddings)
     sv_embeddings = sv_model(embeddings)
     indices_tuple = mining_func(sv_embeddings, labels)
     loss = loss_func(sv_embeddings, labels, indices_tuple)
     loss.backward()
     wav2vec_optimizer.step()# i checked in the pml code, it seems the order of the steps doesnt matter
     sv_optimizer.step()
     if batch_idx % 20 == 0:
       print("Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(epoch, batch_idx, loss, mining_func.num_triplets))


### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def test(train_set, test_set, sv_model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(test_embeddings, 
                                                train_embeddings,
                                                np.squeeze(test_labels),
                                                np.squeeze(train_labels),
                                                False)
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))


def main():
  #Declaring GPU device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f'You are using {device} device')

  #Hyperparametes
  batch_size = 256
  epochs = 20
  learning_rate = 0.003
  optimizer_type = "Adam"
  waveform_length_in_seconds = 3
  sample_rate = 16000

  #Get fairseq wav2vec model
  cp_path = '/home/Daniel/DeepProject/wav2vec/wav2vec_large.pt'
  model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
  model = model[0].to(device)
  model.eval()
  wav2vec_output_dims=[512 , 298]
  helper = sv_helper(model)

  # wav_input_16khz = torch.randn(1,10000).to(device)
  # z = model.feature_extractor(wav_input_16khz)
  # c = model.feature_aggregator(z)
  # print(f'input shape = {wav_input_16khz.shape} ; z shape = {z.shape} ; z shape = {c.shape}')


  #Get train & test datasets and DataLoaders
  train_data = SV_LIBRISPEECH('/home/Daniel/DeepProject/',
                                 folder_in_archive = FOLDER_IN_ARCHIVE_THREE_SEC_REPR_AVG, download=False)
  print(f'Number of training examples(utterances): {len(train_data)}')
  #my_train_loader = torch.utils.data.DataLoader(my_train_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

  test_data = SV_LIBRISPEECH('/home/Daniel/DeepProject/',
                                url = "test-clean",
                                folder_in_archive = FOLDER_IN_ARCHIVE_THREE_SEC_REPR_AVG_TEST,
                                download = False)
  print(f'Number of test examples(utterances): {len(test_data)}')
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

  net = Models.NAIVE_SV(1, helper.get_speakers_num('/home/Daniel/DeepProject/dataset/cut_train_data_360_repr'))
  sv_optimizer = optim.Adam(net.parameters(), lr=learning_rate)  #need to check what the wav2vec2 paper did with the learning rate (i think it was frozen for big amount of steps and afterwards updated each step)
  wav2vec_optimizer = optim.Adam(model.parameters())
  print(net)
  ### pytorch-metric-learning stuff ###
  distance = distances.CosineSimilarity()
  reducer = reducers.ThresholdReducer(low = 0)
  loss_func = losses.TripletMarginLoss(margin = 0.2, distance = distance, reducer = reducer)
  mining_func = miners.TripletMarginMiner(margin = 0.2, distance = distance, type_of_triplets = "semihard")
  accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), avg_of_avgs = True, k = None)
  ### pytorch-metric-learning stuff ###

  for epoch in range(epochs):
    start_train_time = time.time()
    train(net, wav2vec , loss_func, mining_func, device, train_loader, sv_optimizer, wav2vec_optimizer, epoch)
    print(f'Finished train epoch in {(time.time() - start_train_time):.2f}')
    test(train_data, test_data, net, accuracy_calculator)


if __name__ == '__main__':
  main()
