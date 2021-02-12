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
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    walk_files,
)
from multiprocessing import set_start_method
import multiprocessing
# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass
#matplotlib
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


def evaluation(net, dataloader):
  '''This function will calculate the accuracy in gpu'''
  #keeping the network in evaluation mode 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  net.eval()
  total, correct = 0, 0
  with torch.no_grad():
    for data in dataloader:
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = net(inputs)
      _, pred = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (pred == labels).sum().item()
  net.train()
  return 100 * correct / total


def main():
  #Declaring GPU device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f'You are using {device} device')

  #Hyperparametes
  batch_size = 32
  epochs = 10
  learning_rate = 0.003
  optimizer_type = "Adam"
  waveform_length_in_seconds = 3
  sample_rate = 16000

  #Get fairseq wav2vec model
  cp_path = '/home/Daniel/DeepProject/wav2vec/wav2vec_large.pt'
  model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
  model = model[0].to(device)
  model.eval()

  wav_input_16khz = torch.randn(1,10000).to(device)
  z = model.feature_extractor(wav_input_16khz)
  c = model.feature_aggregator(z)
  print(f'input shape = {wav_input_16khz.shape} ; z shape = {z.shape} ; z shape = {c.shape}')

  helper = sv_helper(model)

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
  #Get train & test datasets and DataLoaders
  my_train_data = SV_LIBRISPEECH('/home/Daniel/DeepProject/',
                                 folder_in_archive = FOLDER_IN_ARCHIVE_THREE_SEC_REPR_AVG, download=False)
  print(f'Number of training examples(utterances): {len(my_train_data)}')
  #my_train_loader = torch.utils.data.DataLoader(my_train_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
  my_train_loader = torch.utils.data.DataLoader(my_train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

  my_test_data = SV_LIBRISPEECH('/home/Daniel/DeepProject/',
                                url = "test-clean",
                                folder_in_archive = FOLDER_IN_ARCHIVE_THREE_SEC_REPR_AVG_TEST,
                                download = False)
  print(f'Number of test examples(utterances): {len(my_test_data)}')
  my_test_loader = torch.utils.data.DataLoader(my_test_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

  net = Models.NAIVE_SV(1, helper.get_speakers_num('/home/Daniel/DeepProject/dataset/cut_train_data_360_repr'))
  criterion = nn.CrossEntropyLoss() #Combines nn.LogSoftmax() & nn.NLLLoss
  optimizer = optim.Adam(net.parameters(), lr=learning_rate)  #need to check what the wav2vec2 paper did with the learning rate (i think it was frozen for big amount of steps and afterwards updated each step)
  print(net)
  train_accuracy = torch.zeros(epochs)
  test_accuracy = torch.zeros(epochs)
  #Loop over epochs
  for epoch in range(epochs):
    time0 = time.time() #set timer
    #moving model to GPU & keeping the network in training mode
    net.to(device)
    net.train()
    #iterate through all the batches in each epoch
    for i, data in enumerate(my_train_loader, 0):
      #get inputs and labels and move them to gpu
      time1 = time.time()
      inputs, labels = data[0].to(device), data[1].to(device)
      #clear the gradients
      optimizer.zero_grad()
      #make a forward pass
      outputs = net(inputs)
      #calculate loss
      loss = criterion(outputs, labels)
      #backward pass
      loss.backward()
      #perform a single optimization step (parameter update)
      optimizer.step()
      if i % 500 == 0:
        print(i) 
    #Validation:
    print('***************validation**************')
    for i, data in enumerate(my_train_loader, 0):
      with torch.no_grad():
        net.eval()
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs) 
        #get the most likely label predicted
        _, top_class = outputs.topk(1, dim=1)
        #get number of matches with the real labels
        equals = top_class == labels.view(*top_class.shape)
        train_accuracy[epoch] += torch.sum(equals.type(torch.FloatTensor))
        if i % 500 == 0:
          print(i)

    test_acc = evaluation(net, my_test_loader)

    test_accuracy[epoch] = test_acc
    train_accuracy[epoch] = train_accuracy[epoch] / len(my_train_data)
    print(f'Epoch {epoch + 1}/{epochs} Accuracy: Train = {(train_accuracy[epoch] * 100):.3f}% , Test = {test_acc:.3f}%')
    #print(f'Epoch {epoch + 1}/{epochs} Accuracy: Train = {(train_accuracy[epoch] * 100):.3f}%')
    print(f'Finished pass through epoch in {time.time() - time0}[sec]')


if __name__ == '__main__':
  main()
