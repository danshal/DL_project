from __future__ import print_function

import os
import argparse
import socket
import time
import sys
import fairseq
import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

#from models import model_pool
#from models.util import create_model
# import sys
# sys.path.append('..')
import Models
from sv_dataset import SV_LIBRISPEECH
from sv_dataset import SV_LIBRISPEECH_PAIRS
# from sv_dataset import my_meta_dataset
from sv_dataset import my_meta_dataset

from util import adjust_learning_rate, accuracy, AverageMeter
from eval.meta_eval import meta_test
from eval.cls_eval import validate
import os
from datetime import datetime
import json
import numpy as np
import argparse

my_parser = argparse.ArgumentParser(description='Check different models of ours in one shot setup')
my_parser.add_argument('my_model', metavar='data_path', type=str, help='Options are: Conv, FC, FC_TUNING, classification, classification_TUNING, class_distill')


args = my_parser.parse_args()
args = vars(args)

model_name = args['my_model']

#set for REPRODUCIBILITY
torch.manual_seed(42)
np.random.seed(42)

# region Dataset Constants
rootDir = '/home/Daniel/DeepProject/dataset'
URL = "train-clean-100"
FOLDER_IN_ARCHIVE_THREE_SEC_REPR = "cut_train_data_360_full_repr"
FOLDER_IN_ARCHIVE_THREE_SEC_REPR_AVG = "cut_train_data_360_repr"
FOLDER_IN_ARCHIVE_THREE_SEC_AUDIO = "cut_train_data_360"
FOLDER_IN_ARCHIVE_THREE_SEC_REPR_TEST = "cut_test_full_repr"
FOLDER_IN_ARCHIVE_THREE_SEC_REPR_AVG_TEST = "cut_test_repr"
FOLDER_IN_ARCHIVE_THREE_SEC_AUDIO_TEST = "cut_test-clean"
FOLDER_IN_ARCHIVE_THREE_SEC_AUDIO_NEW_TEST = "cut_test_speakers"
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

#region Hyperparametes
batch_size = 1
learning_rate = 0.001
optimizer_type = 'Adam'
gpus_num = 1
momentum = 0.9
weight_decay = 0.0005
epochs = 100
#endregion

with open('speakers_map.json', 'r') as f:
    labels_map_dict = json.load(f)


save_folder = 'checkpoints'

def parse_option():

    parser = argparse.ArgumentParser('arguments for training')

    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    # optimization
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    # osine annealingc
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    return opt

def main():


    meta_data_train_set = my_meta_dataset('',
                                 folder_in_archive = FOLDER_IN_ARCHIVE_THREE_SEC_AUDIO, download=False, file_ext='.flac')
    print(f'Number of training examples(utterances): {len(meta_data_train_set)}')
    meta_trainloader = DataLoader(meta_data_train_set,
                                    batch_size=batch_size, shuffle=True,
                                    num_workers=4)
    meta_data_test_set = my_meta_dataset('',
                                 folder_in_archive = FOLDER_IN_ARCHIVE_THREE_SEC_AUDIO_NEW_TEST, download=False, file_ext='.flac')
    print(f'Number of training examples(utterances): {len(meta_data_test_set)}')
    meta_testloader = DataLoader(meta_data_test_set,
                                    batch_size=batch_size, shuffle=True,
                                    num_workers=4)
                                     
    speakers_number = len(list(os.walk('dataset/cut_train_data_360_repr/')))
    print(speakers_number)
   
    #model_name = 'class_distill'# Conv, FC, FC_TUNING, classification, classification_TUNING, class_distill
    # load model
    if model_name == 'Conv':
        on_top_model = Models.ConvNet()
        loaded_checkpoint = torch.load('/home/Daniel/DeepProject/checkpoints/feature_extractor/conv_4_1.pt') # no fine tune
        on_top_model.load_state_dict(loaded_checkpoint['state_dict']) 
        model = Models.Wav2vecTuning(on_top_model, True)
    elif model_name == 'FC':
        on_top_model = Models.FC_SV_TUNING(512)
        loaded_checkpoint = torch.load('/home/Daniel/DeepProject/checkpoints/feature_extractor/eer_3_02_p05_n083.pt')
        on_top_model.load_state_dict(loaded_checkpoint)
        model = Models.Wav2vecTuning(on_top_model)
    elif model_name == 'FC_TUNING':
        on_top_model = Models.FC_SV_TUNING(512)
        loaded_checkpoint = torch.load('/home/Daniel/DeepProject/checkpoints/fine_tuning/checkpoint_2021_02_23-11_10_55_PM_2.7.pt')
        model = Models.Wav2vecTuning(on_top_model)
        model.load_state_dict(loaded_checkpoint['state_dict'])
    elif model_name == 'classification':
        on_top_model = Models.FC_SV_CLASSIFICATION(speakers_number)
        loaded_checkpoint = torch.load('/home/Daniel/DeepProject/checkpoints/classification/2021_02_25-07_25_30_AM_last_classification_99_acc.pth')
        on_top_model.load_state_dict(loaded_checkpoint['model'])
        on_top_model.fc = nn.Sequential(*[on_top_model.fc[0]])
        model = Models.Wav2vecTuning(on_top_model)
    elif model_name == 'classification_TUNING':
        on_top_model = Models.FC_SV_CLASSIFICATION(speakers_number)
        loaded_checkpoint = torch.load('/home/Daniel/DeepProject/classification/checkpoints/ckpt_epoch_80.pth')
        model = Models.Wav2vecTuning(on_top_model)
        model.load_state_dict(loaded_checkpoint['model'])
        model.fc = nn.Sequential(*[on_top_model.fc[0]])
    elif model_name == 'class_distill':
        on_top_model = Models.FC_SV_CLASSIFICATION(speakers_number)
        loaded_checkpoint = torch.load('/home/Daniel/DeepProject/classification/checkpoints/distill_class_tuning/distill_ckpt_epoch_20.pth')
        model = Models.Wav2vecTuning(on_top_model)
        model.load_state_dict(loaded_checkpoint['model'])
        model.fc = nn.Sequential(*[on_top_model.fc[0]])
        #do classification + distillation stuff -> remove last layer and check performence
       
    # cp_path = '/home/Daniel/DeepProject/wav2vec/wav2vec_large.pt'
    # model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    # model = model[0]
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    # evaluation
    start = time.time()
    classifier_type = 'Proto'
    val_acc, val_std = meta_test(model, meta_trainloader, classifier=classifier_type)
    # val_acc, val_std = meta_test(model, meta_trainloader)
    val_time = time.time() - start
    print('val_acc: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc, val_std,
                                                                  val_time))

    start = time.time()
    test_acc, test_std = meta_test(model, meta_testloader, classifier=classifier_type)
    # test_acc, test_std = meta_test(model, meta_testloader)
    test_time = time.time() - start
    print('test_acc: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc, test_std,
                                                                    test_time))


if __name__ == '__main__':
    main()