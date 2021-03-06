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

# region Dataset Constants
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

#region Hyperparametes
batch_size = 1
learning_rate = 0.001
optimizer_type = 'Adam'
gpus_num = 1
momentum = 0.9
weight_decay = 0.0005
epochs = 100
#endregion

with open('/home/Daniel/DeepProject/dataset/speakers_map.json', 'r') as f:
    labels_map_dict = json.load(f)


save_folder = '/home/Daniel/DeepProject/classification/checkpoints'

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


    meta_data_set = my_meta_dataset('/home/Daniel/DeepProject/',
                                 folder_in_archive = FOLDER_IN_ARCHIVE_THREE_SEC_AUDIO, download=False, file_ext='.flac')
    print(f'Number of training examples(utterances): {len(meta_data_set)}')
    meta_testloader = DataLoader(meta_data_set,
                                    batch_size=batch_size, shuffle=True,
                                    num_workers=1)
                                     
    speakers_number = len(list(os.walk('/home/Daniel/DeepProject/dataset/cut_train_data_360_repr/')))
    print(speakers_number)
    # meta_valloader = DataLoader(my_meta_dataset(args=opt, partition='val',
    #                                             train_transform=train_trans,
    #                                             test_transform=test_trans,
    #                                             fix_seed=False),
    #                             batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
    # load model
    # on_top_model = Models.FC_SV()
    # loaded_checkpoint = torch.load('/home/Daniel/DeepProject/checkpoints/feature_extractor/eer_3_02_p05_n083.pt')
    # on_top_model.load_state_dict(loaded_checkpoint) 
    # model = Models.Wav2vecTuning(on_top_model)
    cp_path = '/home/Daniel/DeepProject/wav2vec/wav2vec_large.pt'
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    model = model[0]
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    # evalation
    start = time.time()
    val_acc, val_std = meta_test(model, meta_testloader)
    val_time = time.time() - start
    print('val_acc: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc, val_std,
                                                                  val_time))

    start = time.time()
    val_acc_feat, val_std_feat = meta_test(model, meta_valloader, use_logit=False)
    val_time = time.time() - start
    print('val_acc_feat: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc_feat,
                                                                       val_std_feat,
                                                                       val_time))

    start = time.time()
    test_acc, test_std = meta_test(model, meta_testloader)
    test_time = time.time() - start
    print('test_acc: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc, test_std,
                                                                    test_time))

    start = time.time()
    test_acc_feat, test_std_feat = meta_test(model, meta_testloader, use_logit=False)
    test_time = time.time() - start
    print('test_acc_feat: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc_feat,
                                                                         test_std_feat,
                                                                         test_time))


if __name__ == '__main__':
    main()