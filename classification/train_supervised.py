from __future__ import print_function

import os
import argparse
import socket
import time
import sys

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

from util import adjust_learning_rate, accuracy, AverageMeter
from eval.meta_eval import meta_test
from eval.cls_eval import validate
import os
from datetime import datetime
import json


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

#region Hyperparametes
batch_size = 64
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

    # set the path according to the environment
    # if not opt.model_path:
    #     opt.model_path = './models_pretrained'
    # if not opt.tb_path:
    #     opt.tb_path = './tensorboard'
    # if not opt.data_root:
    #     opt.data_root = './data/{}'.format(opt.dataset)
    # else:
    #     opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    # opt.data_aug = True

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # if opt.cosine:
    #     opt.model_name = '{}_cosine'.format(opt.model_name)

    # opt.model_name = '{}_trial_{}'.format(opt.model_name, opt.trial)

    # opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    # if not os.path.isdir(opt.tb_folder):
    #     os.makedirs(opt.tb_folder)

    save_folder = '/home/Daniel/DeepProject/classification/checkpoints'
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)


    return opt


def main():

    opt = parse_option()

    dataset = SV_LIBRISPEECH('/home/Daniel/DeepProject/',
                                 folder_in_archive = FOLDER_IN_ARCHIVE_THREE_SEC_AUDIO, download=False, file_ext='.flac')
    print(f'Number of training examples(utterances): {len(dataset)}')

    # val_data = SV_LIBRISPEECH('/home/Daniel/DeepProject/',
    #                             url = "test-clean",
    #                             folder_in_archive = FOLDER_IN_ARCHIVE_THREE_SEC_REPR_AVG_TEST,
    #                             download = False)
    # print(f'Number of test examples(utterances): {len(val_data)}')
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_data, val_data = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    speakers_number = len(list(os.walk('/home/Daniel/DeepProject/dataset/cut_train_data_360_repr/')))
    print(speakers_number)
    
    

    # model
    #model = create_model(opt.model, n_cls, opt.dataset)
    # model = Models.FC_SV(speakers_number)
    on_top_model = Models.FC_SV(speakers_number)
    loaded_checkpoint = torch.load('/home/Daniel/DeepProject/classification/checkpoints/99_val_acc_checkpoints/2021_02_25-07_25_30_AM_last_classification_99_acc.pth')
    on_top_model.load_state_dict(loaded_checkpoint['model'])
    model = Models.Wav2vecTuning(on_top_model)
    # optimizer
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=0.0005)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate,
                              momentum=momentum,
                              weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        if gpus_num > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir='/home/Daniel/DeepProject/classification/classification_summary', flush_secs=2)

    # set cosine annealing scheduler
    if opt.cosine:
        eta_min = learning_rate * (opt.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min, -1)

    # routine: supervised pre-training
    for epoch in range(1, epochs + 1):

        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer, learning_rate)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict() if gpus_num <= 1 else model.module.state_dict(),
            }
            save_file = os.path.join(save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # save the last model
    state = {
        'opt': opt,
        'model': model.state_dict() if gpus_num <= 1 else model.module.state_dict(),
    }
    cur_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    save_file = os.path.join(save_folder, '{}_last.pth'.format(cur_time))
    torch.save(state, save_file)


def change_labels(labels):
    for i, val in enumerate(labels, 0):
        labels[i] = torch.tensor(int(labels_map_dict[str(val.item())]))
    return labels


def train(epoch, train_loader, model, criterion, optimizer):
    """One epoch training"""
    model.train()
    model.cuda()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        data = data.float()
        target = change_labels(target)
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(data)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        top5.update(acc5[0], data.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


if __name__ == '__main__':
    main()