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
from distill.util import Embed
from distill.criterion import DistillKL, NCELoss, Attention, HintLoss
from util import adjust_learning_rate, accuracy, AverageMeter
from eval.meta_eval import meta_test
from eval.cls_eval import validate
import Models
from sv_dataset import SV_LIBRISPEECH
import json
from torch.utils.tensorboard import SummaryWriter


base_summary_path = '/home/Daniel/DeepProject/classification'
writer_path = f'{base_summary_path}/distillation_summary'
writer = SummaryWriter(writer_path)


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--eval_freq', type=int, default=10, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='2,4,6,8,10', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'contrast', 'hint', 'attention'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=0.5, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0.5, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.5, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # specify folder
    parser.add_argument('--model_path', type=str, default='', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='', help='path to tensorboard')
    parser.add_argument('--data_root', type=str, default='', help='path to data root')

    # setting for meta-learning
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


FOLDER_IN_ARCHIVE_THREE_SEC_AUDIO = "cut_train_data_360"
with open('/home/Daniel/DeepProject/dataset/speakers_map.json', 'r') as f:
    labels_map_dict = json.load(f)

def main():
    best_acc = 0

    opt = parse_option()
    batch_size = 64
    learning_rate = 0.001
    # tensorboard logger
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    dataset = SV_LIBRISPEECH('/home/Daniel/DeepProject/',
                                 folder_in_archive = FOLDER_IN_ARCHIVE_THREE_SEC_AUDIO, download=False, file_ext='.flac')
    print(f'Number of training examples(utterances): {len(dataset)}')
    split_per = 0.9
    # train_data, val_data = torch.utils.data.random_split(dataset, [int(len(dataset) * split_per), len(dataset) - int(len(dataset) * split_per)])
    train_data, val_data = torch.utils.data.random_split(dataset, [int(len(dataset) * split_per), len(dataset) - int(len(dataset) * split_per)])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    n_data = len(train_loader)

    
    # model
    speakers_number = len(list(os.walk('/home/Daniel/DeepProject/dataset/cut_train_data_360_repr/')))
    on_top_model = Models.FC_SV(speakers_number)
    # on_top_net.load_state_dict(torch.load('/home/Daniel/DeepProject/checkpoints/feature_extractor/eer_3_02_p05_n083.pt'))
    #net = Models.Wav2vecTuning(Models.FC_SV)
    #net = Models.Wav2vecTuning(Models.FC_SV_TUNING(512))
    #net = Models.Wav2vecTuning(on_top_model)
    #checkpoint = torch.load('/home/Daniel/DeepProject/checkpoints/fine_tuning/checkpoint_2021_02_23-11_10_55_PM_2.7.pt')
    #checkpoint = torch.load('/home/Daniel/DeepProject/classification/checkpoints/ckpt_epoch_100.pth')
    #net.load_state_dict(checkpoint['model'])

    on_top_model = Models.FC_SV(speakers_number)
    #loaded_checkpoint = torch.load('/home/Daniel/DeepProject/classification/checkpoints/99_val_acc_checkpoints/2021_02_25-07_25_30_AM_last_classification_99_acc.pth')
    #on_top_model.load_state_dict(loaded_checkpoint['model'])
    checkpoint = torch.load('/home/Daniel/DeepProject/classification/checkpoints/ckpt_epoch_90.pth')
    net = Models.Wav2vecTuning(on_top_model)
    net.load_state_dict(checkpoint['model'])

    model_t = net
    model_s = Models.Wav2vecTuning(Models.FC_SV(speakers_number))

    #data = torch.randn(2, 3, 84, 84)
    data = torch.randn(2, 1, 48000)
    model_t.eval()
    model_s.eval()
    # feat_t, _ = model_t(data, is_feat=True)
    # feat_s, _ = model_s(data, is_feat=True)
    feat_t, _ = model_t(data)
    feat_s, _ = model_s(data)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'contrast':
        criterion_kd = NCELoss(opt, n_data)
        embed_s = Embed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = Embed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    # teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls)
    print('teacher accuracy: ', teacher_acc)

    # set cosine annealing scheduler
    if opt.cosine:
        eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)

    # routine: supervised model distillation
    for epoch in range(1, opt.epochs + 1):

        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer, learning_rate)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss, loss_class, loss_diver = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_classification_loss', loss_class, epoch)
        writer.add_scalar('train_divergence_loss', loss_diver, epoch)
        # logger.log_value('train_acc', train_acc, epoch)
        # logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls)

        writer.add_scalar('test_acc', test_acc, epoch)
        writer.add_scalar('test_acc_top5', test_acc_top5, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)
        # logger.log_value('test_acc', test_acc, epoch)
        # logger.log_value('test_acc_top5', test_acc_top5, epoch)
        # logger.log_value('test_loss', test_loss, epoch)

        # regular saving
        if epoch % 2 == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
            }
            save_file = os.path.join('/home/Daniel/DeepProject/classification/checkpoints', 'distill_ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # save the last model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join('/home/Daniel/DeepProject/classification/checkpoints', '{}_last_distill.pth'.format(model_s))
    torch.save(state, save_file)


def change_labels(labels):
    for i, val in enumerate(labels, 0):
        labels[i] = torch.tensor(int(labels_map_dict[str(val.item())]))
    return labels


def train(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch training"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_cls = AverageMeter()
    losses_div = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (data, target) in enumerate(train_loader):
        # if opt.distill in ['contrast']:
        #     input, target, index, contrast_idx = data
        # else:
        #     input, target, index = data
        data_time.update(time.time() - end)

        data = data.float()
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
            target = change_labels(target)
            # index = index.cuda()
            # if opt.distill in ['contrast']:
            #     contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound', 'overhaul']:
            preact = True
        # feat_s, logit_s = model_s(data, is_feat=True)
        logit_s = model_s(data)
        with torch.no_grad():
            # feat_t, logit_t = model_t(input, is_feat=True)
            logit_t = model_t(data)
            # feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        # elif opt.distill == 'contrast':
        #     f_s = module_list[1](feat_s[-1])
        #     f_t = module_list[2](feat_t[-1])
        #     loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        # elif opt.distill == 'hint':
        #     f_s = feat_s[-1]
        #     f_t = feat_t[-1]
        #     loss_kd = criterion_kd(f_s, f_t)
        # elif opt.distill == 'attention':
        #     g_s = feat_s[1:-1]
        #     g_t = feat_t[1:-1]
        #     loss_group = criterion_kd(g_s, g_t)
        #     loss_kd = sum(loss_group)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        losses_cls.update(loss_cls.item(), data.size(0))
        losses_div.update(loss_div.item(), data.size(0))
        
        top1.update(acc1[0], data.size(0))
        top5.update(acc5[0], data.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} TOTAL LOSS : {loss.avg:.3f} CLASSIFICATION LOSS: {loss_cls.avg:.3f} DIVERGANCE LOSS: {loss_div.avg:.3f}'
          .format(top1=top1, top5=top5 , loss=losses , loss_cls = losses_cls , loss_div = losses_div))

    return top1.avg, losses.avg, losses_cls.avg, losses_div.avg


if __name__ == '__main__':
    main()