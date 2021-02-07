#General imports
import argparse
import glob
import os
import os.path as osp
import pprint
import soundfile as sf
from typing import Tuple
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

batch_size =  32#@param {type:"integer"}
hidden_units = 512 #@param {type:"integer"}
p_dropout = 0.45 #@param {type:"slider", min:0, max:1, step:0.1}
epochs_no_drop = 13 #@param {type:"integer"}
epochs_with_drop = 40 #@param {type:"integer"}
learning_rate =  1#@param {type:"integer"}
clipping_threshold_lstm =  5#@param {type:"integer"}
clipping_threshold_gru =  0.25#@param {type:"number"}
seq_num =  20#@param {type:"integer"}
layers_num =  2#@param {type:"integer"}
weight_uniform_init = 0.1 #@param {type:"slider", min:0, max:1, step:0.005}

cp_path = cp_path = '/home/orixyz/DeepProject/wav2vec/wav2vec_large.pt'
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
model = model[0]
model.eval()
wav_input_16khz = torch.randn(1,10000)
z = model.feature_extractor(wav_input_16khz)
c = model.feature_aggregator(z)
print(f'input shape = {wav_input_16khz.shape} ; z shape = {z.shape} ; c shape = {c.shape}')