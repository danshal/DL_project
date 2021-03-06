import os, stat
import torch
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    walk_files,
)
from torch import nn
import numpy as np
import random
import argparse

data_path = '/home/Daniel/DeepProject/dataset/cut_test_full_repr/'
dst_data_path = f'/home/Daniel/DeepProject/dataset/cut_test_conv_repr/'
rootDir = '/home/Daniel/DeepProject/dataset/'

file_ext = '.pt'
walker = walk_files(data_path, suffix=file_ext, prefix=False, remove_suffix=True)
walker = list(walker)
current_speaker_id, _ = walker[0].split("-")
speaker_utterace_counter = 0
num_files_counter = 0
try:  
  os.mkdir(f'{dst_data_path}/{current_speaker_id}')
except OSError:
  print('ERROR delete current dataset folder')
  pass

for i in walker:
    speaker_id, _ = i.split("-")
    if speaker_id != current_speaker_id:
        #Got new speaker file
        current_speaker_id = speaker_id
        speaker_utterace_counter = 0
        num_files_counter = 0
        os.mkdir(f'{dst_data_path}/{speaker_id}')
        print(f'Start working on new speaker now - {current_speaker_id}')

    if num_files_counter < 300:
        file_repr = i + file_ext
        file_repr = os.path.join(data_path, speaker_id, file_repr)
        audio_repr = torch.load(file_repr)  #got shape of [1, 512, 298] in here
        for j in range(0,280,30):
            short_audio_repr = audio_repr[:, 0 + j:20 + j]
            torch.save(short_audio_repr, f'{dst_data_path}/{current_speaker_id}/{current_speaker_id}-{speaker_utterace_counter}.pt')
            num_files_counter += 1
            speaker_utterace_counter += 1
        #print(f'Finished processing 10 audio files of {speaker_id} speaker')

