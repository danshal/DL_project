import os, stat
import torch
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    walk_files,
)
from torch import nn
import numpy as np
model = 'FC'# GRU or FC
data_path = '/home/Daniel/DeepProject/dataset/cut_train_data_360_full_repr/'
avg_data_path = f'/home/Daniel/DeepProject/dataset/cut_train_data_360_max_pool_repr/'
rootDir = '/home/Daniel/DeepProject/dataset/'
try:  
  os.mkdir(avg_data_path)
  #os.chmod(f'{file_path}/{speaker_counter}', stat.S_IRWXG)
except OSError:
  print('ERROR delete current dataset folder')
  pass
file_ext = '.pt'
walker = walk_files(data_path, suffix=file_ext, prefix=False, remove_suffix=True)
walker = list(walker)
current_speaker_id, _ = walker[0].split("-")
speaker_utterace_counter = 0
speaker_counter = 0
file_path = f'{rootDir}cut_train_data_360_max_pool_repr'
print(f'saving data to {file_path}')
try:  
  os.mkdir(f'{file_path}/{speaker_counter}')
  #os.chmod(f'{file_path}/{speaker_counter}', stat.S_IRWXG)
except OSError:
  print('ERROR delete current dataset folder')
  pass


for i in walker:
  speaker_id, utterance_id = i.split("-")
  if speaker_id != current_speaker_id:
    speaker_utterace_counter = 0
    speaker_counter += 1
    current_speaker_id = speaker_id
    os.mkdir(f'{file_path}/{speaker_counter}')
    print(f'Finished on {speaker_counter} speaker')
  file_repr = i + file_ext
  file_repr = os.path.join(data_path, speaker_id, file_repr)
  audio_repr = torch.load(file_repr)
  audio_repr = torch.unsqueeze(audio_repr,0)
  pooling = nn.MaxPool1d(audio_repr.shape[2])
  avg_repr = pooling(audio_repr)
  avg_repr = torch.squeeze(avg_repr)
  torch.save(avg_repr, f'{file_path}/{speaker_counter}/{speaker_counter}-{speaker_utterace_counter}.pt')
  speaker_utterace_counter+=1
  