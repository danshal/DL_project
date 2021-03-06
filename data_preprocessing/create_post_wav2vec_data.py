''' This script will go throw cut audio files, get their representations using wav2vec,
   and afterwards will avgpool them to get [512, 1] representation that will be saved to filesystem '''
import os
import fairseq
import torch
import torchaudio
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    walk_files,
)
from torch import nn
from helper_functions import sv_helper
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#create parser
my_parser = argparse.ArgumentParser(description='Get audio representations of data_path audio files using wav2vec network. Pay Attention: Your data and dst paths have to be under /home/Daniel/DeepProject/dataset/')
#add arguments
my_parser.add_argument('data_path', metavar='data_path', type=str, help='path to audio files to be represented')
my_parser.add_argument('dst_path', metavar='dst_path', type=str, help='path to save audio representations')
my_parser.add_argument('--avg', default=False, action='store_true', help='boolean flag. When true -> save avg on all time axis to get representations shape of [512,1]')
my_parser.add_argument('--max', default=False, action='store_true', help='boolean flag. When true -> save max on all time axis to get representations shape of [512,1]')
my_parser.add_argument('--conv', default=False, action='store_true', help='boolean flag. When true -> save 20 time vectors to get representations shape of [512,20]')

args = my_parser.parse_args()
args = vars(args)

#cp_path = '/home/Daniel/DeepProject/wav2vec/wav2vec_large.pt'
cp_path = '/../wav2vec_large.pt'
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
model = model[0].to(device)
model.eval()

helper = sv_helper(model)

# data_path = '/home/Daniel/DeepProject/dataset/cut_train_data_360/'
# dst_data_path = '/home/Daniel/DeepProject/dataset/cut_train_data_360_full_repr'
data_path = args['data_path']
dst_data_path = args['dst_path']
is_avgpool = args['avg']
is_max = args['max']
is_conv = args['conv']
print(is_avgpool)
print(is_max)
print(is_conv)
rootDir = '/home/Daniel/DeepProject/dataset/'
file_ext = '.flac'
walker = walk_files(data_path, suffix=file_ext, prefix=False, remove_suffix=True)
walker = list(walker)
current_speaker_id, _ = walker[0].split("-")
speaker_utterace_counter = 0
speaker_counter = 0

try:
  if not os.path.exists(f'{dst_data_path}/{speaker_counter}'):
    print('making directory!')
    os.mkdir(f'{dst_data_path}/{speaker_counter}')
except OSError:
  print('ERROR delete current dataset folder')
  pass

print_flag = True

for i in walker:
    speaker_id, utterance_id = i.split("-")
    if speaker_id != current_speaker_id:
        print(f'Finished working on speaker number {speaker_counter}')
        speaker_utterace_counter = 0
        speaker_counter += 1
        current_speaker_id = speaker_id
        os.mkdir(f'{dst_data_path}/{speaker_counter}')
        print(f'Starting working on speaker number {speaker_counter}')
    file_audio_id = i + file_ext
    file_audio = os.path.join(data_path, speaker_id, file_audio_id)
    waveform, _ = torchaudio.load(file_audio) # Load audio - dont care about sample rate
    waveform = waveform.to(device)
    audio_repr = helper.get_tensor_repr(waveform)
    if is_avgpool == True:
      pool_operation = nn.AvgPool1d(audio_repr.shape[2])
      pool_repr = pool_operation(audio_repr)
    elif is_max == True:
      pool_operation = nn.MaxPool1d(audio_repr.shape[2])
      pool_repr = pool_operation(audio_repr)
    elif is_conv == True:
      pool_operation = nn.AvgPool1d(audio_repr.shape[2] - 20 + 1, 1)
      pool_repr = pool_operation(audio_repr)
    else:
      pool_repr = torch.squeeze(audio_repr)
    pool_repr = torch.squeeze(pool_repr)
    torch.save(pool_repr, f'{dst_data_path}/{speaker_counter}/{speaker_counter}-{speaker_utterace_counter}.pt')
    speaker_utterace_counter+=1
