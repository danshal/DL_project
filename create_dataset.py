import os, stat
import torch
import torchaudio
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    walk_files,
)
from torch import nn
import argparse

#create parser
my_parser = argparse.ArgumentParser(description='Cut audio files in src_path and place them in dst_path. Pay Attention: Your src and dst paths have to be under /home/Daniel/DeepProject/dataset/')
#add arguments
my_parser.add_argument('src_path', metavar='src_path', type=str, help='path to audio files to be cut')
my_parser.add_argument('dst_path', metavar='dst_path', type=str, help='path to save cut audio files')
my_parser.add_argument('window_len_sec', default=3, metavar='window_size', type=int, help='windows size to cut with in seconds')

args = my_parser.parse_args()
args = vars(args)
window_len_sec = args['window_len_sec']


waveform_length_in_samples = window_len_sec * 16000
data_path = args['src_path']
cut_data_path = args['dst_path']
# file_path_ext = data_path.split('/')
# print(file_path_ext)
# dataset_word_index = [i for i, x in enumerate(file_path_ext) if x == 'dataset']
# print(dataset_word_index)
# seperator = '/'
# file_path_ext = seperator.join(file_path_ext[dataset_word_index[0]+1:])
#data_path = '/home/Daniel/DeepProject/dataset/LibriSpeech/train-clean-360'
#cut_data_path = '/home/Daniel/DeepProject/dataset/cut_train_data_360/'
rootDir = '/home/Daniel/DeepProject/dataset/'
try:  
  os.mkdir(cut_data_path)
  #os.chmod(f'{file_path}/{speaker_counter}', stat.S_IRWXG)
except OSError:
  print('ERROR delete current dataset folder')
  pass

file_ext = '.flac'
walker = walk_files(data_path, suffix=file_ext, prefix=False, remove_suffix=True)
walker = list(walker)
current_speaker_id, _, _ = walker[0].split("-")
speaker_utterace_counter = 0
#file_path = f'{rootDir}{file_path_ext}'
try:  
  os.mkdir(f'{cut_data_path}/{current_speaker_id}')
except OSError:
  print('ERROR delete current dataset folder')
  pass


for i in walker:
    speaker_id, chapter_id, utterance_id = i.split("-")
    if speaker_id != current_speaker_id:
        speaker_utterace_counter = 0
        current_speaker_id = speaker_id
        os.mkdir(f'{cut_data_path}/{current_speaker_id}')
        print(f'Finished on {current_speaker_id} speaker')
    file_audio_id = i + file_ext
    file_audio = os.path.join(data_path, speaker_id,chapter_id, file_audio_id)
    audio = torchaudio.load(file_audio)
    waveform, _ = torchaudio.load(file_audio) # Load audio - dont care about sample rate
    waveform_length = waveform.shape[1]
    if waveform_length<waveform_length_in_samples:
        continue
    chunks = [waveform[0,x:x+waveform_length_in_samples] for x in range(0, waveform_length, waveform_length_in_samples)]
    num_of_chunks = waveform_length//waveform_length_in_samples
    for chunk_ind in range(0,num_of_chunks-1):
     torchaudio.backend.sox_backend.save(f'{cut_data_path}/{current_speaker_id}/{current_speaker_id}-{speaker_utterace_counter}.flac',chunks[chunk_ind], 16000)
     speaker_utterace_counter+=1