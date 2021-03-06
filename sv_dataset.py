import torch
import torchaudio
import os
import numpy as np
from typing import Tuple
from torch import Tensor
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    walk_files,
)
import random

FOLDER_IN_ARCHIVE_THREE_SEC_AUDIO = "SV_Librispeech_Dataset"
FOLDER_IN_ARCHIVE_ORIGINAL_LIBRI = "LibriSpeech"
FOLDER_IN_ARCHIVE_THREE_SEC_REPR = "cut_train_data_360_full_repr"
transform = transforms.Normalize((0.5), (0.5))


def load_sv_librispeech_item(fileid: str, path: str, ext: str = '.pt') -> Tuple[Tensor, int]:
    '''This function return example as wav2vec representation or librispeech audio
       representation from a given path'''
    speaker_id, utterance_id = fileid.split("-")
    file_to_load = fileid + ext
    file_path = os.path.join(path, speaker_id, file_to_load)
    if ext == '.flac':
      input, _ = torchaudio.load(file_path) # Load audio - dont care about sample rate
    else:
      input = torch.load(file_path)
    #transform_input = torch.squeeze(torch.transpose(transform(torch.unsqueeze(torch.unsqueeze(input,0),2)),0,1))
    return (input, int(speaker_id))


# create my own collate function to avoid stacking same size examples in a batch
def my_collate(batch):
    data = [item[0] for item in batch]  #get 512 X ? tensor in item[0]
    data = tuple(data)
    data = torch.cat(data)
    data = data.detach().clone()
    target = [item[1] for item in batch]        
    target = torch.LongTensor(target)
    return [data, target]


class SV_LIBRISPEECH(Dataset):
    """Create a Dataset for SV_LibriSpeech.

    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``,
            ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"`` and
            ``"train-other-500"``. (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriSpeech"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _ext_audio = ".flac"
    _ext_repr = ".pt"

    def __init__(self, root: str, url: str = "train-clean-360",
                 folder_in_archive: str = "cut_train_data_360_repr",
                 download: bool = False, is_SV: bool = True, wav2vec_fine_tuning: bool = False , file_ext='.pt') -> None:

        if url in [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]:
            ext_archive = ".tar.gz"
            base_url = "http://www.openslr.org/resources/12/"

            url = os.path.join(base_url, url + ext_archive)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(".")[0]
        if is_SV == False:
          folder_in_archive = os.path.join(folder_in_archive, basename)
        folder_in_dataset = os.path.join('dataset/',folder_in_archive)
        self._path = os.path.join(root, folder_in_dataset)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url(url, root, hash_value=checksum)
                extract_archive(archive)
        
        self._ext = file_ext
        walker = walk_files(self._path, suffix=self._ext, prefix=False, remove_suffix=True)
        self._walker = list(walker)
        self.ft = wav2vec_fine_tuning
  

    def __getitem__(self, n: int) -> Tuple[Tensor, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, speaker_id)``
        """
        fileid = self._walker[n]  #get file name without .flac suffix        
        #librispeech_item, speaker_id = load_sv_librispeech_item(fileid, self._path, self._ext_audio)
        return load_sv_librispeech_item(fileid, self._path, self._ext)

    def __len__(self) -> int:
      return len(self._walker)



class SV_LIBRISPEECH_PAIRS(Dataset):
    """Create a Dataset for SV_LibriSpeech while forcing bigger amount of positive pairs in a batch.

    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``,
            ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"`` and
            ``"train-other-500"``. (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriSpeech"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _ext_audio = ".flac"
    _ext_repr = ".pt"

    def __init__(self, root: str, url: str = "train-clean-360",
                 folder_in_archive: str = "cut_train_data_360_repr",
                 download: bool = False, is_SV: bool = True, wav2vec_fine_tuning: bool = False , file_ext='.pt') -> None:

        if url in [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]:
            ext_archive = ".tar.gz"
            base_url = "http://www.openslr.org/resources/12/"

            url = os.path.join(base_url, url + ext_archive)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(".")[0]
        if is_SV == False:
          folder_in_archive = os.path.join(folder_in_archive, basename)
        folder_in_dataset = os.path.join('dataset/',folder_in_archive)
        self._path = os.path.join(root, folder_in_dataset)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url(url, root, hash_value=checksum)
                extract_archive(archive)
        
        self._ext = file_ext
        walker = walk_files(self._path, suffix=self._ext, prefix=False, remove_suffix=True)
        self._walker = list(walker)
        self._last_n = -1
        self._state = 1 #1 -> positive & counter = 0 ;  2 -> positive & counter = 1
  

    def __getitem__(self, n: int) -> Tuple[Tensor, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, speaker_id)``
        """
        #handle state machine
        if self._state == 1:
            fileid = self._walker[n]  #get file name without .flac suffix
            self._last_n = n
            self._state = 2
        elif self._state == 2:
            jump = int(random.uniform(1, 30))
            if self._last_n + jump >= len(self._walker):
                fileid = self._walker[self._last_n - jump]  #get file name without .flac suffix
                #maybe add another saftey check here if self._last_n - jump < 0 and if so just do -> fileid = self._walker[n] 
            else:
                fileid = self._walker[self._last_n + jump]  #get file name without .flac suffix
            self._state = 1
        return load_sv_librispeech_item(fileid, self._path, self._ext)

    def __len__(self) -> int:
      return len(self._walker)
