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


def load_sv_librispeech_list(fileid_list: str, path: str, ext: str = '.pt') -> Tuple[Tensor, int]:
    '''This function return example as wav2vec representation or librispeech audio
       representation from a given path'''
    new_list = []
    for i in fileid_list:
        speaker_id, utterance_id = fileid.split("-")
        file_to_load = fileid + ext
        file_path = os.path.join(path, speaker_id, file_to_load)
        if ext == '.flac':
            input, _ = torchaudio.load(file_path) # Load audio - dont care about sample rate
        else:
            input = torch.load(file_path)
        new_list.append(input)
    #transform_input = torch.squeeze(torch.transpose(transform(torch.unsqueeze(torch.unsqueeze(input,0),2)),0,1))
    #return (input, int(speaker_id))
    return new_list


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



class my_meta_dataset(Dataset):
    _ext_audio = ".flac"
    _ext_repr = ".pt"

    def __init__(self, root: str, url: str = "train-clean-360",
                 folder_in_archive: str = "cut_train_data_360_repr",
                 download: bool = False, is_SV: bool = True, wav2vec_fine_tuning: bool = False , file_ext='.pt') -> None:
        self.fix_seed = True
        self.n_ways = 5
        self.n_shots = 1
        self.n_queries = 11
        self.n_test_runs = 100

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
        #get walker generator that holds information about subdirectories and files
        walker_generator = os.walk(self._path)
        self._walker = walker_generator
        _, self.classes, _ = next(walker_generator) #get all subdirs names (practically speakers ids)
        #walker = walk_files(self._path, suffix=self._ext, prefix=False, remove_suffix=True)
        #self._walker = list(walker)
        self._last_n = -1
        self._state = 1 #1 -> positive & counter = 0 ;  2 -> positive & counter = 1


    def get_file_list_from_dir(self, dir_name):
        return os.listdir(self._path +'/'+ dir_name)


    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        #print(self.classes)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        # print(type((self.data[cls_sampled[0]])[1]))
        # print(cls_sampled)
        #עבור על על הלייבלים שדגמת וקח דוגמה אחת רנדומלית מתוך כל אחד מהם
        for idx, cls in enumerate(cls_sampled):
            # speaker_files = np.asarray(self.get_file_list_from_dir(cls)).astype('uint8')
            speaker_files = np.asarray(self.get_file_list_from_dir(cls))
            # imgs = np.asarray(self.data[cls]).astype('uint8')
            support_xs_ids_sampled = int(np.random.choice(range(speaker_files.shape[0]), self.n_shots, False))
            
            #מתוך 600 תמונות תבחר לי תמונה אחת
            #append the tensor and not the - afterwards using the load_sv_librispeech_list function!!!!!!!!!
            support_exmp =  torchaudio.load(self._path + '/' + cls+'/'+str(speaker_files[support_xs_ids_sampled]))
            support_xs.append(support_exmp[0].numpy())
            #get audio values from this file !!!!!!!!!!!!!!!!!!!!!!!!!!
            #כל פעם מוסיפים לפה את הייצוג של תמונה דגומה
            # print(f'*******{support_xs[0].shape}***********')
            support_ys.append([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(speaker_files.shape[0]), support_xs_ids_sampled)
            # query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            for ids in query_xs_ids:
                query_exmp = torchaudio.load(self._path + '/' + cls+'/'+str(speaker_files[ids]))
                query_xs.append(query_exmp[0].numpy())
            #!!!!!!!!!!!!!!!!! append data itself and not the file
            query_ys.append([idx] * query_xs_ids.shape[0])
        
        support_xs, support_ys, query_xs, query_ys = torch.tensor(support_xs), torch.tensor(support_ys), torch.tensor(
            query_xs), torch.tensor(query_ys)
        # support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(
        #     query_xs), np.array(query_ys)
        # num_ways, n_queries_per_way, height, width, channel = query_xs.shape
        # query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
        # query_ys = query_ys.reshape((num_ways * n_queries_per_way, ))
                
        # support_xs = support_xs.reshape((-1, height, width, channel))
        # # if self.n_aug_support_samples > 1:
        # #     support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
        # #     support_ys = np.tile(support_ys.reshape((-1, )), (self.n_aug_support_samples))
        # support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        # query_xs = query_xs.reshape((-1, height, width, channel))
        # query_xs = np.split(query_xs, query_xs.shape[0], axis=0)
        
        # support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))
        # query_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), query_xs)))

        #Where do i need to call the load_sv_librispeech_list function to convert file names to my actual input values????????

        return support_xs, support_ys, query_xs, query_ys
        
    def __len__(self):
        return self.n_test_runs
