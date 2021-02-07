import os
import soundfile as sf
import torch
import fairseq


class sv_helper(object):
    def __init__(self, wav2vec_model):
        self._model = wav2vec_model


    def read_audio(self, fname):
        ''' Load an audio file and return PCM along with the sample rate '''
        wav, sr = sf.read(fname)
        assert sr == 16e3

        return wav, 16e3


    def audio2tensor(self, fname):
        ''' This function will convert audio samples to tensor '''
        input, _ = read_audio(fname)
        input_tensor = torch.from_numpy(input).float()
        return input_tensor.reshape(1, input_tensor.size(0))


    def get_audio_repr(self, fname):
        ''' This function will get audio as input and will output its representation
            as wav2vec model outputs '''
        input_tensor = audio2tensor(fname)
        z = self._model.feature_extractor(input_tensor)
        c = self._model.feature_aggregator(z)
        return c


    def get_tensor_repr(self, input_tensor):
        ''' This function will get a tensor and return its representation as wav2vec
            model outputs '''
        with torch.no_grad():
            z = self._model.feature_extractor(input_tensor)
            c = self._model.feature_aggregator(z)
            return c


    def get_sv_example_generator(self, data):
        ''' This function gets data that loaded as LIBRISPEECH dataset and yields its
            wav2vec tensor representation with the speaker id ''' 
        train_data_iter = iter(data)
        for i in train_data_iter:
            yield [get_tensor_repr(i[0]), i[3]]


    def get_utterances_number(self, train_data):
        ''' This function gets train_data(dataset object) and returns number of
            utterances per one audio file '''
        example = next(iter(train_data))
        width = example[0].shape[2]
        return width


    def get_speakers_num(self, dataset_path):
        ''' This function will get the dataset path and will return number of
            subdirectoris in this path that is equal ti the number of speakers in this
            dataset '''
        num_speakers = len(list(os.walk(dataset_path)))
        return num_speakers
