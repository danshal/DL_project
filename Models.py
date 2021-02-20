import torch
from torch import nn
import torch.nn.functional as F
import fairseq

class NAIVE_SV(nn.Module):
    '''This class represents a model that can take as input a [512,width]
      (width -> ?[sec] uuterance) and outputs a softmax for speaker verification
       task.
       Model Architecture:
       1. Avg pool on whole time axis (298 samples)
       2. Linear transformation that will output as number of speakers'''

    def __init__(self, width, speakers_num):
        super(NAIVE_SV, self).__init__()
        self._height = 512  # according wav2vec last conv layer kernels amount
        self._width = width
        self.speakers_num = speakers_num
        #self.avg_pool = nn.AvgPool1d(kernel_size=self._width)
        self.fc = nn.Sequential(
            nn.Linear(self._height, self.speakers_num)
        )
        # Dont think we need any regularization for this thin network but just in case:
        # self.batch_norm = nn.BatchNorm1d(self.speakers_num)
        # self.drop = nn.Dropout(0.5)  #Hard coded probability for now...

    def forward(self, x):
        # Forward pass
        #x = self.avg_pool(x)
        #x = x.view(-1, self._height)
        x = self.fc(x)
        return x

class FC_SV(nn.Module):
    '''This class represents a model that can take as input a [512,width]
      (width -> ?[sec] uuterance) and outputs a softmax for speaker verification
       task.
       Model Architecture:
       1. Avg pool on whole time axis (298 samples)
       2. Linear transformation that will output as number of speakers'''

    def __init__(self):
        super(FC_SV, self).__init__()
        self._height = 512  # according wav2vec last conv layer kernels amount
        self.output_dim = 128
        #self.avg_pool = nn.AvgPool1d(kernel_size=self._width)
        self.fc = nn.Sequential(
            nn.Linear(self._height, self.output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.output_dim),
            nn.Linear(self.output_dim,self.output_dim)
        )

    def forward(self, x):
        # Forward pass
        #x = self.avg_pool(x)
        #x = x.view(-1, self._height)
        x = self.fc(x)
        return x




class Wav2vecTuning(nn.Module):
  '''This class is the wav2vec fine tuning model for speaker verification task'''
  def __init__(self, top_model):
    super(Wav2vecTuning, self).__init__()
    self._cp_path = '/home/Daniel/DeepProject/wav2vec/wav2vec_large.pt'
    self.wav2vec, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([self._cp_path])
    self.wav2vec = self.wav2vec[0]  #load pretrained model
    self.top_model = top_model  #our wav2vec on top model
    self.pooling = nn.AvgPool1d(298)
  
  def forward(self, x):
    x = x.squeeze()
    x = self.wav2vec.feature_extractor(x)
    x = self.wav2vec.feature_aggregator(x)
    #maybe add an avgpool layer in here?
    #ori: ofcourse!!
    x =  torch.squeeze(self.pooling(x))
    #x = x.view(-1, x.shape[1] * x.shape[2]) #The assumption here is that we get the full repr size
    x = self.top_model(x)
    return x



class FC_SV_TUNING(nn.Module):
    '''This class represents a model that can take as input a [512,width]
      (width -> ?[sec] uuterance) and outputs a softmax for speaker verification
       task.
       Model Architecture:
       1. Avg pool on whole time axis (298 samples)
       2. Linear transformation that will output as number of speakers'''

    def __init__(self, height):
        super(FC_SV_TUNING, self).__init__()
        self._height = height  # according wav2vec last conv layer kernels amount
        self.output_dim = 128
        self.output_dim2 = 64
        #self.avg_pool = nn.AvgPool1d(kernel_size=self._width)
        self.fc = nn.Sequential(
            nn.Linear(self._height, self.output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.output_dim),
            nn.Linear(self.output_dim, self.output_dim2)
        )

    def forward(self, x):
        # Forward pass
        x = self.fc(x)
        return x