import torch
from torch import nn
import torch.nn.functional as F

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


# maybe for future use:
class SIAMESE_SV(nn.Module):
    def __init__(self, width, speakers_num):
        super(SIAMESE_SV, self).__init__()
        self._height = self._height  # according wav2vec last conv layer kernels amount
        self._width = width
        self.speakers_num = speakers_num
        self.avg_pool = nn.AvgPool1d(kernel_size=self._width)
        self.fc = nn.Sequential(
            nn.Linear(self._height, self.speakers_num),
            nn.ReLU()
        )

        def forward_once(self, x):
            # Forward pass
            x = self.avg_pool(x)
            x = x.view(-1, self._height)
            x = self.fc(x)
            return x

        def forward(self, input1, input2):
            # forward pass of input 1
            output1 = self.forward_once(input1)
            # forward pass of input 2
            output2 = self.forward_once(input2)
            # returning the feature vectors of two inputs
            return output1, output2
