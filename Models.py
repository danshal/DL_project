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
class RNN_model(nn.Module):
  def __init__(self, num_classes, winit, p_drop ,seq_length,hidden_units,layers_num, batch_size):
    super().__init__()
    self._num_classes = num_classes
    self._layers_num = layers_num
    self._hidden_units = hidden_units
    self.dropout = nn.Dropout(p=p_drop)
    self.winit = winit
    self._batch_size = batch_size
    self._rnn = nn.GRU(self._hidden_units, self._hidden_units, self._layers_num,
                        batch_first=True, dropout=p_drop)
    self.decoder = nn.Linear(self._hidden_units*seq_length, self._num_classes)
    self.reset_parameters()
    

  def reset_parameters(self):
    '''This function will set the weights to a uniform
       distribution in [-winit, winit] range'''
    for param in self.parameters():
      nn.init.uniform_(param, -self.winit, self.winit)

  def init_hidden(self):
    weight = next(self.parameters()).data
    return weight.new_zeros(self._layers_num,  self._batch_size, self._hidden_units)


  def forward(self, x, states):
    x = self.dropout(x)
    x, states = self._rnn(x, states)
    x = self.dropout(x)
    #x = x.contiguous().view(-1, hidden_units) #flatten before fully connected layer
    x = x.contiguous().view(self._batch_size, -1)
    scores = self.decoder(x)
    return scores, states

def getPerplexity(netToEval, batchGenerator):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  netToEval.to(device)
  with torch.no_grad():
    netToEval.eval()
    criterionEval = nn.CrossEntropyLoss()
    hidden = netToEval.init_hidden()
    if type(hidden) == tuple:
      hidden = [state.detach().to(device) for state in hidden] 
    else:
      hidden = hidden.detach().to(device)
    
    meanPerplexity = 0
    for i, (features, targets) in enumerate(batchGenerator, start=0):     
      inputs = features.to(device)
      output, hidden = netToEval(inputs, hidden)
      targets = targets.reshape(-1).to(device)
      loss = criterionEval(output, targets.long())
      meanPerplexity = (meanPerplexity*i + loss.item())/(i+1)
    netToEval.train() # reset to train mode after iterating through data
  return np.exp(meanPerplexity)


  ### MNIST code
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)


class Wav2vecTuning(nn.Module):
  '''This class is the wav2vec fine tuning model for speaker verification task'''
  def __init__(self, top_model):
    super(Wav2vecTuning, self).__init__()
    self._cp_path = '/home/Daniel/DeepProject/wav2vec/wav2vec_large.pt'
    self.wav2vec, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([self._cp_path])
    self.wav2vec = self.wav2vec[0]  #load pretrained model
    self.top_model = top_model  #our wav2vec on top model
    self.pooling = nn.MaxPool1d(298)
  
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