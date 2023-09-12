import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import librosa
import numpy as np
import random

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def save_model(model, outpath):
    from torch import save
    from os import path
    return save(model.state_dict(), outpath)


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r

class MCD(object):
    def __init__(self, n_mfcc=13):
        self.n_mfcc = n_mfcc

    def mcd(self, C, C_hat):
        """C and C_hat are NumPy arrays of shape (T, D),
        representing mel-cepstral coefficients.

        """
        K = 10 / np.log(10) * np.sqrt(2)
        return K * np.mean(np.sqrt(np.sum((C - C_hat) ** 2, axis=1)))

    def __call__(self, y_head, y):
        pred_log = np.log10((y_head.detach().cpu().numpy())**2)
        org_log = np.log10((y.detach().cpu().numpy())**2)
        mfccs_pred = librosa.feature.mfcc(S=pred_log.T,
                                         dct_type=2, n_mfcc=self.n_mfcc, norm='ortho', lifter=0)
        mfccs_org = librosa.feature.mfcc(S=org_log.T,
                                         dct_type=2, n_mfcc=self.n_mfcc, norm='ortho', lifter=0)

        mfcc_pred_T = mfccs_pred.T
        mfcc_org_T = mfccs_org.T
        MCD = self.mcd(mfcc_pred_T[:,1:12], mfcc_org_T[:,1:12])
        return MCD

class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        return F.cross_entropy(input, target)

class RegressionLoss(torch.nn.Module):
    def forward(self, input, target):
        return F.mse_loss(input, target)

class MyLSTM(nn.Module):
    def __init__(self, D_in = 54, H = 256, D_out = 80, num_layers= 3, bidirectional=False):
        super(MyLSTM, self).__init__()
        self.hidden_dim = H
        self.num_layers = num_layers
        self.num_direction =  2 if bidirectional else 1
        self.lstm = nn.LSTM(D_in, H, num_layers, bidirectional=bidirectional)
        self.hidden2out = nn.Linear(self.num_direction*self.hidden_dim, D_out)

    def init_hidden(self, x):
        h, c = (Variable(torch.zeros(self.num_layers * self.num_direction, x.shape[1], self.hidden_dim)),
                Variable(torch.zeros(self.num_layers * self.num_direction, x.shape[1], self.hidden_dim)))
        return h, c

    def forward(self, sequence, h, c):
        output, (h, c) = self.lstm(sequence, (h, c))
        output = self.hidden2out(output)
        return output

class MyBLSTM(nn.Module):
    def __init__(self, D_in, H, D_out, num_layers=1, bidirectional=True):
        super(MyBLSTM, self).__init__()
        self.hidden_dim = H
        self.num_layers = num_layers
        self.num_direction =  2 if bidirectional else 1
        self.lstm = nn.LSTM(D_in, H, num_layers, bidirectional=bidirectional)
        self.hidden2out = nn.Linear(self.num_direction*self.hidden_dim, D_out)

    def init_hidden(self, batch_size=1):
        h, c = (Variable(torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim)),
                Variable(torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim)))
        return h,c

    def forward(self, sequence, h, c):
        output, (h, c) = self.lstm(sequence, (h, c))
        output = self.hidden2out(output)
      #  output = self.hidden2out(output.view(len(sequence), -1))
        return output

class DNN(torch.nn.Module):
    def __init__(self, D_in, H, D_out, num_layers=2):
        super(DNN, self).__init__()
        self.first_linear = nn.Linear(D_in, H)
        self.hidden_layers = nn.ModuleList([nn.Linear(H, H) for _ in range(num_layers)])
        self.last_linear = nn.Linear(H, D_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.first_linear(x))
        for hl in self.hidden_layers:
            h = self.relu(hl(h))
        return self.last_linear(h)

