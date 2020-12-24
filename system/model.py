import torch
import torch.nn as nn
import numpy as np

from dataGenerators import DataGeneratorVector
from systemLoop import LoopStatus


class AlbertNet(nn.Module):

    def __init__(self, hidden_size, input_size=2, num_layers=1):
        super(AlbertNet, self).__init__()

        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.final = nn.Linear(hidden_size, 2)

    def forward(self, seq):
        output, (hidden, _) = self.LSTM(seq)
        output = torch.tanh(output[-1, :, :])
        pred = self.final(output)

        return pred

class Model:

    DEFAULT_PATH =  '/content/drive/MyDrive/diplomovka/LSTM/models/Albert_vector_1.npy'

    def __init__(self, path=None, device='cuda:0'):
        self.path = path if path is not None else Model.DEFAULT_PATH
        self.device = device

        self.model = self.load_model()

    def load_model(self):

        if self.device == 'cuda:0':
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'

        d = torch.load(self.path, map_location=map_location)
        model = AlbertNet(**d['args']['model_args'])
        model.load_state_dict(d['model_state_dict'])
        model.to(self.device)
        model.eval()

        return model

    def normalize(self, data):
        model_data = data * 1000 + 10
        model_data = DataGeneratorVector.normalizeInput(model_data.transpose(1,0,2))
        model_data = torch.tensor(model_data, device=self.device).float()
        return model_data

    def denormalize(self, data):
        output = DataGeneratorVector.denormalizeOutput(data)
        output = (output)/1000
        return output


    def predict(self, status: LoopStatus):
        data = status.data
        model_data = self.normalize(data)
        output = self.model(model_data).cpu().detach().numpy()
        output = self.denormalize(output)
        status.directions = output
        status.predictions = data[:,-1] + output

