import torch
import torch.nn as nn                   # All neural network models, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim             # For all optimization algorithms, SGD, Adam, etc.
from torch.optim import lr_scheduler    # To change (update) the learning rate.
import torch.nn.functional as F         # All functions that don't have any parameters.
import numpy as np
from numpy import hstack
import torchvision
from torchvision import datasets        # Has standard datasets that we can import in a nice way.
from torchvision import models
from torchvision import transforms      # Transformations we can perform on our datasets.
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import matplotlib
from datetime import datetime as dt
import matplotlib.gridspec as gridspec
from pandas import DataFrame
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.optim import Adam
from torch.nn import BCELoss
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from matplotlib import cm
import seaborn as sns
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import csv
from csv import DictWriter
import xlsxwriter
import openpyxl
from functions import import_file, min_max_T, normalization, split_multistep_sequences, mean_absolute_percentage_error, create_data

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.seq_length = lookback
        #
        # self.lstm_layers = nn.Sequential(
        #     nn.LSTM(input_size=input_size, hidden_size=hidden_size,
        #             num_layers=num_layers, batch_first=True),
        #
        #     nn.LSTM(input_size=input_size, hidden_size=hidden_size,
        #             num_layers=num_layers, batch_first=True),
        #
        #     nn.LSTM(input_size=input_size, hidden_size=hidden_size,
        #             num_layers=num_layers, batch_first=True)
        # )

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x, h):
        batch_size, seq_len, _ = x.size()

        # # Propagate input through LSTM
        # out, h = self.lstm_layers[0](x, h)
        # out, h = self.lstm_layers[1](x, h)
        # out, h = self.lstm_layers[2](x, h)

        out, h = self.lstm(x, h)
        out = out[:, -1, :]
        # h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(out)

        return out, h

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        hidden = (hidden_state, cell_state)  # HIDDEN is defined as a TUPLE
        return hidden

