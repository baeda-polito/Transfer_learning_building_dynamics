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
from functions import import_file, min_max_T, normalization, split_multistep_sequences, mean_absolute_percentage_error, create_data, define_period
from NN_architectures import LSTM



def train_model(model, epochs, train_dl, optimizer, criterion, train_batch_size, min_T, max_T, train_metrics_path, loss_path, lr_scheduler='', mode=''):
    model.train()
    # initialize the training loss and the validation loss
    TRAIN_LOSS = []
    MAE_list = []
    MSE_list = []
    MAPE_list = []
    for t in range(epochs):

        # TRAINING LOOP
        loss = []
        h = model.init_hidden(train_batch_size)  # hidden state is initialized at each epoch
        for x, label in train_dl:
            h = model.init_hidden(train_batch_size) # since the batch is big enough, a stateless mode is used (also considering the possibility to shuffle the training examples, which increase the generalization ability of the network)
            h = tuple([each.data for each in h])
            output, h = model(x.float(), h)
            #label = label.unsqueeze(1) # utilizzo .unsqueeze per non avere problemi di dimensioni
            loss_c = criterion(output, label.float())
            optimizer.zero_grad()
            loss_c.backward()
            optimizer.step()
            loss.append(loss_c.item())


        output = output.detach().numpy()
        label = label.detach().numpy()
        output = min_T + output * (max_T - min_T)
        label = min_T + label * (max_T - min_T)


        MAE_list.append([mean_absolute_error(label[:, i], output[:, i]) for i in range(6)])
        mae_pd = pd.DataFrame(MAE_list, columns=['MAE1', 'MAE2', 'MAE3', 'MAE4', 'MAE5', 'MAE6'])
        mae_pd['MAE_avg'] = mae_pd.mean(axis=1)


        MSE_list.append([mean_squared_error(label[:, i], output[:, i]) for i in range(6)])
        mse_pd = pd.DataFrame(MSE_list, columns=['MSE1', 'MSE2', 'MSE3', 'MSE4', 'MSE5', 'MSE6'])
        mse_pd['MSE_avg'] = mse_pd.mean(axis=1)


        MAPE_list.append([mean_absolute_percentage_error(label[:, i], output[:, i]) for i in range(6)])
        mape_pd = pd.DataFrame(MAPE_list, columns=['MAPE1', 'MAPE2', 'MAPE3', 'MAPE4', 'MAPE5', 'MAPE6'])
        mape_pd['MAPE_avg'] = mape_pd.mean(axis=1)

        train_metrics = pd.concat([mae_pd, mse_pd, mape_pd], axis=1)
        train_metrics.to_csv(path_or_buf=train_metrics_path, sep=',', decimal='.', index=False)

        TRAIN_LOSS.append(np.sum(loss)/train_batch_size)

        # Save the training loss
        loss_pd = np.array(TRAIN_LOSS).reshape(-1, 1)
        loss_pd = pd.DataFrame(loss_pd, columns=['train_loss'])
        loss_pd.to_csv(path_or_buf=loss_path, sep=',', decimal='.', index=False)

        if mode == 'TL':
            lr_scheduler.step()

        print('Epoch : ', t, 'Training Loss : ', TRAIN_LOSS[-1])

    return TRAIN_LOSS, train_metrics




def test_model(model, test_dl, maxT, minT, batch_size, test_metrics_path):
    model.eval()
    h = model.init_hidden(batch_size)
    y_pred = []
    y_lab = []
    MAE_list = []
    MSE_list = []
    MAPE_list = []

    for inputs, labels in test_dl:
        h = tuple([each.data for each in h])
        test_output, h = model(inputs.float(), h)
        #labels = labels.unsqueeze(1)

        # RESCALE OUTPUTS
        test_output = test_output.detach().numpy()
        # test_output = np.reshape(test_output, (-1, 1))
        test_output = minT + test_output*(maxT-minT)

        # RESCALE LABELS
        labels = labels.detach().numpy()
        # labels = np.reshape(labels, (-1, 1))
        labels = minT + labels*(maxT-minT)


    MAE_list.append([mean_absolute_error(labels[:, i], test_output[:, i]) for i in range(6)])
    mae_pd = pd.DataFrame(MAE_list, columns=['MAE1', 'MAE2', 'MAE3', 'MAE4', 'MAE5', 'MAE6'])
    mae_pd['MAE_avg'] = mae_pd.mean(axis=1)


    MSE_list.append([mean_squared_error(labels[:, i], test_output[:, i]) for i in range(6)])
    mse_pd = pd.DataFrame(MSE_list, columns=['MSE1', 'MSE2', 'MSE3', 'MSE4', 'MSE5', 'MSE6'])
    mse_pd['MSE_avg'] = mse_pd.mean(axis=1)


    MAPE_list.append([mean_absolute_percentage_error(labels[:, i], test_output[:, i]) for i in range(6)])
    mape_pd = pd.DataFrame(MAPE_list, columns=['MAPE1', 'MAPE2', 'MAPE3', 'MAPE4', 'MAPE5', 'MAPE6'])
    mape_pd['MAPE_avg'] = mape_pd.mean(axis=1)


    test_metrics = pd.concat([mae_pd, mse_pd, mape_pd], axis=1)
    test_metrics.to_csv(path_or_buf=test_metrics_path, sep=',', decimal='.', index=False)

    return y_pred, y_lab, test_metrics
