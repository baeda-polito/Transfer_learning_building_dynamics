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
from functions import import_file, min_max_T, normalization, create_data, split_multistep_sequences, mean_absolute_percentage_error

# ___________________________________________IMPORT AND NORMALIZATION___________________________________________________

# year = '2015'
# list_3_years = ['1990', '1991', '1992']
#list_5_years = ['1990', '1991', '1992', '1993', '1994']
# list_10_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999']

# ZONE: CONFROOM_BOT_1, CONFROOM_MID_2, ENCLOSEDOFFICE_BOT_2, OPENOFFICE_BOT_3
zone = 'CONFROOM_BOT_1'
clm = '3C'
eff = 'Standard'
list_year = ['2016', 'TMY3']
occ = 'run_1'
df = import_file(zone, clm, eff, list_year, occ)
# df = import_file(year, eff='')
# df = import_file(list_3_years)
# df = import_file(list_5_years)
# df = import_file(list_10_years)

max_T, min_T = min_max_T(df=df, column='CONFROOM_BOT_1 ZN:Zone Mean Air Temperature[C]')

# # Temperature plot
# plt.plot(df[1056:2112, -1])#'CONFROOM_BOT_1 ZN:Zone Mean Air Temperature[C]'
# plt.xlim(0, 600)
# plt.title('Mean zone air temperature [°C]', size=15)
# plt.show()


df = normalization(df)

# ______________________________________________________________________________________________________________________
# 1h -> 6 timesteps
# 24h = 6*24 = 144 timesteps
# 1 weak = 144*7 = 1008 timesteps
# 1 month = 144*30 (circa) =  4320 timesteps
# ______________________________________________________________________________________________________________________

def define_period(df, train_time, test_period):
    df_def = pd.DataFrame()
    if train_time == '1_week':
        l_train = 1008+48
        l_val = int(l_train*2)
        df_def = df[:int(l_train*3)]
    if train_time == '1_month':
        l_train = 4464
        l_val = int(l_train * 2)
        df_def = df[:int(l_train * 3)]
    if train_time == '1_year':
        l_train = int(0.5 * len(df))  # 31536 (per un anno)
        if test_period == '1_week':
            l_val = int(l_train + 1056)  # da 31536 a 42048, cioè 10512 valori (per un anno)
            l_test = int(l_val + 1056)
        if test_period == '1_month':
            l_val = int(l_train + 4464)
            l_test = int(l_val + 4464)
        if test_period == '1_year':
            l_val = int(l_train + 1056)
            l_test = len(df)
        df_def = df

    return df_def, l_train, l_val, l_test

test_period = '1_year'
df, l_train, l_val, l_test = define_period(df, train_time='1_year', test_period=test_period)


# ______________________________________Datasets_preprocessing__________________________________________________________
period = 6
# l_train = int(0.8 * len(df)) # 31536 (per un anno)
# l_val = int(l_train + 0.2*len(df)) # da 31536 a 42048, cioè 10512 valori (per un anno)
# l_val = int(l_train+0.05*len(df)) # da 31536 a 42048, cioè 10512 valori (per un anno)

train_df, val_df, test_df = create_data(df=df, col_name='CONFROOM_BOT_1 ZN:Zone Mean Air Temperature[C]', l_train=l_train, l_val=l_val, period=period, l_test=l_test)
train_df, val_df, test_df = train_df.to_numpy(), val_df.to_numpy(), test_df.to_numpy()


# ________________________________________Splitting in X, Y data________________________________________________________
n_steps = 48 # (8 hours)
train_X, train_Y = split_multistep_sequences(train_df, n_steps)
val_X, val_Y = split_multistep_sequences(val_df, n_steps)
test_X, test_Y = split_multistep_sequences(test_df, n_steps)
#
# print(train_X.shape, train_Y.shape)
# print(val_X.shape, val_Y.shape)
# print(test_X.shape, test_Y.shape)

# Convert medium office to tensors
train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).float()
val_X = torch.from_numpy(val_X).float()
val_Y = torch.from_numpy(val_Y).float()
test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).float()

print(type(train_X), train_X.shape)
print(type(train_Y), train_Y.shape)
print(type(val_X), val_X.shape)
print(type(val_Y), val_Y.shape)
print(type(test_X), test_X.shape)
print(type(test_Y), test_Y.shape)



# ________________________________________________MLP NETWORK ___________________________________________________________
# Multivariate model definition
class MLP(nn.Module):
    # define model elements
    def __init__(self, n_features):
        super(MLP, self).__init__()
        self.hidden1 = Linear(n_features, 100) # input to first hidden layer
        self.act1 = ReLU()
        self.hidden2 = Linear(100, 70)
        self.act2 = ReLU()
        self.hidden3 = Linear(70, 70)
        self.act3 = ReLU()
        self.hidden4 = Linear(70, 10)
        self.act4 = ReLU()
        self.hidden5 = Linear(10, 6)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        # fourth hidden layer and output
        X = self.hidden4(X)
        X = self.act4(X)
        # fifth hidden layer and output
        X = self.hidden5(X)

        return X


# __________________________________________________TRAINING PHASE______________________________________________________
if test_period == '1_week':
    train_batch_size = 400
    val_batch_size, test_batch_size = 100, 100
if test_period == '1_month':
    train_batch_size = 400
    val_batch_size, test_batch_size = 200, 200
if test_period == '1_year':
    train_batch_size, val_batch_size, test_batch_size = 400, 100, 100


# train_batch_size = 500
train_data = TensorDataset(train_X, train_Y)
train_dl = DataLoader(train_data, batch_size=train_batch_size, shuffle=False, drop_last=True)

# val_batch_size = 500
val_data = TensorDataset(val_X, val_Y)
val_dl = DataLoader(val_data, batch_size=val_batch_size, shuffle=False, drop_last=True)


# PARAMETERS
lr = 0.007567558023327467 #0.008
n_timestep = 48
features = 8
# lstm = LSTM(n_features, n_timesteps)

# n_features = n_timestep*train_batch_size
n_features = 384 # 48 * 8
model = MLP(n_features)
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def train_model(model, epochs, train_dl, val_dl, optimizer, criterion, lr_scheduler, mode='', TLmode='ML'):
    # START THE TRAINING PROCESS
    model.train()
    # initialize the training loss and the validation loss
    TRAIN_LOSS = []
    VAL_LOSS = []

    for t in range(epochs):

        # TRAINING LOOP
        loss = []
        # h = model.init_hidden(train_batch_size)  # hidden state is initialized at each epoch
        for x, label in train_dl:
            # h = model.init_hidden(train_batch_size) # since the batch is big enough, a stateless mode is used (also considering the possibility to shuffle the training examples, which increase the generalization ability of the network)
            # h = tuple([each.data for each in h])
            x = x.reshape(-1, n_features)
            # x = x.view(x.size(0), -1)
            output = model(x.float())
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

        MAPE = mean_absolute_percentage_error(label, output)
        MSE = mean_squared_error(label, output)
        MAE = mean_absolute_error(label, output)

        MAPE1 = mean_absolute_percentage_error(label[:, 0], output[:, 0])
        MSE1 = mean_squared_error(label[:, 0], output[:, 0])
        MAE1 = mean_absolute_error(label[:, 0], output[:, 0])

        MAPE2 = mean_absolute_percentage_error(label[:, 1], output[:, 1])
        MSE2 = mean_squared_error(label[:, 1], output[:, 1])
        MAE2 = mean_absolute_error(label[:, 1], output[:, 1])

        MAPE3 = mean_absolute_percentage_error(label[:, 2], output[:, 2])
        MSE3 = mean_squared_error(label[:, 2], output[:, 2])
        MAE3 = mean_absolute_error(label[:, 2], output[:, 2])

        MAPE4 = mean_absolute_percentage_error(label[:, 3], output[:, 3])
        MSE4 = mean_squared_error(label[:, 3], output[:, 3])
        MAE4 = mean_absolute_error(label[:, 3], output[:, 3])

        MAPE5 = mean_absolute_percentage_error(label[:, 4], output[:, 4])
        MSE5 = mean_squared_error(label[:, 4], output[:, 4])
        MAE5 = mean_absolute_error(label[:, 4], output[:, 4])

        MAPE6 = mean_absolute_percentage_error(label[:, -1], output[:, -1])
        MSE6 = mean_squared_error(label[:, -1], output[:, -1])
        MAE6 = mean_absolute_error(label[:, -1], output[:, -1])

        # Create the file saving metrics
        train_metrics = pd.DataFrame([[MSE1, MAE1, MAPE1, MSE2, MAE2, MAPE2, MSE3, MAE3, MAPE3, MSE4, MAE4, MAPE4, MSE5, MAE5, MAPE5, MSE6, MAE6, MAPE6, MSE, MAE, MAPE]], columns=['MSE1', 'MAE1', 'MAPE1', 'MSE2', 'MAE2', 'MAPE2', 'MSE3', 'MAE3', 'MAPE3', 'MSE4', 'MAE4', 'MAPE4', 'MSE5', 'MAE5', 'MAPE5', 'MSE6', 'MAE6', 'MAPE6', 'MSE', 'MAE', 'MAPE'])

        excel_path = 'C:\\Users\\ricme\\Desktop\\Politecnico\\Tesi magistrale\\TL_coding\\meta_data\\code\\thesis_project\\ML\\' + source_period + '\\' + test_period + '\\' +zone+'_'+clm+'_'+eff+'_'+occ+'_'+TLmode+'_'+source_period+'_'+test_period+'_mlp_train_metrics.csv'
        # excel_path = 'E:\\tesi_messina\\thesis_project\\optuna_img\\mlp_img\\optuna_metrics.csv'

        # IMPORTANTE PROMEMORIA: il numero di righe del file excel sarebbe la parte intera della divisione tra righe data set di testing sul batch size di testing
        if not os.path.exists(excel_path):
            train_metrics.to_csv(path_or_buf=excel_path,
                           sep=',', decimal='.', index=False)
        else:
            runsLog = pd.read_csv(excel_path, sep=',', decimal='.')
            runsLog = runsLog.append(train_metrics)
            runsLog.to_csv(path_or_buf=excel_path,
                            sep=',', decimal='.', index=False)


        TRAIN_LOSS.append(np.sum(loss)/train_batch_size)
        if mode == 'tuning':
            lr_scheduler.step()
        # print("Epoch: %d, training loss: %1.5f" % (train_episodes, LOSS[-1]))

        # VALIDATION LOOP
        val_loss = []
        # h = model.init_hidden(val_batch_size)
        for inputs, labels in val_dl:
            inputs = inputs.reshape(-1, n_features)
            # x = x.view(x.size(0), -1)
            # h = tuple([each.data for each in h])
            val_output = model(inputs.float())
            #val_labels = labels.unsqueeze(1) # CAPIRE SE METTERLO O NO
            val_loss_c = criterion(val_output, labels.float())
            val_loss.append(val_loss_c.item())
        # VAL_LOSS.append(val_loss.item())
        VAL_LOSS.append(np.sum(val_loss)/val_batch_size)
        print('Epoch : ', t, 'Training Loss : ', TRAIN_LOSS[-1], 'Validation Loss :', VAL_LOSS[-1])

    return TRAIN_LOSS, VAL_LOSS

source_period = '1_year'
epochs = 120
# train_loss, val_loss = train_model(model, epochs=epochs, train_dl=train_dl, val_dl=val_dl, optimizer=optimizer, criterion=criterion, lr_scheduler='', mode='', TLmode='ML')



#
# # Plot to verify validation and train loss, in order to avoid underfitting and overfitting
# plt.plot(train_loss,'--',color='r', linewidth = 1, label = 'Train Loss')
# plt.plot(val_loss,color='b', linewidth = 1, label = 'Validation Loss')
# plt.yscale('log')
# plt.ylabel('Loss (MSE)')
# plt.xlabel('Epoch')
# plt.xticks(np.arange(0, int(epochs), 50))
# plt.grid(b=True, which='major', color='#666666', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.title("Multi-steps training VS Validation loss", size=15)
# plt.legend()
# # plt.savefig('immagini/2015/mlp/MLP_Train_VS_Val_LOSS({}_epochs_lr_{}).png'.format(epochs, lr))
# plt.show()

# ____________________________________________________SAVE THE MODEL____________________________________________________

# Load the best model obtained with optuna
year = '2016'
model_epochs = '120'
model_lr = '0.007302394287066261'
model_mape = '1.0967751033604145'
model.load_state_dict(torch.load('optuna_models/OPTUNA_MLP_train_on_'+year+'_epochs_'+model_epochs+'_lr_'+model_lr+'_MAPE_'+model_mape+'.pth'))
# model_fe.load_state_dict(torch.load('optuna_models/OPTUNA_LSTM_train_on_'+year+'_epochs_'+model_epochs+'_lr_'+model_lr+'_MAPE_'+model_mape+'.pth'))
# OPTUNA_MLP_train_on_2016_epochs_120_lr_0.007302394287066261_MAPE_1.0967751033604145
# period = 'year'
# year = '2015'
# torch.save(model.state_dict(), 'MLP_train_on_' + period + '_' + str(year) + '_epochs_'+str(epochs)+'_lr_' + str(lr) + '.pth')
# torch.save(lstm.state_dict(), 'train_on_year_2015_epochs_150_lr_0.008.pth')

# Load a model
# model = MLP(n_features)
# period = 'week'
# year = '2015'
# model_epochs = 190
# model_lr = 0.009
# model.load_state_dict(torch.load('train_on_'+period+'_'+year+'_epochs_'+str(model_epochs)+'_lr_'+str(model_lr)+'.pth'))
# model.load_state_dict(torch.load('train_on_10_years_2015_epochs_25_lr_0.008_batch_2000.pth'))

# __________________________________________________6h PREDICTION TESTING_______________________________________________

# test_batch_size = 400 # devo mettere il batch ad 1 perchè così ad ogni batch mi appendo il primo dei 6 valori predetti
test_data = TensorDataset(test_X, test_Y)
test_dl = DataLoader(test_data, shuffle=False, batch_size=test_batch_size, drop_last=True)


def test_model(model, test_dl, maxT, minT, batch_size, TLmode):
    model.eval()
    # h = model.init_hidden(batch_size)
    y_pred = []
    y_lab = []
    # y_lab6 = []
    # y_pred6 = []
    # mape_test = []
    for inputs, labels in test_dl:
        # h = tuple([each.data for each in h])
        # test_output, h = model(inputs.float(), h)
        # labels = labels.unsqueeze(1)

        inputs = inputs.reshape(-1, n_features)
        test_output = model(inputs.float())
        # loss = F.mse_loss(output.view(-1), labels.float())

        # RESCALE OUTPUTS
        test_output = test_output.detach().numpy()
        # # test_output = np.reshape(test_output, (-1, 1))
        test_output = minT + test_output*(maxT-minT)

        # RESCALE LABELS
        labels = labels.detach().numpy()
        # # labels = np.reshape(labels, (-1, 1))
        labels = minT + labels*(maxT-minT)
        #
        # mape = mean_absolute_percentage_error(test_output, labels)
        # mape_test.append(mape)
        #
        # y_pred.append(test_output[:, 0]) # test_output[0] per appendere solo il primo dei valori predetti ad ogni step
        # y_lab.append(labels[:, 0]) # labels[0] per appendere solo il primo dei valori predetti ad ogni step
        # y_pred6.append(test_output[:, 5])
        # y_lab6.append(labels[:, 5])

    MAPE = mean_absolute_percentage_error(labels, test_output)
    MSE = mean_squared_error(labels, test_output)
    MAE = mean_absolute_error(labels, test_output)

    MAPE1 = mean_absolute_percentage_error(labels[:, 0], test_output[:, 0])
    MSE1 = mean_squared_error(labels[:, 0], test_output[:, 0])
    MAE1 = mean_absolute_error(labels[:, 0], test_output[:, 0])

    MAPE2 = mean_absolute_percentage_error(labels[:, 1], test_output[:, 1])
    MSE2 = mean_squared_error(labels[:, 1], test_output[:, 1])
    MAE2 = mean_absolute_error(labels[:, 1], test_output[:, 1])

    MAPE3 = mean_absolute_percentage_error(labels[:, 2], test_output[:, 2])
    MSE3 = mean_squared_error(labels[:, 2], test_output[:, 2])
    MAE3 = mean_absolute_error(labels[:, 2], test_output[:, 2])

    MAPE4 = mean_absolute_percentage_error(labels[:, 3], test_output[:, 3])
    MSE4 = mean_squared_error(labels[:, 3], test_output[:, 3])
    MAE4 = mean_absolute_error(labels[:, 3], test_output[:, 3])

    MAPE5 = mean_absolute_percentage_error(labels[:, 4], test_output[:, 4])
    MSE5 = mean_squared_error(labels[:, 4], test_output[:, 4])
    MAE5 = mean_absolute_error(labels[:, 4], test_output[:, 4])

    MAPE6 = mean_absolute_percentage_error(labels[:, -1], test_output[:, -1])
    MSE6 = mean_squared_error(labels[:, -1], test_output[:, -1])
    MAE6 = mean_absolute_error(labels[:, -1], test_output[:, -1])

    # Create the file saving metrics
    test_metrics = pd.DataFrame([[MSE1, MAE1, MAPE1, MSE2, MAE2, MAPE2, MSE3, MAE3, MAPE3, MSE4, MAE4, MAPE4, MSE5, MAE5, MAPE5, MSE6, MAE6, MAPE6, MSE, MAE, MAPE]], columns=['MSE1', 'MAE1', 'MAPE1', 'MSE2', 'MAE2', 'MAPE2', 'MSE3', 'MAE3', 'MAPE3', 'MSE4', 'MAE4', 'MAPE4', 'MSE5', 'MAE5', 'MAPE5', 'MSE6', 'MAE6', 'MAPE6', 'MSE', 'MAE', 'MAPE'])

    excel_path = 'C:\\Users\\ricme\\Desktop\\Politecnico\\Tesi magistrale\\TL_coding\\meta_data\\code\\thesis_project\\ML\\' + source_period + '\\' + testing_time + '\\' + zone + '_' + clm + '_' + eff + '_' + occ + '_' + TLmode + '_' + source_period + '_' + test_period + '_mlp_test_metrics.csv'
    # excel_path = 'E:\\tesi_messina\\thesis_project\\optuna_img\\mlp_img\\optuna_metrics.csv'

    # IMPORTANTE PROMEMORIA: il numero di righe del file excel sarebbe la parte intera della divisione tra righe data set di testing sul batch size di testing
    if not os.path.exists(excel_path):
        test_metrics.to_csv(path_or_buf=excel_path,
                             sep=',', decimal='.', index=False)
    else:
        runsLog = pd.read_csv(excel_path, sep=',', decimal='.')
        runsLog = runsLog.append(test_metrics)
        runsLog.to_csv(path_or_buf=excel_path,
                        sep=',', decimal='.', index=False)

    return y_pred, y_lab #, y_pred6, y_lab6, mape_test


source_period = '1_year'
testing_time = test_period
y_pred, y_lab = test_model(model=model, test_dl=test_dl, maxT=max_T, minT=min_T, batch_size=test_batch_size, TLmode='ML')


mlp_test_metrics = pd.read_csv('ML/1_year/1_year/CONFROOM_BOT_1_3C_Standard_run_1_ML_1_year_1_year_mlp_test_metrics.csv')

#
# flatten = lambda l: [item for sublist in l for item in sublist]
# y_pred = flatten(y_pred)
# y_lab = flatten(y_lab)
# y_pred = np.array(y_pred, dtype=float)
# y_lab = np.array(y_lab, dtype=float)
# #
# # y_pred6 = flatten(y_pred6)
# # y_lab6 = flatten(y_lab6)
# # y_pred6 = np.array(y_pred6, dtype=float)
# # y_lab6 = np.array(y_lab6, dtype=float)
# #
# # # Shift values of 6 positions because it's the sixth hour
# # y_pred6 = pd.DataFrame(y_pred6)
# # y_pred6 = y_pred6.shift(6, axis=0)
# # y_lab6 = pd.DataFrame(y_lab6)
# # y_lab6 = y_pred6.shift(6, axis=0)
#
#
#
# error = []
# error = y_pred - y_lab
#
# plt.hist(error, 100, linewidth=1.5, edgecolor='black', color='orange')
# plt.xticks(np.arange(-0.6, 0.6, 0.1))
# plt.xlim(-0.6, 0.6)
# plt.title('LSTM model 6h prediction error')
# # plt.xlabel('Error')
# plt.grid(True)
# # plt.savefig('immagini/2015/mlp/MLP_model_error({}_epochs_lr_{}_with_sixth_hour).png'.format(epochs, lr))
# plt.show()
#
#
# plt.plot(y_pred, color='orange', label="Predicted")
# plt.plot(y_lab, color="b", linestyle="dashed", linewidth=1, label="Real")
# # plt.plot(y_pred6, color='green', label="Predicted6")
# # plt.plot(y_lab6, color="b", linewidth=1, label="Real6")# , linestyle="purple"
# plt.grid(b=True, which='major', color='#666666', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.xlim(0, 600)
# plt.ylabel('Mean Air Temperature [°C]')
# plt.xlabel('Time [h]')
# plt.title("6h prediction: Real VS predicted temperature", size=15)
# plt.legend()
# # plt.savefig('immagini/2015/mlp/MLP_real_VS_predicted_temperature({}_epochs_lr_{}_with_sixth_hour).png'.format(epochs, lr))
# plt.show()
#
#
# # METRICS
# MAPE = mean_absolute_percentage_error(y_lab, y_pred)
# MSE = mean_squared_error(y_lab, y_pred)
# MAE = mean_absolute_error(y_lab, y_pred)
# R2 = r2_score(y_lab, y_pred)
#
# print('MAPE:%0.5f%%'%MAPE)
# print('MSE:', MSE.item())
# print('MAE:', MAE.item())
# print('R2:', R2.item())
#
#
# plt.scatter(y_lab, y_pred,  color='k', edgecolor= 'white', linewidth=1, alpha=0.5)
# plt.text(18, 24.2, 'MAPE: {:.3f}'.format(MAPE), fontsize=15, bbox=dict(facecolor='red', alpha=0.5))
# plt.text(18, 26.2, 'MSE: {:.3f}'.format(MSE), fontsize=15, bbox=dict(facecolor='green', alpha=0.5))
# plt.text(18, 28.2, 'MAE: {:.3f}'.format(MAE), fontsize=15, bbox=dict(facecolor='blue', alpha=0.5))
# plt.plot([18, 28], [18, 28], color='red')
# plt.grid(b=True, which='major', color='#666666', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.xlabel('Real Temperature [°C]')
# plt.ylabel('Predicted Temperature [°C]')
# plt.title("6h prediction: Prediction distribution", size=15)
# # plt.savefig('immagini/2015/mlp/MLP_prediction_distribution({}_epochs_lr_{}_with_sixth_hour).png'.format(epochs, lr))
# plt.show()
#
#
#












