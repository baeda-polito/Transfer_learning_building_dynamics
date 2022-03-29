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
import optuna
from optuna.trial import TrialState
import plotly

# ___________________________________________IMPORT AND NORMALIZATION___________________________________________________
list_1year = ['2016', '2017']
# list_3_years = ['1990', '1991', '1992']
# list_5_years = ['1990', '1991', '1992', '1993', '1994']
# list_10_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999']

clm = '3C'
eff = 'Standard'
occ = 'run_1'

df = import_file(clm, eff, list_1year, occ)
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

def define_period(df, time):
    if time == 'week':
        l_train = 1008+48
        l_val = int(l_train*2)
        df_def = df[:int(l_train*3)]
    if time == 'month':
        l_train = 4464
        l_val = int(l_train * 2)
        df_def = df[:int(l_train * 3)]
    if time == 'year':
        l_train = int(0.5 * len(df))  # 31536 (per un anno)
        l_val = int(l_train + 0.02 * len(df))  # da 31536 a 42048, cioè 10512 valori (per un anno)
        df_def = df

    return df_def, l_train, l_val

# time = 'month'
df, l_train, l_val = define_period(df, time='year')


# ______________________________________Datasets_preprocessing__________________________________________________________
period = 6
# l_train = int(0.8 * len(df)) # 31536 (per un anno)
# l_val = int(l_train + 0.2*len(df)) # da 31536 a 42048, cioè 10512 valori (per un anno)
# l_val = int(l_train+0.05*len(df)) # da 31536 a 42048, cioè 10512 valori (per un anno)

train_df, val_df, test_df = create_data(df=df, col_name='CONFROOM_BOT_1 ZN:Zone Mean Air Temperature[C]', l_train=l_train, l_val=l_val, period=period)
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

bestmape = 5

# ________________________________________________DEFINE OPTUNA FUNCTION________________________________________________
def define_model(trial, batch_size):

    # num_layers = trial.suggest_int("n_layers", 3, 7, step=2)
    num_layers = 3
    num_hidden = trial.suggest_int("n_hidden", 70, 100, step=10)

    class LSTM(nn.Module):

        def __init__(self, num_classes, input_size, hidden_size, num_layers):
            super(LSTM, self).__init__()

            self.num_classes = num_classes
            self.num_layers = num_layers
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.seq_length = lookback

            self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                                num_layers=self.num_layers, batch_first=True)

            self.fc = nn.Linear(self.hidden_size, self.num_classes)
            # self.af = nn.ReLU()
            # self.fc1 = nn.Linear(10, self.num_classes)

        def forward(self, x, h):
            batch_size, seq_len, _ = x.size()
            # h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
            # c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
            # Propagate input through LSTM
            out, h = self.lstm(x, h)
            out = out[:, -1, :]
            # h_out = h_out.view(-1, self.hidden_size)
            out = self.fc(out)
            # out = self.af(out)
            # out = self.fc1(out)

            return out, h

        def init_hidden(self, batch_size):
            hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            hidden = (hidden_state, cell_state)  # HIDDEN is defined as a TUPLE
            return hidden


    # __________________________________________________PARAMETERS______________________________________________________
    lookback = 48
    n_features = train_X.shape[2]
    n_timesteps = train_X.shape[1]
    n_outputs = 6

    # train_batch_size = 400
    train_data = TensorDataset(train_X, train_Y)
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)

    # val_batch_size = 240
    # val_data = TensorDataset(val_X, val_Y)
    # val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)

    # test_batch_size = 240  # devo mettere il batch ad 1 perchè così ad ogni batch mi appendo il primo dei 6 valori predetti
    test_data = TensorDataset(test_X, test_Y)
    test_dl = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)

    # num_hidden = 40
    lstm_net = LSTM(num_classes=n_outputs, input_size=n_features, hidden_size=num_hidden, num_layers=num_layers)

    return lstm_net, train_data, train_dl, test_data, test_dl



def objective(trial):

    batch_size = trial.suggest_int("batch_size", 800, 1000, step=100)
    lr = trial.suggest_float("lr", 7e-3, 85e-4, log=True)
    # epochs = trial.suggest_int("epochs", 80, 120, step=20)
    epochs = 90
    #optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])


    # initialize the network,criterion and optimizer
    lstm_net, train_data, train_dl, test_data, test_dl = define_model(trial, batch_size)
    # lstm_net = lstm_net.to(device)
    # optimizer = getattr(optim, optimizer_name)(mv_net.parameters(), lr=lr)
    optimizer = torch.optim.Adam(lstm_net.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()  # reduction='sum' created huge loss value

    # ________________________________________________TRAIN LOOP_____________________________________________________

    lstm_net.train()

    TRAIN_LOSS = []
    VAL_LOSS = []

    for t in range(epochs):

        # TRAINING LOOP
        loss = []
        h = lstm_net.init_hidden(batch_size)  # hidden state is initialized at each epoch
        for x, label in train_dl:
            h = lstm_net.init_hidden(batch_size) # since the batch is big enough, a stateless mode is used (also considering the possibility to shuffle the training examples, which increase the generalization ability of the network)
            h = tuple([each.data for each in h])
            output, h = lstm_net(x.float(), h)
            #label = label.unsqueeze(1) # utilizzo .unsqueeze per non avere problemi di dimensioni
            loss_c = criterion(output, label.float())
            optimizer.zero_grad()
            loss_c.backward()
            optimizer.step()
            loss.append(loss_c.item())
        TRAIN_LOSS.append(np.sum(loss)/batch_size)
        # if mode == 'tuning':
        #     lr_scheduler.step()
        # print("Epoch: %d, training loss: %1.5f" % (train_episodes, LOSS[-1]))
        print('Epoch : ', t, 'Training Loss : ', TRAIN_LOSS[-1])

        # # VALIDATION LOOP
        # val_loss = []
        # h = lstm_net.init_hidden(batch_size)
        # for inputs, labels in val_dl:
        #     h = tuple([each.data for each in h])
        #     val_output, h = lstm_net(inputs.float(), h)
        #     #val_labels = labels.unsqueeze(1) # CAPIRE SE METTERLO O NO
        #     val_loss_c = criterion(val_output, labels.float())
        #     val_loss.append(val_loss_c.item())
        # # VAL_LOSS.append(val_loss.item())
        # VAL_LOSS.append(np.sum(val_loss)/batch_size)
        # print('Epoch : ', t, 'Training Loss : ', TRAIN_LOSS[-1], 'Validation Loss :', VAL_LOSS[-1])

    # ________________________________________________TEST LOOP_________________________________________________________
    lstm_net.eval()
    h = lstm_net.init_hidden(batch_size)
    y_pred = []
    y_lab = []
    # y_lab6 = []
    # y_pred6 = []
    for inputs, labels in test_dl:
        h = tuple([each.data for each in h])
        test_output, h = lstm_net(inputs.float(), h)
        # labels = labels.unsqueeze(1)

        # RESCALE OUTPUTS
        test_output = test_output.detach().numpy()
        # test_output = np.reshape(test_output, (-1, 1))
        test_output = min_T + test_output * (max_T - min_T)

        # RESCALE LABELS
        labels = labels.detach().numpy()
        # labels = np.reshape(labels, (-1, 1))
        labels = min_T + labels * (max_T - min_T)

        y_pred.append(
            test_output[:, 0])  # test_output[0] per appendere solo il primo dei valori predetti ad ogni step
        y_lab.append(labels[:, 0])  # labels[0] per appendere solo il primo dei valori predetti ad ogni step
        # y_pred6.append(test_output[:, 5])
        # y_lab6.append(labels[:, 5])

    # METRICS
    # MSE
    MSE = mean_squared_error(y_lab, y_pred)

    # MAE
    MAE = mean_absolute_error(y_lab, y_pred)

    # MAPE
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    MAPE = mean_absolute_percentage_error(y_lab, y_pred)

    # TODO: fit the weights saving

    if MAPE < bestmape:
        best_mape = MAPE
        train_year = '2016'
        torch.save(lstm_net.state_dict(), 'weights/OPTUNA_LSTM_train_on_' + train_year + '_epochs_' + str(epochs) + '_lr_' + str(lr) + '_MAPE_' + str(best_mape) + '.pth')

    metrics = pd.DataFrame([[MSE, MAE, MAPE]], columns=['MSE', 'MAE', 'MAPE'])

    excel_path = 'C:\\Users\\ricme\\Desktop\\Politecnico\\Tesi magistrale\\TL_coding\\meta_data\\code\\thesis_project\\optuna_img\\lstm\\optuna_lstm_results.csv.csv'
    # excel_path = 'E:\\tesi_messina\\thesis_project\\optuna_img\\lstm_img\\optuna_metrics.csv'

    if not os.path.exists(excel_path):
        metrics.to_csv(path_or_buf=excel_path,
                      sep=',', decimal='.', index=False)
    else:
        runsLog = pd.read_csv(excel_path, sep=',', decimal='.')
        runsLog = runsLog.append(metrics)
        runsLog.to_csv(path_or_buf=excel_path,
                       sep=',', decimal='.', index=False)


    # trial.report(MAPE_train, epochs)

    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return MAPE


if __name__ == "__main__":
    # study = optuna.create_study(directions=["minimize", "minimize"])
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2, timeout=None)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    hyperparam_import = optuna.importance.get_param_importances(study, evaluator=None, params=None, target=None)
    print(hyperparam_import)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


    fig1 = optuna.visualization.plot_param_importances(study)
    fig1.show()
    fig2 = optuna.visualization.plot_slice(study)
    fig2.show()

# lstm_metrics = pd.read_csv('otuna_img/lstm/optuna_lstm_results.csv', encoding='latin1')





