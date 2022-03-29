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
from functions import import_file, min_max_T, normalization, create_data, split_multistep_sequences, mean_absolute_percentage_error, save_file
# test_metrics_norelu = pd.read_csv('ML/1_year/1_year/CONFROOM_BOT_1_3C_Standard_run_1_ML_1_year_1_year_testmetrics_withoutrelu.csv', encoding='latin1')
# test_metrics_relu = pd.read_csv('ML/1_year/1_year/CONFROOM_BOT_1_3C_Standard_run_1_ML_1_year_1_year_testmetrics_with_relu.csv', encoding='latin1')


# no relu
# ML_source = pd.read_csv('ML/1_year/1_year/CONFROOM_BOT_1_3C_Standard_run_1_ML_1_year_1_year_testmetrics_withoutrelu.csv')



# ZONES: CONFROOM_BOT_1, CONFROOM_MID_2, ENCLOSEDOFFICE_BOT_2, OPENOFFICE_BOT_3
# EFF: Low, Standard, High
# OCC: run_1, run_2, run_3, run_4, run_5



# todo: CONFROOM_MID_2 ---> TRANSFER OK
CR_MID2_ML_test = pd.read_csv('ML/1_month/1_month/CONFROOM_MID_2_3C_Standard_run_1_ML_1_month_1_month_test_metrics.csv', encoding = 'latin1') # PEGGIO DI TUTTI GLI ALTRI CASI
# From original punto 3
CR_MID2_FE_test = pd.read_csv('TL/1_year/1_month/punto_3/CONFROOM_MID_2_3C_Standard_run_1_fe_1_year_1_month_test_metrics.csv', encoding='latin1')
CR_MID2_WI_test = pd.read_csv('TL/1_year/1_month/punto_3/CONFROOM_MID_2_3C_Standard_run_1_wi_1_year_1_month_test_metrics.csv', encoding='latin1')
CR_MID2_FE_test_lr = pd.read_csv('TL/1_year/1_month/CONFROOM_MID_2_3C_Standard_run_1_fe_1_year_1_month_test_metrics_lr_0.002_stepsize_70.csv', encoding='latin1')
CR_MID2_WI_test_lr = pd.read_csv('TL/1_year/1_month/CONFROOM_MID_2_3C_Standard_run_1_wi_1_year_1_month_test_metrics_lr_0.002_stepsize_70.csv', encoding='latin1')



# todo: ENCLOSEDOFFICE_BOT_2 ---> TRANSFER NO

EO_BOT2_ML_test = pd.read_csv('ML/1_month/1_month/ENCLOSEDOFFICE_BOT_2_3C_Standard_run_1_ML_1_month_1_month_test_metrics.csv', encoding = 'latin1') # PEGGIO DI TUTTI GLI ALTRI CASI
# From original punto 3
EO_BOT2_FE_tesT = pd.read_csv('TL/1_year/1_month/punto_3/ENCLOSEDOFFICE_BOT_2_3C_Standard_run_1_fe_1_year_1_month_test_metrics.csv', encoding='latin1')
EO_BOT2_WI_tesT = pd.read_csv('TL/1_year/1_month/punto_3/ENCLOSEDOFFICE_BOT_2_3C_Standard_run_1_wi_1_year_1_month_test_metrics.csv', encoding='latin1')


# todo: OPENOFFICE_BOT_3 ---> TRANSFER NO
OO_BOT3_ML_test = pd.read_csv('ML/1_month/1_month/OPENOFFICE_BOT_3_3C_Standard_run_1_ML_1_month_1_month_test_metrics.csv', encoding = 'latin1') # PEGGIO DI TUTTI GLI ALTRI CASI
# From original punto 3
OO_BOT3_FE_test = pd.read_csv('TL/1_year/1_month/punto_3/OPENOFFICE_BOT_3_3C_Standard_run_1_fe_1_year_1_month_test_metrics.CSV', encoding='latin1')
OO_BOT3_WI_test = pd.read_csv('TL/1_year/1_month/punto_3/OPENOFFICE_BOT_3_3C_Standard_run_1_wi_1_year_1_month_test_metrics.csv', encoding='latin1')
OO_BOT3_FE_test_lr = pd.read_csv('TL/1_year/1_month/OPENOFFICE_BOT_3_3C_Standard_run_1_fe_1_year_1_month_lr_0.0007703248894562769_epochs_20_test_metrics.csv', encoding='latin1')
OO_BOT3_WI_test_lr = pd.read_csv('TL/1_year/1_month/OPENOFFICE_BOT_3_3C_Standard_run_1_wi_1_year_1_month_lr_0.0007703248894562769_epochs_20_test_metrics.csv', encoding='latin1')

