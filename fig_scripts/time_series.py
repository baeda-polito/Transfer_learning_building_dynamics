
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

#
# # ____________________________________________ALL TESTING METRICS COMPARING_____________________________________________
# # Import file
# technique = 'ML'
# train_period = '1_month'
# test_period = '1_month'
# file = 'ENCLOSEDOFFICE_BOT_2_3C_Standard_run_1_fe_1_year_1_month_lr_0.002_epochs_80_test_metrics'
#
# df = pd.read_csv('results/{}/{}/{}/{}.csv'.format(technique, train_period, test_period, file), encoding='latin1')
#
#
# def plotting_3_metrics(df, file):
#     dfnp = df.to_numpy()
#     dfnp = dfnp.T  # Transposition to vertical columns
#
#     # Division of dataframe per metric
#     df_mae = pd.DataFrame(dfnp[:7])
#     df_mse = pd.DataFrame(dfnp[7:14])
#     df_mape = pd.DataFrame(dfnp[14:])
#
#     # Concatening of metrics and step names
#     df_def = pd.concat([df_mae, df_mse, df_mape], axis=1)
#     df_def.columns = ['MAE', 'MSE', 'MAPE']
#     df_def['steps'] = ['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6', 'Mean']
#
#     items = file.split('_')[:12]
#
#     # Plotting
#     df_def.plot(x="steps", y=["MAE", "MSE", "MAPE"], kind="bar")
#     plt.xticks(rotation=0)
#     plt.xlabel('Prediction steps')
#     plt.title('{} on {}: Prediction metrics'.format(technique, ' '.join(test_period.split('_'))), size=15)
#     plt.savefig('results/img/'+'_'.join(items)+'_metrics.png')
#     plt.show()
#
# plotting_3_metrics(df, file)
#


# ___________________________________________MAE COMPARING BETWEEN DIFFERENT TECHNIQUES_________________________________
# Import file
# CONFROOM_MID_2, OPENOFFICE_BOT_3, ENCLOSEDOFFICE_BOT_2
test_period = '1_month'
file1 = 'ENCLOSEDOFFICE_BOT_2_3C_Standard_run_1_ML_1_month_1_month_test_metrics'
file2 = 'ENCLOSEDOFFICE_BOT_2_3C_Standard_run_1_fe_1_year_1_month_test_metrics'
file3 = 'ENCLOSEDOFFICE_BOT_2_3C_Standard_run_1_wi_1_year_1_month_test_metrics'

df1 = pd.read_csv('other_results/ML/{}/{}.csv'.format(test_period, file1), encoding='latin1')
df2 = pd.read_csv('other_results/TL/1_year/{}/{}.csv'.format(test_period, file2), encoding='latin1')
df3 = pd.read_csv('other_results/TL/1_year/{}/{}.csv'.format(test_period, file3), encoding='latin1')


def comparing_tech(df1, df2, df3):
    df_ML = df1.to_numpy()
    df_FE = df2.to_numpy()
    df_WI = df3.to_numpy()

    df_ML = df_ML.T  # Transposition to vertical columns
    df_FE = df_FE.T  # Transposition to vertical columns
    df_WI = df_WI.T  # Transposition to vertical columns

    # Concatening of metrics and step names
    df_ML_mae = pd.DataFrame(df_ML[:6])
    df_FE_mae = pd.DataFrame(df_FE[:6])
    df_WI_mae = pd.DataFrame(df_WI[:6])

    # Concatening of metrics and step names
    df_ML_mse = pd.DataFrame(df_ML[7:13])
    df_FE_mse = pd.DataFrame(df_FE[7:13])
    df_WI_mse = pd.DataFrame(df_WI[7:13])

    # Concatening of metrics and step names
    df_def_mae = pd.concat([df_ML_mae, df_FE_mae, df_WI_mae], axis=1)
    df_def_mae.columns = ['ML', 'TL-FE', 'TL-WI']
    df_def_mae['steps'] = ['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6']

    # Concatening of metrics and step names
    df_def_mse = pd.concat([df_ML_mse, df_FE_mse, df_WI_mse], axis=1)
    df_def_mse.columns = ['ML', 'TL-FE', 'TL-WI']
    df_def_mse['steps'] = ['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6']

    # items = file1.split('_')[:7]

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    # plt.yticks(np.arange(0,0.50,0.05))
    df_def_mae.plot(x="steps", y=['ML', 'TL-FE', 'TL-WI'], kind="bar", color=['orange', sns.xkcd_rgb["mid blue"], sns.xkcd_rgb["soft green"]], edgecolor='black', ax=axs[0])
    # axs[0].set_ylabel('MAE [°C]', size=20)
    # axs[0].set_xlabel('Prediction steps', size=20)
    axs[0].grid(axis='y')
    axs[0].tick_params('x', labelrotation=0)
    axs[0].set_xticklabels(labels=['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6'], size=18)
    axs[0].set_ylim([0, 0.45])
    axs[0].tick_params(axis='y', labelsize=18)
    axs[0].set_xlabel(None)
    axs[0].set_title('MAE [°C]', size=20)
    df_def_mse.plot(x="steps", y=['ML', 'TL-FE', 'TL-WI'], kind="bar", color=['orange', sns.xkcd_rgb["mid blue"], sns.xkcd_rgb["soft green"]], edgecolor='black', ax=axs[1])
    # axs[1].set_ylabel('MSE', size=20)
    axs[1].set_xlabel(None)
    axs[1].grid(axis='y')
    axs[1].tick_params('x', labelrotation=0)
    axs[1].set_xticklabels(labels=['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6'], size=18)
    axs[1].set_ylim([0, 0.45])
    axs[1].tick_params(axis='y', labelsize=18)
    axs[1].set_title('MSE', size=20)
    plt.suptitle('{}: MAE and MSE comparison between ML / TL-FE / TL-WI'.format(' '.join(file1.split('_')[:3])), size=20)
    plt.tight_layout()
    plt.savefig('fig_scripts/img/'+'_'.join(file1.split('_')[:7])+'_mae_mse.png')
    plt.show()

    return df_ML, df_FE, df_WI

df_ML, df_FE, df_WI = comparing_tech(df1, df2, df3)
# navajowhite,lightsteelblue, greenyellow







# __________________________________________________TRAINING LOSS_______________________________________________________

# CONFROOM_MID_2, OPENOFFICE_BOT_3, ENCLOSEDOFFICE_BOT_2

test_period = '1_month'
loss1 = 'ENCLOSEDOFFICE_BOT_2_3C_Standard_run_1_ML_1_month_1_month_train_loss'
loss2 = 'ENCLOSEDOFFICE_BOT_2_3C_Standard_run_1_fe_1_year_1_month_train_loss'
loss3 = 'ENCLOSEDOFFICE_BOT_2_3C_Standard_run_1_wi_1_year_1_month_train_loss'

loss_df1 = pd.read_csv('other_results/ML/{}/{}.csv'.format(test_period, loss1), encoding='latin1')
loss_df2 = pd.read_csv('other_results/TL/1_year/{}/{}.csv'.format(test_period, loss2), encoding='latin1')
loss_df3 = pd.read_csv('other_results/TL/1_year/{}/{}.csv'.format(test_period, loss3), encoding='latin1')
df_loss = pd.concat([loss_df1, loss_df2, loss_df3], axis=1).dropna()
del df_loss['Unnamed: 0']
df_loss.columns = ['ML', 'TL-FE', 'TL-WI']

items = loss1.split('_')[:7]
#['orange', 'forestgreen', 'gold']
plt.plot(df_loss['ML'], c='orange', label='ML')
plt.plot(df_loss['TL-FE'], c='forestgreen', label='TL-FE')
plt.plot(df_loss['TL-WI'], c='gold', label='TL-WI', alpha=0.8)
plt.grid()
plt.yscale('log')
plt.legend()
plt.title('{}: ML / TL-FE / TL-WI training loss comparing'.format(' '.join(loss1.split('_')[:3])), size=12)
plt.xlabel('Epochs', size=13)
plt.tight_layout()
# plt.savefig('results/img/' + '_'.join(items) + '_loss.png')
plt.show()
# navajowhite,lightsteelblue, greenyellow






# =============================================== NEW CODE =============================================================



import pandas as pd
import os
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
import pydotplus
from pydotplus import graph_from_dot_data
import joypy
from matplotlib import cm
import numpy as np


PATH = "results"
EXT = "*test_metrics.csv"

#https://perials.com/getting-csv-files-directory-subdirectories-using-python/
#https://www.geeksforgeeks.org/python-pandas-split-strings-into-two-list-columns-using-str-split/



all_csv_files = []
df = pd.DataFrame()
for path, subdir, files in os.walk(PATH):
    for file in glob(os.path.join(path, EXT)):
        all_csv_files.append(file)
        file_df = pd.read_csv(file, header=0)
        file_df['file_name'] = file
        df = df.append(file_df)

df3 = df['file_name'].str.rsplit("_", 13,expand=True)[:]
df4=pd.DataFrame()
# df4['Zone'] = df3[1]+df3[2]
df4['Climate'] = df3[3]
df4['Efficiency'] = df3[4]
df4['Occupancy'] = df3[5]+df3[6]
df4['Technique'] = df3[7]
df4['Training'] = df3[8]+df3[9]
df4['Testing'] = df3[10]+df3[11]

df_test_metrics = pd.concat([df4, df], axis=1)

df_test_metrics = df_test_metrics.reset_index(drop=True)

# Delete the nan
df_test_metrics.MAE_avg.fillna(df_test_metrics.MAE, inplace=True)
df_test_metrics.MAPE_avg.fillna(df_test_metrics.MAPE, inplace=True)
df_test_metrics.MSE_avg.fillna(df_test_metrics.MSE, inplace=True)
del df_test_metrics['MAE']
del df_test_metrics['MAPE']
del df_test_metrics['MSE']


# Split dataset based on technique
df_ML = df_test_metrics.loc[df_test_metrics['Technique'] == 'ML']
df_fe = df_test_metrics.loc[df_test_metrics['Technique'] == 'fe']
df_wi = df_test_metrics.loc[df_test_metrics['Technique'] == 'wi']


# DATA SET PER TRAINING PERIOD
# df_1month = df_test_metrics.loc[df_test_metrics['Training'] == '1month']
# df_1week = df_test_metrics.loc[df_test_metrics['Training'] == '1week']
# df_1year = df_test_metrics.loc[df_test_metrics['Training'] == '1year']

df_1week_ML = df_ML.loc[df_test_metrics['Training'] == '1week']
df_1week_fe = df_fe.loc[df_test_metrics['Testing'] == '1week']
df_1week_wi = df_wi.loc[df_test_metrics['Testing'] == '1week']
df_1week = pd.concat([df_1week_ML, df_1week_fe, df_1week_wi], axis=0)


df_1month_ML = df_ML.loc[df_test_metrics['Training'] == '1month']
df_1month_fe = df_fe.loc[df_test_metrics['Testing'] == '1month']
df_1month_wi = df_wi.loc[df_test_metrics['Testing'] == '1month']
df_1month = pd.concat([df_1month_ML, df_1month_fe, df_1month_wi], axis=0)


df_1year_ML = df_ML.loc[df_test_metrics['Training'] == '1year']
df_1year_fe = df_fe.loc[df_test_metrics['Testing'] == '1year']
df_1year_wi = df_wi.loc[df_test_metrics['Testing'] == '1year']
df_1year = pd.concat([df_1year_ML, df_1year_fe, df_1year_wi], axis=0)


# SWARM PLOT____________________________________________________________________________________________________________
ax = sns.swarmplot(x="Technique", y="MAE_avg", data=df_test_metrics, size=3.5)
# ax = sns.violinplot(x="Technique", y="MAE1", data=df_test_metrics, inner=None)
plt.grid()
plt.title('Mean MAE distribution per technique', size=15)
# plt.savefig('results/img/Mean_MAE_distribution_per_technique.png')
plt.show()


ax1 = sns.swarmplot(x="Testing", y="MAE_avg", hue="Technique", data=df_test_metrics, size=3.5)
# ax = sns.violinplot(x="Testing", y="MAE1", data=df_test_metrics, inner=None)
plt.grid()
plt.title('Mean MAE distribution per technique and testing period', size=15)
# plt.savefig('results/img/Mean_MAE_distribution_per_technique_and_testing_period.png')
plt.show()


ax1 = sns.swarmplot(x="Climate", y="MAE_avg", hue="Technique", data=df_test_metrics, size=3.5)
# ax = sns.violinplot(x="Testing", y="MAE1", data=df_test_metrics, inner=None)
plt.grid()
plt.title('Mean MAE distribution per technique and climate', size=15)
# plt.savefig('results/img/Mean_MAE_distribution_per_technique_and_climate.png')
plt.show()


ax = sns.swarmplot(x="Technique", y="MAPE_avg", data=df_test_metrics)
plt.show()

ax = sns.swarmplot(x="Testing", y="MAE1",hue="Technique", data=df_test_metrics)
plt.show()

ax = sns.swarmplot(x="Testing", y="MAE1",hue="Efficiency", data=df_test_metrics)
plt.show()

ax = sns.swarmplot(x="Testing", y="MAE1",hue="Climate", data=df_test_metrics)
plt.show()

ax = sns.swarmplot(y="MAE1",hue="Efficiency", data=df_test_metrics)
plt.show()
# ______________________________________________________________________________________________________________________


# sim = df_test_metrics.groupby(['Technique' ,'Training', 'Testing', 'Climate', 'Occupancy', 'Efficiency']).size()



# values = list(df_test_metrics.columns)
# x = df_test_metrics[values]
# df_test_metrics.target()
#
# Y = df_test_metrics.columns
# clf = LinearRegression()
# clf = clf.fit(X, Y)
# tree.plot_tree(clf)
#
#
# tree = DecisionTreeClassifier(criterion='gini')
# one_hot_data = pd.get_dummies(df_test_metrics[['Zone', 'Climate', 'Efficiency', 'Occupancy', 'Technique', 'Training', 'Testing']], drop_first=True)
# # y_data = pd.get_dummies(df_test_metrics[['MAE1']])
# tree.fit(one_hot_data, df_test_metrics['MAE1']) # per funzionare, anche MAE1 dev'essere categorica
#

# # Categorize the MAE_avg
# for i in range(0, len(df_test_metrics)):
#     if df_test_metrics['MAE_avg'][i] >=0 and df_test_metrics['MAE_avg'][i] < 0.87:
#         df_test_metrics['MAE_avg'][i] = 'High_performance'
#         continue
#     if df_test_metrics['MAE_avg'][i] >=0.87 and df_test_metrics['MAE_avg'][i] < 1.74:
#         df_test_metrics['MAE_avg'][i] = 'Mid_performance'
#         continue
#     if df_test_metrics['MAE_avg'][i] >=1.74:
#         df_test_metrics['MAE_avg'][i] = 'Low_performance'
#         continue
#
#

# import re
# tot_high = len(re.findall("High_performance", df_test_metrics['MAE1']))
#
# from collections import Counter
# h=0
# for i in range(0, len(df_test_metrics)):
#     counter = Counter(df_test_metrics['MAE1'][i])
#     h += counter['High_performance']


# HIST PLOT ____________________________________________________________________________________________________________

def steps_and_tech(df_ML, df_FE, df_WI, training_period):
    # df_ML = df.loc[['MAE1', 'MAE2','MAE3','MAE4', 'MAE5', 'MAE6']].to_numpy()
    df_ML = df_ML[['MAE1', 'MAE2', 'MAE3', 'MAE4', 'MAE5', 'MAE6']].mean()
    df_FE = df_FE[['MAE1', 'MAE2', 'MAE3', 'MAE4', 'MAE5', 'MAE6']].mean()
    df_WI = df_WI[['MAE1', 'MAE2', 'MAE3', 'MAE4', 'MAE5', 'MAE6']].mean()

    # df = df.T  # Transposition to vertical columns

    # Concatening of metrics and step names
    # df_ML = pd.DataFrame(df_ML[:6])

    # Concatening of metrics and step names
    df_def = pd.concat([df_ML, df_FE, df_WI], axis=1)
    df_def.columns = ['ML', 'TL-FE', 'TL-WI']
    df_def['steps'] = ['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6']

    # items = file1.split('_')[:7]

    # Plotting
    df_def.plot(x="steps", y=['ML', 'TL-FE', 'TL-WI'], kind="bar", color=['navajowhite', 'lightsteelblue', 'greenyellow'])
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.xlabel('Prediction steps')
    plt.title('{}: MAE comparison between ML / TL-FE / TL-WI step by step'.format(' '.join(training_period.split('_'))), size=12)
    # plt.savefig('results/img/'+'_'.join(training_period.split('_'))+'_mae_step_by_step.png')
    plt.show()

    return df_def, df_ML, df_FE, df_WI

# For 1 week, 1 month and 1 year
df_def_1week, df_ML_1week, df_FE_1week, df_WI_1week = steps_and_tech(df_1week_ML, df_1week_fe, df_1week_wi, training_period='1_week')
df_def_1month, df_ML_1month, df_FE_1month, df_WI_1month = steps_and_tech(df_1month_ML, df_1month_fe, df_1month_wi, training_period='1_month')
df_def_1year, df_ML_1year, df_FE_1year, df_WI_1year = steps_and_tech(df_1year_ML, df_1year_fe, df_1year_wi, training_period='1_year')


# todo: l'asse delle x non rispetta i valori di mae ottenuti, le aree rimangono le stesse qualunque sia l'intervallo di valori in x

def plot_joyplot(df, training_period):
    plt.figure(figsize=(16,10), dpi=80)
    fig, axes = joypy.joyplot(df, by="Technique", column=['MAE1', 'MAE2', 'MAE3', 'MAE4', 'MAE5', 'MAE6'], ylim='own', figsize=(12, 8), legend=True, alpha=0.4, overlap=0.2) #, grid=True
    plt.grid(which='minor')
    # plt.suptitle('Mean MAE Density distribution for step by step {} prediction'.format(' '.join(training_period.split('_'))), size=22)
    plt.title('Training period: ' + ' '.join(training_period.split('_')), size=25)
    # plt.ylabel(['ML', 'fe', 'wi'], size=30, color='black')
    # plt.legend(labels=['MAE 1', 'MAE 2', 'MAE 3', 'MAE 4', 'MAE 5', 'MAE 6'], fontsize=20)
    # plt.legend(prop={'size':13.89})
    plt.xlabel('MAE [°C]', fontsize=20, color='black')
    plt.xticks(fontsize=20)
    # plt.yticks(fontsize=18)
    plt.xlim(left=0)
    plt.grid()
    plt.tight_layout()
    # plt.savefig('results/img/{}_density_distribution_step_by_step.png'.format(training_period))
    plt.show()


# , color=['#274e13', 'red', '#f1c232', 'green', 'blue', 'yellow']
# color=['GnBu', 'PuBu', 'YIGnBu', 'PuBuGn', 'BuGn', 'YIGn']

plot_joyplot(df_1month, training_period='1_month')
plot_joyplot(df_1week, training_period='1_week')
plot_joyplot(df_1year, training_period='1_year')



# BOXPLOT_______________________________________________________________________________________________________________
# MAE1
sns.boxplot(x=df_test_metrics['Technique'], y = df_test_metrics['MAE1'])
plt.title('First step MAE distribution for different techniques', size=15)
plt.grid()
# plt.savefig('results/img/First_step_MAE_distribution_for_different_techniques.png')
plt.show()

# MAE_avg
palette = {'ML': 'orange', 'fe': sns.xkcd_rgb["mid blue"], 'wi': sns.xkcd_rgb["soft green"]}
sns.boxplot(x=df_test_metrics['Technique'], y = df_test_metrics['MAE_avg'], palette=palette)
plt.suptitle('Mean MAE distribution for different techniques', size=15)
plt.title('Average MAE [°C]', size=18)
plt.yticks(fontsize=15)
plt.xticks(fontsize=17)
plt.ylabel(None)
plt.xlabel(None)
plt.grid()
# plt.savefig('results/img/Mean_MAE_distribution_for_different_techniques.png')
plt.show()


# Density plot
fig = sns.kdeplot(df_ML['MAE_avg'], shade=True, color="r", label='ML')
fig = sns.kdeplot(df_fe['MAE_avg'], shade=True, color="g", label='FE')
fig = sns.kdeplot(df_wi['MAE_avg'], shade=True, color="b", label='WI')
plt.xlim(left=0)
plt.legend()
plt.title('Mean MAE density distribution for technique', size=15)
plt.grid()
# plt.savefig('results/img/Mean_MAE_density_distribution_for_technique.png')
plt.show()


fig = sns.kdeplot(df_1week['MAE_avg'], shade=True, color="b", label='1 week')
fig = sns.kdeplot(df_1month['MAE_avg'], shade=True, color="g", label='1 month')
fig = sns.kdeplot(df_1year['MAE_avg'], shade=True, color="r", label='1 year')
plt.xlim(left=0)
plt.legend()
plt.title('Mean MAE density distribution for testing period', size=15)
plt.grid()
# plt.savefig('results/img/Mean_MAE_density_distribution_for_testing_period.png')
plt.show()


# # Plotting
# df_ML.plot(x="MAE1", y=[], kind="bar", color=['navajowhite', 'lightsteelblue', 'greenyellow'])
# plt.xticks(rotation=0)
# plt.grid(axis='y')
# plt.xlabel('Prediction steps')
# plt.title('{}: MAE comparison between ML / TL-FE / TL-WI'.format(' '.join(file1.split('_')[:3])), size=12)
# # plt.savefig('results/img/'+'_'.join(file1.split('_')[:7])+'_mae.png')
# plt.show()
# ______________________________________________________________________________________________________________________


# todo: facetgrid boxplot seaborn
# todo:histplot, samples nodo foglia

# ['orange', sns.xkcd_rgb["mid blue"], sns.xkcd_rgb["soft green"]]
fig, axs = plt.subplots(1, 3, figsize=(15, 7))
g = sns.boxplot(x="Climate", y="MAE_avg", hue='Technique', data=df_1week, ax=axs[0], palette=['orange', sns.xkcd_rgb["mid blue"], sns.xkcd_rgb["soft green"]])
axs[0].grid(True)
axs[0].set_title('Testing: 1 week', size=20)
axs[0].set_ylabel('MAE avg [°C]', size=17)
axs[0].set_xlabel(None)
axs[0].set_ylim(0, 1.4)
axs[0].set_xticklabels(labels=['1A', '3C', '5A'], size=17)
sns.boxplot(x="Climate", y="MAE_avg", hue='Technique', data=df_1month, ax=axs[1], palette=['orange', sns.xkcd_rgb["mid blue"], sns.xkcd_rgb["soft green"]])
axs[1].grid(True)
axs[1].set_title('Testing: 1 month', size=20)
axs[1].set_ylabel(None)
axs[1].set_xlabel('Climate', size=18)
axs[1].set_ylim(0, 1.4)
axs[1].set_xticklabels(labels=['1A', '3C', '5A'], size=17)
sns.boxplot(x="Climate", y="MAE_avg", hue='Technique', data=df_1year, ax=axs[2], palette=['orange', sns.xkcd_rgb["mid blue"], sns.xkcd_rgb["soft green"]])
axs[2].grid(True)
axs[2].set_title('Testing: 1 year', size=20)
axs[2].set_ylabel(None)
axs[2].set_xlabel(None)
axs[2].set_ylim(0, 1.4)
axs[2].set_xticklabels(labels=['3C', '1A', '5A'], size=17)
plt.suptitle('Mean MAE distribution based on technique, climate and testing period', size=20)
plt.tight_layout()
# plt.savefig('results/img/multiple_boxplot.png')
plt.show()



my_pal = {"ML": "#eab676", "fe": "#154c79", "wi":"#7c0012"}
g = sns.catplot(x="Testing", y="MAE_avg",
                hue="Technique", col="Climate",
                data=df_test_metrics, kind="box",
                height=5, aspect=.7,legend_out= True,palette=my_pal,order=['1week','1month','1month1year'])
# Put the legend out of the figure
plt.xticks(np.arange(3),('1 week','1 month','1 year'))
#plt.legend(labels=['Machine learning','Feature extraction','Weight initialization'],loc='upper right', bbox_to_anchor=(0.5, 0.5))
new_labels = ['Machine learning','Feature extraction','Weight initialization']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)
# sns.move_legend(g, "upper left",ncol=1, bbox_to_anchor=(.12, 0.9))
g.set_ylabels('MAE [°C]')
plt.ylim([0,1.5])
g.refline(y=0.5)
#plt.tight_layout()
# plt.savefig('fig_scripts/img/Mean_MAE_distribution_per_technique_and_testing_period.png')
plt.show()

















ax1 = sns.swarmplot(x="Climate", y="MAE_avg", hue="Technique", data=df_1week, size=3.5)
# ax = sns.violinplot(x="Testing", y="MAE1", data=df_test_metrics, inner=None)
plt.grid()
plt.title('Mean MAE weekly distribution per technique and climate', size=15)
plt.savefig('results/img/Mean_MAE_weekly_distribution_per_technique_and_climate.png')
plt.show()


ax1 = sns.swarmplot(x="Climate", y="MAE_avg", hue="Technique", data=df_1month, size=3.5)
# ax = sns.violinplot(x="Testing", y="MAE1", data=df_test_metrics, inner=None)
plt.grid()
plt.title('Mean MAE monthly distribution per technique and climate', size=15)
plt.savefig('results/img/Mean_MAE_monthly_distribution_per_technique_and_climate.png')
plt.show()


ax1 = sns.swarmplot(x="Climate", y="MAE_avg", hue="Technique", data=df_1year, size=3.5)
# ax = sns.violinplot(x="Testing", y="MAE1", data=df_test_metrics, inner=None)
plt.grid()
plt.title('Mean MAE yearly distribution per technique and climate', size=15)
plt.savefig('results/img/Mean_MAE_yearly_distribution_per_technique_and_climate.png')
plt.show()


ax1 = sns.swarmplot(x="Climate", y="MAE_avg", hue="Technique", data=df_test_metrics, size=3.5)
# ax = sns.violinplot(x="Testing", y="MAE1", data=df_test_metrics, inner=None)
plt.grid()
plt.title('Mean MAE distribution per technique and climate', size=15)
plt.savefig('results/img/Mean_MAE_distribution_per_technique_and_climate.png')
plt.show()


# REGRESSION TREE_______________________________________________________________________________________________________
# Total regression tree
df_dummies = pd.get_dummies(df_test_metrics[['Climate', 'Efficiency', 'Occupancy', 'Technique', 'Training', 'Testing', 'MAE_avg']])
df_dummies = df_dummies.reset_index(drop=True)
X = df_dummies.iloc[:, 1:]
Y = df_dummies.iloc[:, 0]
regressor_1week = DecisionTreeRegressor(random_state=0, min_samples_leaf=10)
tree_tot = regressor_1week.fit(X, Y)
# tree.plot_tree(tree_1week)
# export_graphviz(regressor, out_file='tree.dot')
# class_name = list(pd.DataFrame(Y_1week).columns)
feature_name = list(pd.DataFrame(X).columns)
fig = plt.figure(figsize=(18, 18), dpi=350)
_ = tree.plot_tree(tree_tot, filled=True, max_depth=5, rounded=True, feature_names=feature_name)#  feature_names=X_1week.columns, class_names=Y_1week.columns,
plt.title('Regression Tree for the total dataset', size=30)
plt.savefig('results/img/Regression_Tree_for_the_total_dataset_new.png')
plt.show()


# # Per week
# df_1week_dummies = pd.get_dummies(df_1week[['Climate', 'Efficiency', 'Occupancy', 'Technique', 'Training', 'Testing', 'MAE_avg']])
# df_1week_dummies = df_1week_dummies.reset_index(drop=True)
# X_1week = df_1week_dummies.iloc[:, 1:]
# Y_1week = df_1week_dummies.iloc[:, 0]
# regressor_1week = DecisionTreeRegressor(random_state=0)
# tree_1week = regressor_1week.fit(X_1week, Y_1week)
# # tree.plot_tree(tree_1week)
# # export_graphviz(regressor, out_file='tree.dot')
# # class_name = list(pd.DataFrame(Y_1week).columns)
# feature_name = list(pd.DataFrame(X_1week).columns)
# fig_week = plt.figure(figsize=(25,20))
# _ = tree.plot_tree(tree_1week, filled=True, max_depth=2, rounded=True, feature_names=feature_name)#  feature_names=X_1week.columns, class_names=Y_1week.columns,
# plt.title('Regression Tree for a training period of 1 week', size=30)
# # plt.savefig('results/img/Regression_Tree_for_a_training_period_of_1_week.png')
# plt.show()
#
#
# # Per month
# df_1month_dummies = pd.get_dummies(df_1month[['Climate', 'Efficiency', 'Occupancy', 'Technique', 'Training', 'Testing', 'MAE_avg']])
# df_1month_dummies = df_1month_dummies.reset_index(drop=True)
# X_1month = df_1month_dummies.iloc[:, 1:]
# Y_1month = df_1month_dummies.iloc[:, 0]
# regressor_1month = DecisionTreeRegressor(random_state=0)
# tree_1month = regressor_1month.fit(X_1month, Y_1month)
# # tree.plot_tree(tree_prova)
# # export_graphviz(regressor, out_file='tree.dot')
# fig_1month = plt.figure(figsize=(25,20))
# _ = tree.plot_tree(tree_1month, filled=True, max_depth=2, rounded=True)# , feature_names=X_1month.columns, class_names=Y_1month.columns,
# plt.title('Regression Tree for a training period of 1 month', size=30)
# # plt.savefig('results/img/Regression_Tree_for_a_training_period_of_1_month.png')
# plt.show()
#
#
# # Per year
# df_1year_dummies = pd.get_dummies(df_1year[['Climate', 'Efficiency', 'Occupancy', 'Technique', 'Training', 'Testing', 'MAE_avg']])
# df_1year_dummies = df_1year_dummies.reset_index(drop=True)
# X_1year = df_1year_dummies.iloc[:, 1:]
# Y_1year = df_1year_dummies.iloc[:, 0]
# regressor_1year = DecisionTreeRegressor(random_state=0)
# tree_1year = regressor_1year.fit(X_1year, Y_1year)
# # tree.plot_tree(tree_prova)
# # export_graphviz(regressor, out_file='tree.dot')
# fig_1year = plt.figure(figsize=(25,20))
# _ = tree.plot_tree(tree_1year, filled=True, max_depth=2, rounded=True) # feature_names=X_1year.columns, class_names=Y_1year.columns,
# plt.title('Regression Tree for a training period of 1 year', size=30)
# # plt.savefig('results/img/Regression_Tree_for_a_training_period_of_1_year.png')
# plt.show()

# ______________________________________________________________________________________________________________________


# CLASSIFICATION TREE___________________________________________________________________________________________________
# Categorization of MAE_avg
df_1month['MAE_avg'] = pd.cut(df_1month['MAE_avg'], bins=[0, 0.3, 0.6, 3], include_lowest=True, labels=[1, 2, 3])
df_1week['MAE_avg'] = pd.cut(df_1week['MAE_avg'], bins=[0, 0.3, 0.6, 2.8], include_lowest=True, labels=[1, 2, 3])
df_1year['MAE_avg'] = pd.cut(df_1year['MAE_avg'], bins=[0, 0.3, 0.6, 2.8], include_lowest=True, labels=[1, 2, 3])


# Weekly classification tree
dfcls_1week_dummies = pd.get_dummies(df_1week[['Climate', 'Efficiency', 'Occupancy', 'Technique', 'Testing']])
dfcls_1week_dummies = dfcls_1week_dummies.reset_index(drop=True)
Xcls_1week = dfcls_1week_dummies.iloc[:,:]
# Ycls_1week = dfcls_1week_dummies.iloc[:, -3:]
Ycls_1week = df_1week.iloc[:, 12]
clstree_1week = tree.DecisionTreeClassifier(random_state=0, min_samples_leaf=10)
clstree_1week = clstree_1week.fit(Xcls_1week, Ycls_1week)
tree.plot_tree(clstree_1week)

figcls_1week = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clstree_1week, filled=True, max_depth=5, feature_names=list(pd.DataFrame(Xcls_1week).columns), class_names=['High', 'Medium', 'Low'], rounded=True)
plt.title('Classification Tree for a testing period of 1 week', size=30)
# plt.savefig('results/img/Classification_Tree_for_a_testing_period_of_1_week_new.png')
plt.show()

# class_names = list(Ycls_1week.columns)
# # rappresentazione migliore con graphviz
# dot_data = export_graphviz(clstree_1week, filled=True, rounded=True, max_depth=2,
#                                     class_names=clstree_1week.classes_, # Ycls_1week.columns
#                                     feature_names=Xcls_1week.columns,
#                                     out_file=None)
# graph = graph_from_dot_data(dot_data)
# graph.write_png('results/img/weekly_tree_prova.png')



# Monthly classification tree
dfcls_1month_dummies = pd.get_dummies(df_1month[['Climate', 'Efficiency', 'Occupancy', 'Technique', 'Testing']])
dfcls_1month_dummies = dfcls_1month_dummies.reset_index(drop=True)
Xcls_1month = dfcls_1month_dummies.iloc[:, :]
# Ycls_1month = dfcls_1month_dummies.iloc[:, -3:]
Ycls_1month = df_1month.iloc[:, 12]
clstree_1month = tree.DecisionTreeClassifier(random_state=0, min_samples_leaf=10)
clstree_1month = clstree_1month.fit(Xcls_1month, Ycls_1month)
# tree.plot_tree(clstree_1month)

figcls_1month = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clstree_1month, filled=True, max_depth=5, feature_names=list(pd.DataFrame(Xcls_1month).columns), class_names=['High', 'Medium', 'Low'], rounded=True)
plt.title('Classification Tree for a testing period of 1 month', size=30)
# plt.savefig('results/img/Classification_Tree_for_a_testing_period_of_1_month_new.png')
plt.show()




# Yearly classification tree
dfcls_1year_dummies = pd.get_dummies(df_1year[['Climate', 'Efficiency', 'Occupancy', 'Technique', 'Testing']])
dfcls_1year_dummies = dfcls_1year_dummies.reset_index(drop=True)
Xcls_1year = dfcls_1year_dummies.iloc[:, :]
# Ycls_1year = dfcls_1year_dummies.iloc[:, -3:]
Ycls_1year = df_1year.iloc[:, 12]
clstree_1year = tree.DecisionTreeClassifier(random_state=0, min_samples_leaf=10)
clstree_1year = clstree_1year.fit(Xcls_1year, Ycls_1year)
tree.plot_tree(clstree_1year)

figcls_1year = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clstree_1year, filled=True, max_depth=5, feature_names=list(pd.DataFrame(Xcls_1year).columns), class_names=['High', 'Medium', 'Low'], rounded=True)
plt.title('Classification Tree for a testing period of 1 year', size=30)
# plt.savefig('results/img/Classification_Tree_for_a_testing_period_of_1_year_new.png')
plt.show()

# ______________________________________________________________________________________________________________________



# OCCUPATION PROFILE EXPLORATION________________________________________________________________________________________
occ1 = pd.read_csv('data/CONFROOM_BOT_1_3C_Standard_TMY3_run_1.csv', encoding='latin1')
occ2 = pd.read_csv('data/CONFROOM_BOT_1_3C_Standard_TMY3_run_2.csv', encoding='latin1')
occ3 = pd.read_csv('data/CONFROOM_BOT_1_3C_Standard_TMY3_run_3.csv', encoding='latin1')

occ = pd.concat([occ1['CONFROOM_BOT_1 ZN:Zone People Occupant Count[]'], occ2['CONFROOM_BOT_1 ZN:Zone People Occupant Count[]'], occ3['CONFROOM_BOT_1 ZN:Zone People Occupant Count[]']], axis=1)
occ.columns = ['occ1', 'occ2', 'occ3']
occ[occ > 0] = 1


fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
axs[0].plot(occ[['occ1']], c='g')
axs[0].set_ylabel('Occ 1')
axs[0].grid(True)
axs[1].plot(occ[['occ2']], c='chocolate')
axs[1].set_ylabel('Occ 2')
axs[1].grid(True)
axs[2].plot(occ[['occ3']], c='royalblue')
axs[2].set_ylabel('Occ 3')
axs[2].grid(True)
plt.setp(axs, xlim=(144, 288), yticks=[0, 1])
plt.suptitle('Occupancy profiles', size=15)
plt.xlabel('Time')
# plt.savefig('results/img/occupancy_profiles_1_day.png')
plt.show()



# TEMPERATURE TRENDS____________________________________________________________________________________________________

# TL
real_temp = pd.read_csv('results/temperature_plot/CONFROOM_BOT_1_3C_High_run_2_ML_1_month_1_month_real_temp.csv', encoding='latin1')
ml_temp = pd.read_csv('results/temperature_plot/CONFROOM_BOT_1_3C_High_run_2_ML_1_month_1_month_ML_temp.csv', encoding='latin1')
fe_temp = pd.read_csv('results/temperature_plot/CONFROOM_BOT_1_3C_High_run_2_ML_1_month_1_month_FE_temp.csv', encoding='latin1')
wi_temp = pd.read_csv('results/temperature_plot/CONFROOM_BOT_1_3C_High_run_2_ML_1_month_1_month_WI_temp.csv', encoding='latin1')

# NTL
real_temp_NTL = pd.read_csv('results/temperature_plot/CONFROOM_BOT_1_5A_Low_run_2_ML_1_month_1_month_real_temp.csv', encoding='latin1')
ml_temp_NTL = pd.read_csv('results/temperature_plot/CONFROOM_BOT_1_5A_Low_run_2_ML_1_month_1_month_ML_temp.csv', encoding='latin1')
fe_temp_NTL = pd.read_csv('results/temperature_plot/CONFROOM_BOT_1_5A_Low_run_2_ML_1_month_1_month_fe_temp.csv', encoding='latin1')
wi_temp_NTL = pd.read_csv('results/temperature_plot/CONFROOM_BOT_1_5A_Low_run_2_ML_1_month_1_month_wi_temp.csv', encoding='latin1')


# real_temp['mean'] = real_temp.mean()
# real_temp_cp = real_temp.copy()

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 6))
plt.rc('legend',fontsize=14)
axs[0].plot(real_temp['MAE1'], '-', label='Real', c='palegreen',linewidth=2.5)#orangered
axs[0].plot(ml_temp['MAE1'], '--', label='ML', c='gold',linewidth=2.5)
axs[0].plot(fe_temp['MAE1'], '-', ms=6, markevery=(0, 0.04), label='FE', c='red',linewidth=2.5,alpha=0.9)
axs[0].plot(wi_temp['MAE1'], '--', ms=5, markevery=(0, 0.07), label='WI', c='dodgerblue',linewidth=2.5,alpha=0.7)
axs[0].set_title('Effective TL', size=20)
axs[0].set_ylabel('Indoor air temperature [°C]', size=17)
axs[0].set_xlabel('Time', size=17)
axs[0].set_ylim([17, 24.2])
axs[0].tick_params(axis='y', labelsize=17)
axs[0].tick_params(axis='x', labelsize=17)
axs[0].legend(loc='lower right')
axs[1].plot(real_temp_NTL['MAE1'], '-', label='Real', c='palegreen',linewidth=2.5)#orangered
axs[1].plot(ml_temp_NTL['MAE1'], '--', label='ML', c='gold',linewidth=2.5)
axs[1].plot(fe_temp_NTL['MAE1'], '-', ms=6, markevery=(0, 0.04), label='FE', c='red',linewidth=2.5,alpha=0.9)
axs[1].plot(wi_temp_NTL['MAE1'], '--', ms=5, markevery=(0, 0.07), label='WI', c='dodgerblue',linewidth=2.5,alpha=0.7)
axs[1].set_title('Negative TL', size=20)
axs[1].set_ylabel('Indoor air temperature [°C]', size=17)
# axs[1].set_ylabel(None)
axs[1].set_ylim([17, 24.2])
axs[1].set_xlabel('Time', size=17)
axs[1].tick_params(axis='y', labelsize=17)
axs[1].tick_params(axis='x', labelsize=17)
axs[1].legend(loc='lower right')
# plt.legend(loc='lower right')
plt.xlim(210, 400)
plt.tight_layout()
plt.savefig('fig_scripts/img/TL_vs_NTL_temperature_evolution.png')
plt.show()


# markers = ['.',',','o','v','^','<','>','1','2','3','4','8','s','p','P','*','h','H','+','x','X','D','d','|','_']
# descriptions = ['point', 'pixel', 'circle', 'triangle_down', 'triangle_up','triangle_left',
#                 'triangle_right', 'tri_down', 'tri_up', 'tri_left', 'tri_right', 'octagon',
#                 'square', 'pentagon', 'plus (filled)','star', 'hexagon1', 'hexagon2', 'plus',
#                 'x', 'x (filled)','diamond', 'thin_diamond', 'vline', 'hline']
#





# SWARM PLOT____________________________________________________________________________________________________________
#
# # Per week
# swarm_week = sns.swarmplot(x="Technique", y="MAE_avg", hue="Climate", data=df_1week, size=3.5)
# # ax = sns.violinplot(x="Testing", y="MAE1", data=df_test_metrics, inner=None)
# plt.title('Mean MAE distribution with a training period of 1 week', size=15)
# plt.show()
#
#
# # Per month
# swarm_month = sns.swarmplot(x="Technique", y="MAE_avg", hue="Climate", data=df_1month, size=3.5)
# # ax = sns.violinplot(x="Testing", y="MAE1", data=df_test_metrics, inner=None)
# plt.title('Mean MAE distribution with a training period of 1 month', size=15)
# plt.show()
#
#
# # Per month
# swarm_year = sns.swarmplot(x="Technique", y="MAE_avg", hue="Climate", data=df_1year, size=3.5)
# # ax = sns.violinplot(x="Testing", y="MAE1", data=df_test_metrics, inner=None)
# plt.title('Mean MAE distribution with a training period of 1 year', size=15)
# plt.show()


# ______________________________________________________________________________________________________________________
