
import pandas as pd
import os
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
import pydotplus
from pydotplus import graph_from_dot_data
import plotly.graph_objects as go
sns.set_theme(style="ticks")
plt.rcParams["font.family"] = "Times New Roman"

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
df_test_metrics = df_test_metrics[df_test_metrics['Testing'] != '1year']
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





# BOXPLOT_______________________________________________________________________________________________________________

# MAE_avg
sns.boxplot(x=df_test_metrics['Technique'], y = df_test_metrics['MAE_avg'])
plt.title('Mean MAE distribution for different techniques', size=15)
plt.grid()
plt.ylim([0,2])
# plt.savefig('results/img/Mean_MAE_distribution_for_different_techniques.png')
plt.show()

import numpy as np
# MAE_avg
plt.figure(figsize = (4,6))
my_pal = {"1week": "#019b9e", "1month": "#e1dfe0", "1month1year":"#d790c1"}
sns.boxplot(x= df_ML['Testing'], y =df_ML['MAE_avg'] , order=['1week','1month','1month1year'],palette=my_pal,width=0.6)
plt.title('ML performance', size=15)
plt.xticks(np.arange(3),('1 week','1 month','1 year'),fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Training',fontsize=15)
plt.ylabel('MAE [°C]',fontsize=15)
#plt.grid()
plt.ylim([0,1.25])
plt.tight_layout()
plt.savefig('fig_scripts/img/ML_different_period.png')
plt.show()



plt.figure(figsize = (6,6))
fig = sns.histplot(df_fe['MAE_avg'],  color="#154c79", label='Feature extraction',alpha=0.2,stat="percent",element='step')
fig = sns.histplot(df_wi['MAE_avg'], color="#7c0012", label='Weight initialization',alpha=0.4,stat="percent",element='step')
fig = sns.histplot(df_ML['MAE_avg'], color="#eab676", label='Machine learning',bins=90,alpha=0.2,stat="percent",element='step')
plt.axvline(statistics.median(df_fe['MAE_avg']), 0,25, color='#154c79',linestyle='--',linewidth=2)
plt.axvline(statistics.median(df_wi['MAE_avg']), 0,25, color='#7c0012',linestyle='--',linewidth=2)
plt.axvline(statistics.median(df_ML['MAE_avg']), 0,25, color='#eab676',linestyle='--',linewidth=2)
plt.xlim([0,1.25])
plt.legend()
plt.xlabel('Average MAE [°C]',fontsize=14)
plt.ylabel('Percent',fontsize=14)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title('Average MAE distribution', size=15)
#plt.grid()
plt.savefig('fig_scripts/img/Mean_MAE_density_distribution_for_technique.png')
plt.show()



df_1week_ML = df_ML.loc[df_test_metrics['Training'] == '1week']
df_1week_fe = df_fe.loc[df_test_metrics['Testing'] == '1week']
df_1week_wi = df_wi.loc[df_test_metrics['Testing'] == '1week']
df_1week = pd.concat([df_1week_ML, df_1week_fe, df_1week_wi], axis=0)


df_1month_ML = df_ML.loc[df_test_metrics['Training'] == '1month']
df_1month_fe = df_fe.loc[df_test_metrics['Testing'] == '1month']
df_1month_wi = df_wi.loc[df_test_metrics['Testing'] == '1month']
df_1month = pd.concat([df_1month_ML, df_1month_fe, df_1month_wi], axis=0)


df_1year_ML = df_ML.loc[df_test_metrics['Testing'] == '1month1year']
df_1year_fe = df_fe.loc[df_test_metrics['Testing'] == '1month1year']
df_1year_wi = df_wi.loc[df_test_metrics['Testing'] == '1month1year']
df_1year = pd.concat([df_1year_ML, df_1year_fe, df_1year_wi], axis=0)



plt.figure(figsize = (6,6))
fig = sns.histplot(df_1month_fe['MAE_avg'],  color="#154c79", label='Feature extraction',alpha=0.2,stat="percent",element='step')
fig = sns.histplot(df_1month_wi['MAE_avg'], color="#7c0012", label='Weight initialization',alpha=0.2,stat="percent",element='step')
fig = sns.histplot(df_1month_ML['MAE_avg'], color="#eab676", label='Machine learning',bins=10,alpha=0.3,stat="percent",element='step')
plt.axvline(statistics.median(df_1month_fe['MAE_avg']), 0,25, color='#154c79',linestyle='--',linewidth=2)
plt.axvline(statistics.median(df_1month_wi['MAE_avg']), 0,25, color='#7c0012',linestyle='--',linewidth=2)
plt.axvline(statistics.median(df_1month_ML['MAE_avg']), 0,25, color='#eab676',linestyle='--',linewidth=2)
#plt.xlim([0,1.25])
plt.legend()
plt.xlabel('Average MAE [°C]',fontsize=14)
plt.title('One month training period', size=15)
plt.ylabel('Percent',fontsize=14)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
#plt.grid()
plt.savefig('fig_scripts/img/Mean_MAE_1_month.png')
plt.show()










# Import Data
df_parallel = df_test_metrics.iloc[:,0:4]
df_parallel = pd.concat([df_parallel,df_test_metrics.iloc[:,5],df_test_metrics.iloc[:,12]],axis=1)
df_parallel['MAE_avg'] = pd.cut(df_parallel['MAE_avg'], bins=[0, 0.3, 0.7, 6], include_lowest=True, labels=['high_performance', 'mid_performance', 'low_performance'])
df_parallel['Testing'] = df_parallel['Testing'].replace({'1week': '1 week' ,'1month': '1 month' ,'1month1year': '1 year'})
df_parallel['Technique'] = df_parallel['Technique'].replace({'fe': 'Feature extraction' ,'wi': 'Weight initialization'})

# Create dimensions
Climate_dim=go.parcats.Dimension(values=df_parallel['Climate'],label="Climate")
Eff_dim=go.parcats.Dimension(values=df_parallel['Efficiency'], label="Efficiency")
Occ_dim=go.parcats.Dimension(values=df_parallel['Occupancy'], label="Occupancy")
Tech_dim=go.parcats.Dimension(values=df_parallel['Technique'],label="Technique", ticktext = ['Parameter-based','Feature-based','Instance-based','Relation-based'])
Test_dim = go.parcats.Dimension(values=df_parallel['Testing'],label="Training")
MAE_dim = go.parcats.Dimension(values=df_parallel['MAE_avg'],label="MAE")
# Create parcats trace

df_parallel['MAE_avg'] = df_parallel['MAE_avg'].replace({'high_performance': 1 ,'mid_performance': 2 ,'low_performance': 3})
color = df_parallel['MAE_avg']
fig = go.Figure(data = [go.Parcats(dimensions=[Tech_dim, Test_dim, Climate_dim, Eff_dim,Occ_dim],
        line={'color':color,'colorscale': 'RdYlGn_r','shape': 'hspline'},
        hoveron='color', hoverinfo='count+probability',
        labelfont={'size': 18, 'family': 'Times'},
        tickfont={'size': 16, 'family': 'Times'},
        arrangement='freeform')])
fig.write_html('first_figure.html', auto_open=True)
plt.tight_layout()
#plt.savefig("prova.png", bbox_inches='tight', pad_inches=0,dpi=500)
fig.show()

df_average_metrics = df_1month
df_average_metrics = df_average_metrics.groupby('Technique').mean()

df1 = df_average_metrics.iloc[0,:]
df2 = df_average_metrics.iloc[1,:]
df3 = df_average_metrics.iloc[2,:]

df_MAE = df_average_metrics.iloc[:,:7]
df_MSE = df_average_metrics.iloc[:,7:14]
df_MAPE = df_average_metrics.iloc[:,14:]




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

    df_ML_mape = pd.DataFrame(df_ML[14:-1])
    df_FE_mape = pd.DataFrame(df_FE[14:-1])
    df_WI_mape = pd.DataFrame(df_WI[14:-1])

    # Concatening of metrics and step names
    df_def_mae = pd.concat([df_ML_mae, df_FE_mae, df_WI_mae], axis=1)
    df_def_mae.columns = ['ML', 'FE', 'WI']
    df_def_mae['steps'] = ['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6']

    # Concatening of metrics and step names
    df_def_mse = pd.concat([df_ML_mse, df_FE_mse, df_WI_mse], axis=1)
    df_def_mse.columns = ['ML', 'FE', 'WI']
    df_def_mse['steps'] = ['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6']

    # Concatening of metrics and step names
    df_def_mape = pd.concat([df_ML_mape, df_FE_mape, df_WI_mape], axis=1)
    df_def_mape.columns = ['ML', 'FE', 'WI']
    df_def_mape['steps'] = ['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6']

    # items = file1.split('_')[:7]

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(15, 7))
    # plt.yticks(np.arange(0,0.50,0.05))
    df_def_mae.plot(x="steps", y=['ML', 'FE', 'WI'], kind="bar", color=["#eab676", '#154c79', "#7c0012"], edgecolor='black', ax=axs[0])
    # axs[0].set_ylabel('MAE [°C]', size=20)
    # axs[0].set_xlabel('Prediction steps', size=20)
    axs[0].grid(axis='y')
    axs[0].tick_params('x', labelrotation=90)
    axs[0].set_xticklabels(labels=['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6'], size=18)
    axs[0].set_ylim([0, 0.45])
    axs[0].tick_params(axis='y', labelsize=18)
    axs[0].set_xlabel(None)
    axs[0].set_title('MAE', size=20)
    df_def_mse.plot(x="steps", y=['ML', 'FE', 'WI'], kind="bar", color=["#eab676", '#154c79', "#7c0012"], edgecolor='black', ax=axs[1], legend=False)
    # axs[1].set_ylabel('MSE', size=20)
    axs[1].set_xlabel(None)
    axs[1].grid(axis='y')
    axs[1].tick_params('x', labelrotation=90)
    axs[1].set_xticklabels(labels=['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6'], size=18)
    #axs[1].set_ylim([0, 0.45])
    axs[1].tick_params(axis='y', labelsize=18)
    axs[1].set_title('MSE', size=20)
    #plt.suptitle('{}: MAE and MSE comparison between ML / TL-FE / TL-WI'.format(' '.join(file1.split('_')[:3])), size=20)
    df_def_mape.plot(x="steps", y=['ML', 'FE', 'WI'], kind="bar",color=["#eab676", '#154c79', "#7c0012"], edgecolor='black',
                    ax=axs[2],legend=False)
    # axs[0].set_ylabel('MAE [°C]', size=20)
    # axs[0].set_xlabel('Prediction steps', size=20)
    axs[2].grid(axis='y')
    axs[2].tick_params('x', labelrotation=90)
    axs[2].set_xticklabels(labels=['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6'], size=18)
    #axs[2].set_ylim([0, 0.45])
    axs[2].tick_params(axis='y', labelsize=18)
    axs[2].set_xlabel(None)
    axs[2].set_title('MAPE', size=20)

    plt.tight_layout()
    plt.savefig('fig_scripts/img/average_metrics.png')
    plt.show()

    return df_ML, df_FE, df_WI

df_ML, df_FE, df_WI = comparing_tech(df1, df2, df3)
# navajowhite,lightsteelblue, greenyellow