import pandas as pd
import os
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
sns.set_theme(style="ticks")
plt.rcParams["font.family"] = "Times New Roman"
PATH = "other_results"
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
df4['Zone'] = df3[1]+df3[2]
df4['Climate'] = df3[3]
df4['Efficiency'] = df3[4]
df4['Occupancy'] = df3[5]+df3[6]
df4['Technique'] = df3[7]
df4['Training'] = df3[8]+df3[9]
df4['Testing'] = df3[10]+df3[11]

df_test_metrics = pd.concat([df4, df], axis=1)

df_test_metrics = df_test_metrics.reset_index(drop=True)
#df_test_metrics = df_test_metrics.drop([67])
df_test_metrics = df_test_metrics[df_test_metrics['Testing'] != '1year']
# Delete the nan
df_test_metrics.MAE_avg.fillna(df_test_metrics.MAE, inplace=True)
df_test_metrics.MAPE_avg.fillna(df_test_metrics.MAPE, inplace=True)
df_test_metrics.MSE_avg.fillna(df_test_metrics.MSE, inplace=True)
del df_test_metrics['MAE']
del df_test_metrics['MAPE']
del df_test_metrics['MSE']

df_test_metrics.drop(df_test_metrics.index[df_test_metrics['Zone'] == 'BOT1'], inplace=True)




import numpy as np
my_pal = {"ML": "#eab676", "fe": "#154c79", "wi":"#7c0012"}
g = sns.catplot(x="Testing", y="MAE_avg",
                hue="Technique", col="Zone",
                data=df_test_metrics, kind="box",
                height=5, aspect=.7,legend_out= True,palette=my_pal,order=['1week','1month','1month1year'],)
# Put the legend out of the figure
plt.xticks(np.arange(3),('1 week','1 month','1 year'))
#plt.legend(labels=['Machine learning','Feature extraction','Weight initialization'],loc='upper right', bbox_to_anchor=(0.5, 0.5))
new_labels = ['Machine learning','Feature extraction','Weight initialization']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)
sns.move_legend(g, "upper left",ncol=1, bbox_to_anchor=(.12, 0.9))
g.set_ylabels('MAE [°C]')
g.set_axis_labels("Training", "MAE [°C]",fontsize=15)
#plt.ylim([0,1.5])
g.refline(y=0.5)
#plt.tight_layout()
#plt.savefig('fig_scripts/img/Mean_MAE_distribution_per_technique_and_testing_period.png')
plt.show()



df_prova = df_test_metrics.pivot_table(
    values='MAE_avg',
    index=['Zone','Testing'],
    columns='Technique'
    )
df_prova.reset_index(inplace=True)



df_prova = df_prova.reset_index()


order = ['1_week','1_month','1_year']

df_prova['Testing'] = df_prova['Testing'].replace({'1week': '1_week' ,'1month': '1_month' ,'1month1year': '1_year'})

df_prova['shortname'] = df_prova['Zone']+str('_')+df_prova['Testing']
plt.figure(figsize = (7,7))
sns.scatterplot(y="shortname",x="ML",data = df_prova,color='#eab676',alpha=1,edgecolor='black')
sns.scatterplot(y="shortname",x="fe",data = df_prova,color='#154c79',alpha=0.7,edgecolor='black')
sns.scatterplot(y="shortname",x="wi",data = df_prova,color='#7c0012',alpha=0.7,edgecolor='black')
plt.hlines(y=df_prova.index, xmin=df_prova.wi, xmax=df_prova.ML, color='k', alpha=0.4)
plt.xlim(0,1.1)
plt.xlabel('MAE [°C]')
plt.ylabel('Model')
plt.tight_layout()
#plt.savefig('fig_scripts/img/sensitivity.png')
plt.show()