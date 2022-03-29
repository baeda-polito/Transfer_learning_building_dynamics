import pandas as pd
import os
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_theme(style="ticks")
plt.rcParams["font.family"] = "Times New Roman"

PATH = "results"
EXT = "*train_metrics.csv"

#https://perials.com/getting-csv-files-directory-subdirectories-using-python/
#https://www.geeksforgeeks.org/python-pandas-split-strings-into-two-list-columns-using-str-split/

all_csv_files = []
df = pd.DataFrame()
for path, subdir, files in os.walk(PATH):
    for file in glob(os.path.join(path, EXT)):
        all_csv_files.append(file)
        file_df = pd.read_csv(file, header=0)
        file_df['file_name'] = file
        file_df = file_df.reset_index()
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

df_train_metrics = pd.concat([df4, df], axis=1)

df_train_metrics = df_train_metrics.reset_index(drop=True)
df_train_metrics.MAE_avg.fillna(df_train_metrics.MAE, inplace=True)
df_train_metrics.MAPE_avg.fillna(df_train_metrics.MAPE, inplace=True)
df_train_metrics.MSE_avg.fillna(df_train_metrics.MSE, inplace=True)
del df_train_metrics['MAE']
del df_train_metrics['MAPE']
del df_train_metrics['MSE']

df_train_metrics = df_train_metrics.reset_index()

diffs = np.append(np.diff(df_train_metrics['index']), 50)
discont_indices = np.abs(diffs) > np.pi
df_test_metrics2 = df_train_metrics
df_test_metrics2[discont_indices] = np.nan
# sns.lineplot(df_test_metrics2['index'],df_test_metrics2.MAE_avg,hue=df_test_metrics2.Technique)
# plt.show()
df_test_metrics2['Testing'] = df_test_metrics2['Testing'].replace({'1week': '1 week' ,'1month': '1 month' ,'1year': '1 year'})
df_test_metrics2 = df_test_metrics2.rename(columns={'index':'Episodes'})

d = {'color': ['#eab676', '#154c79','#7c0012'], "ls" : ["-","-","--"]}
g = sns.FacetGrid(df_test_metrics2, col="Testing", col_order=['1 week','1 month','1 year'], hue='Technique', hue_kws=d,height=3.5, col_wrap=3)
g.map(sns.lineplot, "index",'MAE_avg')
g.set_axis_labels("Episodes", "MAE [°C]")
#plt.ylabel('MAE [°C]')
plt.legend(['ML','FE','WI'])
plt.ylim([0,8])
plt.xlim([0,90])
plt.tight_layout()
plt.savefig('fig_scripts/img/jumpstart')
plt.show()

g = sns.FacetGrid(df_test_metrics2, col="Testing", hue='Technique',height=3.5, col_wrap=3)
g.map(sns.lineplot, "index",'MAE_avg')
plt.xlabel('Episodes')
plt.ylim(0.2,0.5)
plt.xlim([70,80])

plt.show()



df_jumpstart = df_test_metrics2.loc[df_test_metrics2['index'] == 0]


a = pd.Series()
for index, row in df_jumpstart.iterrows():
    if row['Technique'] == 'ML':
        df_jumpstart.loc[index, 'jumpstart'] = 0
    else:
        MAE_tl = row['MAE_avg']
        Climate = row['Climate']
        Efficiency = row['Efficiency']
        Occupancy = row['Occupancy']
        Testing = row['Testing']
        ML_row = df_jumpstart.loc[(df_jumpstart['Climate'] == Climate) & (df_jumpstart['Efficiency'] == Efficiency) & (df_jumpstart['Occupancy'] == Occupancy) & (
                    df_jumpstart['Testing'] == Testing) & (df_jumpstart['Technique'] == 'ML')]
        MAE_ml = ML_row['MAE_avg']
        jumpstart = (MAE_ml-MAE_tl)
        a = a.append(pd.Series(jumpstart))
        df_jumpstart.loc[index, 'jumpstart'] = jumpstart.values


# Split dataset based on technique
df_ML = df_jumpstart.loc[df_jumpstart['Technique'] == 'ML']
df_fe = df_jumpstart.loc[df_jumpstart['Technique'] == 'fe']
df_wi = df_jumpstart.loc[df_jumpstart['Technique'] == 'wi']

df_1week_ML = df_ML.loc[df_jumpstart['Training'] == '1week']
df_1week_fe = df_fe.loc[df_jumpstart['Testing'] == '1week']
df_1week_wi = df_wi.loc[df_jumpstart['Testing'] == '1week']
df_1week = pd.concat([df_1week_ML, df_1week_fe, df_1week_wi], axis=0)


df_1month_ML = df_ML.loc[df_jumpstart['Training'] == '1month']
df_1month_fe = df_fe.loc[df_jumpstart['Testing'] == '1month']
df_1month_wi = df_wi.loc[df_jumpstart['Testing'] == '1month']
df_1month = pd.concat([df_1month_ML, df_1month_fe, df_1month_wi], axis=0)


df_1year_ML = df_ML.loc[df_jumpstart['Training'] == '1year']
df_1year_fe = df_fe.loc[df_jumpstart['Testing'] == '1year']
df_1year_wi = df_wi.loc[df_jumpstart['Testing'] == '1year']
df_1year = pd.concat([df_1year_ML, df_1year_fe, df_1year_wi], axis=0)


sns.distplot(df_1year_ML.MAE_avg,kde=True,hist=False)
sns.distplot(df_1month_ML.MAE_avg,kde=True,hist=False)
sns.distplot(df_1week_ML.MAE_avg,kde=True,hist=False)
plt.legend(['1 year','1 month','1 week'])
#plt.xlim([0,3])
plt.show()

df1week = df_1week_fe
df1week = pd.concat([df1week,df_1week_wi])
ax1 = sns.boxplot(x="Climate", y="jumpstart", hue="Technique", data=df1week)
# ax = sns.violinplot(x="Testing", y="MAE1", data=df_test_metrics, inner=None)
plt.grid()
plt.title('Jumpstart 1 week', size=15)
# plt.savefig('results/img/Mean_MAE_distribution_per_technique_and_testing_period.png')
plt.show()

df1month = df_1month_fe
df1month = pd.concat([df1month,df_1month_wi])
ax1 = sns.boxplot(x="Climate", y="jumpstart", hue="Technique", data=df1month)
# ax = sns.violinplot(x="Testing", y="MAE1", data=df_test_metrics, inner=None)
plt.title('Jumpstart 1 month', size=15)
plt.grid()
# plt.savefig('results/img/Mean_MAE_distribution_per_technique_and_testing_period.png')
plt.show()

df1year = df_1year_fe
df1year = pd.concat([df1year,df_1year_wi])
ax1 = sns.boxplot(x="Climate", y="jumpstart", hue="Technique", data=df1year)
# ax = sns.violinplot(x="Testing", y="MAE1", data=df_test_metrics, inner=None)
plt.grid()
plt.title('Jumpstart 1 year', size=15)
# plt.savefig('results/img/Mean_MAE_distribution_per_technique_and_testing_period.png')
plt.show()

df_violin_plot = pd.concat([df_1week,df1month,df1year])
df_violin_plot = df_violin_plot[df_violin_plot.Technique != 'ML']
g = sns.catplot(x="Climate", y="jumpstart",
                hue="Technique", col="Testing",
                data=df_violin_plot, kind="box",
                height=4, aspect=.7)
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
plt.show()