import pandas as pd
import os
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
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
df4['Zone'] = df3[1]+df3[2]
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


'''
a = pd.Series()
for index, row in df_test_metrics.iterrows():
    if row['Technique'] == 'ML':
        c = 1
    else:
        MAE_tl = row['MAE_avg']
        Climate = row['Climate']
        Efficiency = row['Efficiency']
        Occupancy = row['Occupancy']
        Testing = row['Testing']
        ML_row = df_test_metrics.loc[(df_test_metrics['Climate'] == Climate) & (df_test_metrics['Efficiency'] == Efficiency) & (df_test_metrics['Occupancy'] == Occupancy) & (
                    df_test_metrics['Testing'] == Testing) & (df_test_metrics['Technique'] == 'ML')]
        MAE_ml = ML_row['MAE_avg']
        asy = (MAE_ml-MAE_tl)
        a = a.append(pd.Series(asy))


#analsys without 1 year data availability
b = pd.Series()
for index, row in df_test_metrics.iterrows():
    if row['Technique'] == 'ML':
        c = 1
    elif (row['Technique'] == 'fe') & (row['Testing'] == '1year'):
        c = 1
    elif (row['Technique'] == 'wi') & (row['Testing'] == '1year'):
        c = 1
    else:
        MAE_tl = row['MAE_avg']
        Climate = row['Climate']
        Efficiency = row['Efficiency']
        Occupancy = row['Occupancy']
        Testing = row['Testing']
        ML_row = df_test_metrics.loc[(df_test_metrics['Climate'] == Climate) & (df_test_metrics['Efficiency'] == Efficiency) & (df_test_metrics['Occupancy'] == Occupancy) & (
                    df_test_metrics['Testing'] == Testing) & (df_test_metrics['Technique'] == 'ML')]
        MAE_ml = ML_row['MAE_avg']
        asy = (MAE_ml-MAE_tl)
        b = b.append(pd.Series(asy))

sns.distplot(a)
plt.show()
sns.histplot(b)
plt.show()
'''

#https://jamesrledoux.com/code/group-by-aggregate-pandas

b = pd.Series()
for index, row in df_test_metrics.iterrows():
    if row['Technique'] == 'ML':
        c = 1
        df_test_metrics.loc[index, 'asy'] = 0
    else:
        MAE_tl = row['MAE_avg']
        Climate = row['Climate']
        Efficiency = row['Efficiency']
        Occupancy = row['Occupancy']
        Testing = row['Testing']
        Technique = row['Technique']
        ML_row = df_test_metrics.loc[(df_test_metrics['Climate'] == Climate) & (df_test_metrics['Efficiency'] == Efficiency) & (df_test_metrics['Occupancy'] == Occupancy) & (
                    df_test_metrics['Testing'] == Testing) & (df_test_metrics['Technique'] == 'ML')]
        MAE_ml = ML_row['MAE_avg']
        asy = (MAE_ml-MAE_tl)
        b = b.append(pd.Series(asy))
        df_test_metrics.loc[index, 'asy'] = asy.values




# Split dataset based on technique
df_ML = df_test_metrics.loc[df_test_metrics['Technique'] == 'ML']
df_fe = df_test_metrics.loc[df_test_metrics['Technique'] == 'fe']
df_wi = df_test_metrics.loc[df_test_metrics['Technique'] == 'wi']

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


g = sns.catplot(x="Testing", y="MAE_avg",
                hue="Technique", col="Climate",
                data=df_test_metrics, kind="box", order=['1week','1month','1year'],
                height=4, aspect=.7,)
# Put the legend out of the figure
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylim([0,1.5])
plt.tight_layout()
plt.show()


df1week = df_1week_fe
df1week = pd.concat([df1week,df_1week_wi])
df1month = df_1month_fe
df1month = pd.concat([df1month,df_1month_wi])
df1year = df_1year_fe
df1year = pd.concat([df1year,df_1year_wi])

df_violin_plot = pd.concat([df_1week,df1month,df1year])
df_violin_plot = df_violin_plot[df_violin_plot.Technique != 'ML']



df_eff_standard_cl_3c = df_test_metrics.loc[(df_test_metrics['Efficiency'] == 'Standard') & (df_test_metrics['Climate'] == '3C')]
df_eff_standard_cl_3c = df_eff_standard_cl_3c.loc[df_eff_standard_cl_3c['Occupancy'] != 'run1']

order=['1week','1month','1month1year']

df_eff_standard_run1 = df_test_metrics.loc[(df_test_metrics['Efficiency'] == 'Standard') & (df_test_metrics['Occupancy'] == 'run1')]
df_eff_standard_run1 = df_eff_standard_run1.loc[df_eff_standard_run1['Climate'] != '3C']


df_cl3C_run1 = df_test_metrics.loc[(df_test_metrics['Climate'] == '3C') & (df_test_metrics['Occupancy'] == 'run1')]
df_cl3C_run1 = df_cl3C_run1.loc[df_cl3C_run1['Efficiency'] != 'Standard']

sns.scatterplot(x='Occupancy',y='MAE_avg', data=df_eff_standard_cl_3c)
plt.show()



g = sns.FacetGrid(df_test_metrics, col="Climate",  row="Efficiency",hue="Technique")
g.map(sns.scatterplot, "Testing", "MAE_avg", alpha=0.5)
plt.show()



g = sns.FacetGrid(df_eff_standard_cl_3c, col="Occupancy", hue="Technique")
g.map(sns.scatterplot, "Testing", "MAE_avg", alpha=0.7, )
plt.ylim(0,1)
g.refline(y=0.5)
plt.show()


g = sns.FacetGrid(df_eff_standard_run1, col="Climate", hue="Technique")
g.map(sns.scatterplot, "Testing", "MAE_avg", alpha=0.7, )
plt.ylim(0,1)
g.refline(y=0.5)
plt.show()


g = sns.FacetGrid(df_cl3C_run1, col="Efficiency", hue="Technique")
g.map_dataframe(sns.scatterplot, "Testing", "MAE_avg", alpha=0.7, )
g.refline(y=0.5)
plt.show()


df_prova = df_eff_standard_cl_3c
df_prova = df_prova.append(df_eff_standard_run1)
df_prova = df_prova.append(df_cl3C_run1)
df_prova.Testing = pd.Categorical(df_prova.Testing,categories=['1week','1month','1month1year'])

df_prova = df_prova.pivot_table(
    values='MAE_avg',
    index=['Occupancy','Efficiency','Climate','Testing'],
    columns='Technique'
    )
df_prova.reset_index(inplace=True)



df_prova = df_prova.reset_index()


order = ['1_week','1_month','1_year']

df_prova['Testing'] = df_prova['Testing'].replace({'1week': '1_week' ,'1month': '1_month' ,'1month1year': '1_year'})

df_prova['shortname'] = df_prova['Efficiency']+str('_')+df_prova['Climate']+str('_')+df_prova['Occupancy']+str('_')+df_prova['Testing']
plt.figure(figsize = (7,7))
sns.scatterplot(y="shortname",x="ML",data = df_prova,color='#eab676',alpha=1,edgecolor='black')
sns.scatterplot(y="shortname",x="fe",data = df_prova,color='#154c79',alpha=0.7,edgecolor='black')
sns.scatterplot(y="shortname",x="wi",data = df_prova,color='#7c0012',alpha=0.7,edgecolor='black')
plt.hlines(y=df_prova.index, xmin=df_prova.wi, xmax=df_prova.ML, color='k', alpha=0.4)
plt.xlim(0,1.1)
plt.xlabel('MAE [°C]')
plt.ylabel('Model')
plt.tight_layout()
plt.savefig('fig_scripts/img/sensitivity.png')
plt.show()

