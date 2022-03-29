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
#df_test_metrics = df_test_metrics.drop([67])
df_test_metrics = df_test_metrics[df_test_metrics['Testing'] != '1year']
# Delete the nan
df_test_metrics.MAE_avg.fillna(df_test_metrics.MAE, inplace=True)
df_test_metrics.MAPE_avg.fillna(df_test_metrics.MAPE, inplace=True)
df_test_metrics.MSE_avg.fillna(df_test_metrics.MSE, inplace=True)
del df_test_metrics['MAE']
del df_test_metrics['MAPE']
del df_test_metrics['MSE']


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



####################### Separate plot
'''
ax1 = sns.boxplot(x="Climate", y="MAE_avg", hue="Technique", data=df_1week)
# ax = sns.violinplot(x="Testing", y="MAE1", data=df_test_metrics, inner=None)
plt.grid()
plt.title('Asymptotic performance distribution per technique and climate', size=15)
# plt.savefig('results/img/Mean_MAE_distribution_per_technique_and_testing_period.png')
plt.show()


ax1 = sns.boxplot(x="Climate", y="MAE_avg", hue="Technique", data=df_1month)
# ax = sns.violinplot(x="Testing", y="MAE1", data=df_test_metrics, inner=None)
plt.title('Asymptotic performance distribution per technique and climate', size=15)
plt.grid()
# plt.savefig('results/img/Mean_MAE_distribution_per_technique_and_testing_period.png')
plt.show()


ax1 = sns.boxplot(x="Climate", y="MAE_avg", hue="Technique", data=df_1year)
# ax = sns.violinplot(x="Testing", y="MAE1", data=df_test_metrics, inner=None)
plt.grid()
plt.title('Asymptotic performance distribution per technique and climate', size=15)
# plt.savefig('results/img/Mean_MAE_distribution_per_technique_and_testing_period.png')
plt.show()

'''
import numpy as np
my_pal = {"ML": "#eab676", "fe": "#154c79", "wi":"#7c0012"}
g = sns.catplot(x="Testing", y="MAE_avg",
                hue="Technique", col="Climate",
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
plt.ylim([0,1.5])
g.refline(y=0.5)
#plt.tight_layout()
plt.savefig('fig_scripts/img/Mean_MAE_distribution_per_technique_and_testing_period.png')
plt.show()




##################### ASYMPTOTIC PERFORMANCE NEW DATASET WITH ONLY TL
df1week = df_1week_fe
df1week = pd.concat([df1week,df_1week_wi])
df1month = df_1month_fe
df1month = pd.concat([df1month,df_1month_wi])
df1year = df_1year_fe
df1year = pd.concat([df1year,df_1year_wi])

df_violin_plot = pd.concat([df_1week,df1month,df1year])
df_violin_plot = df_violin_plot[df_violin_plot.Technique != 'ML']


my_pal2 = { "fe": "#154c79", "wi":"#7c0012"}
g = sns.catplot(x="Testing", y="asy",
                hue="Technique", col="Climate",
                data=df_violin_plot, kind="box",
                height=4, aspect=.6,legend_out= True,palette=my_pal2,order=['1week','1month','1month1year'])
# Put the legend out of the figure
plt.xticks(np.arange(3),('1 week','1 month','1 year'))
#plt.legend(labels=['Machine learning','Feature extraction','Weight initialization'],loc='upper right', bbox_to_anchor=(0.5, 0.5))
new_labels = ['Feature extraction','Weight initialization']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)
sns.move_legend(g, "upper left",ncol=1, bbox_to_anchor=(.12, 0.9))
g.set_axis_labels("Training", "Asymptotic performance [°C]")
plt.ylim([-0.5,1])
g.refline(y=0)
plt.tight_layout()
plt.savefig('fig_scripts/img/asymptotic_performance.png')
plt.show()




'''ax1 = sns.violinplot(x="Climate", y="asy", hue="Technique", data=df_negative_transfer)
# ax = sns.violinplot(x="Testing", y="MAE1", data=df_test_metrics, inner=None)
plt.grid()
plt.title('Asymptotic performance distribution per technique and climate', size=15)
# plt.savefig('results/img/Mean_MAE_distribution_per_technique_and_testing_period.png')
#plt.ylim([-0.7,2.6])
plt.show()

ax1 = sns.violinplot(x="Occupancy", y="asy", hue="Technique", data=df_negative_transfer)
# ax = sns.violinplot(x="Testing", y="MAE1", data=df_test_metrics, inner=None)
plt.grid()
plt.title('Asymptotic performance distribution per technique and climate', size=15)
# plt.savefig('results/img/Mean_MAE_distribution_per_technique_and_testing_period.png')
#plt.ylim([-0.7,2.6])
plt.show()

ax1 = sns.violinplot(x="Efficiency", y="asy", hue="Technique", data=df_negative_transfer)
# ax = sns.violinplot(x="Testing", y="MAE1", data=df_test_metrics, inner=None)
plt.grid()
plt.title('Asymptotic performance distribution per technique and climate', size=15)
# plt.savefig('results/img/Mean_MAE_distribution_per_technique_and_testing_period.png')
#plt.ylim([-0.7,2.6])
plt.show()
'''





# Prepare Data
df_negative_transfer = df_test_metrics.loc[df_test_metrics['asy'] < -0.05]
df = df_negative_transfer
x = df.loc[:, ['asy']]
df['colors'] = ['red' if x < 0 else 'darkgreen' for x in df['asy']]
df.sort_values('asy', inplace=True)
df.reset_index(inplace=True)


df['shortname'] = df['Climate']+df['Efficiency']+df['Occupancy']+df['Technique']+df['Testing']
df['shortname'] = df['Efficiency']+'_'+df['Occupancy']+'_'+df['Technique']
df = df.drop(columns=['Training'])
df = df.rename(columns={'Testing':'Training'})
df['Training'] = df['Training'].replace({'1week': '1 week' ,'1month': '1 month' ,'1month1year': '1 year'})
# Draw plot
# Draw plot

markers = {"1A": "o", "5A": "X",'3C':'D'}
my_pal3 = {"1 week": "#019b9e", "1 month": "#e1dfe0", "1 year":"#d790c1"}
plt.figure(figsize=(6,6), dpi= 300)
sns.scatterplot(df.asy, df.index, s=100, alpha=.9, hue=df.Training,style=df.Climate,markers=markers,edgecolors='face',palette=my_pal3, hue_order=['1 week','1 month','1 year'],style_order=['1A','3C','5A'])
# for x, y, tex in zip(df.asy, df.index, df.asy):
#     t = plt.text(x, y, round(tex, 1), horizontalalignment='center',
#                  verticalalignment='center', fontdict={'color':'black'})
plt.yticks(df.index, df.shortname)
plt.title('Negative transfer')
plt.xlabel('Asymptotic Performance [°C]')
plt.grid(linestyle='--')
plt.xlim(-0.35, -0.04)
plt.tight_layout()
plt.savefig('fig_scripts/img/negative_transfer.png')
plt.show()



# Prepare Data
df = df_test_metrics
x = df.loc[:, ['asy']]
df['colors'] = ['red' if x < -0.05  else 'blue' for x in df['asy']]
df.sort_values('asy', inplace=True)
df['shortname'] = df['Climate']+df['Efficiency']+df['Occupancy']+df['Technique']+df['Testing']
df = df.sort_values(['asy'])
df = df[df.asy != 0]
df = df.reset_index(drop=True)
# Draw plot
df['Transfer category'] = pd.cut(df['asy'], bins=[-0.5, -0.05, 0.1, 6], include_lowest=True, labels=['Negative', 'Neutral', 'Effective'])
df['Percentage'] = df.index*100/156

my_pal4 = {"Negative": "#B81D13", "Neutral": "#EFB700", "Effective":"#008450"}
plt.figure(figsize=(6,6), dpi= 300)
sns.scatterplot(df.asy, df.Percentage, s=25, alpha=.6, hue=df['Transfer category'], palette=my_pal4)
plt.title('Transfer learning effectiveness')
plt.xlim(-0.4,1)
plt.ylim(-5,100)
plt.grid(linestyle='--')
plt.xlabel('Asymptotic Performance [°C]')
plt.savefig('fig_scripts/img/transfer_effectiveness.png')
plt.show()








#https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/#15.-Ordered-Bar-Chart
df_negative_parallel = df_test_metrics.iloc[:,1:5]
df_negative_parallel = pd.concat([df_negative_parallel,df_test_metrics.iloc[:,6],df_test_metrics.iloc[:,-3]],axis=1)
df_negative_parallel = df_negative_parallel.loc[(df_negative_parallel['Technique'] == 'fe') | (df_negative_parallel['Technique'] == 'wi')]
df_negative_parallel['asy'] = pd.cut(df_negative_parallel['asy'], bins=[-0.5, -0.05, 0.1, 6], include_lowest=True, labels=['negative', 'neutral', 'effective'])
df_negative_parallel['Testing'] = df_negative_parallel['Testing'].replace({'1week': '1 week' ,'1month': '1 month' ,'1month1year': '1 year'})
df_negative_parallel['Technique'] = df_negative_parallel['Technique'].replace({'fe': 'Feature extraction' ,'wi': 'Weight initialization'})


# Create dimensions

Climate_dim=go.parcats.Dimension(values=df_negative_parallel['Climate'],label="Climate")
Eff_dim=go.parcats.Dimension(values=df_negative_parallel['Efficiency'], label="Efficiency")
Occ_dim=go.parcats.Dimension(values=df_negative_parallel['Occupancy'], label="Occupation")
Tech_dim=go.parcats.Dimension(values=df_negative_parallel['Technique'],label="Technique", ticktext = ['Feature extraction','Weight initialization'])
Test_dim = go.parcats.Dimension(values=df_negative_parallel['Testing'],label="Training")
asy_dim = go.parcats.Dimension(values=df_negative_parallel['asy'],label="MAE")
# Create parcats trace

df_negative_parallel['asy'] = df_negative_parallel['asy'].replace({'negative': 1 ,'neutral': 2 ,'effective': 3})
color = df_negative_parallel['asy']
fig = go.Figure(data = [go.Parcats(dimensions=[Tech_dim, Test_dim, Climate_dim, Eff_dim,Occ_dim],
        line={'color':color,'colorscale': 'RdYlGn','shape': 'hspline'},
        hoveron='color', hoverinfo='count+probability',
        labelfont={'size': 18, 'family': 'Times'},
        tickfont={'size': 16, 'family': 'Times'},
        arrangement='freeform')])
plt.tight_layout()
fig.write_html('first_figure.html', auto_open=True)

#plt.savefig("prova.png", bbox_inches='tight', pad_inches=0,dpi=500)
fig.show()



