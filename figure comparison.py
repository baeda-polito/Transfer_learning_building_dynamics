import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler
import pandas as pd
import numpy as np
from utils import import_file, min_max_T, normalization, split_multistep_sequences, mean_absolute_percentage_error, create_data, define_period
from utils_deploy import define_period_deploy
from models import LSTM
from training_testing_functions import train_model, test_model
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

df = pd.read_excel("C:\\Users\\BAEDA\\PycharmProjects\\Transfer_learning_building_dynamics\\comparison.xlsx")
df = df.reset_index()
tidy = df.melt(id_vars='index').rename(columns=str.title)

f, (ax1, ax2) = plt.subplots(ncols=1, nrows=2,
                             sharex=True)
ax1 = sns.barplot(x='Index', y='Value', hue='Variable', data=tidy, ax = ax1)
ax2 = sns.barplot(x='Index', y='Value', hue='Variable', data=tidy)
ax2 =sns.regplot(x='index', y='Online ML',  data=df[1:], order=2,scatter=False)
ax2 =sns.regplot(x='index', y='Online TL',  data=df, order=1,scatter=False)
ax2 = sns.regplot(x='index', y='Static TL',  data=df, order=1,scatter=False)
ax1.set_ylim(0.8, 1.2)
ax2.set_ylim(0, 0.25)
# the upper part does not need its own x axis as it shares one with the lower part
ax1.get_xaxis().set_visible(False)
# by default, each part will get its own "Latency in ms" label, but we want to set a common for the whole figure
# first, remove the y label for both subplots
ax1.set_ylabel("")
ax2.set_ylabel("")
# then, set a new label on the plot (basically just a piece of text) and move it to where it makes sense (requires trial and error)
f.text(0.05, 0.55, "MAE [°C]", va="center", rotation="vertical")

# by default, seaborn also gives each subplot its own legend, which makes no sense at all
# soe remove both default legends first

# then create a new legend and put it to the side of the figure (also requires trial and error)
ax2.legend(loc=(1.025, 0.5), title="Design")

# let's put some ticks on the top of the upper part and bottom of the lower part for style
ax1.xaxis.tick_top()
ax2.xaxis.tick_bottom()

# finally, adjust everything a bit to make it prettier (this just moves everything, best to try and iterate)
f.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)

plt.show()

sns.barplot(x='Index', y='Value', hue='Variable', data=tidy)
#sns.regplot(x='index', y='Online ML',  data=df[1:], order=2,scatter=False)
#sns.regplot(x='index', y='Online TL',  data=df, order=1,scatter=False)
#sns.regplot(x='index', y='Static TL',  data=df, order=1,scatter=False)
plt.show()

ax = sns.scatterplot(x='Value', y='Index', hue='Variable', data=tidy)
#sns.regplot(x='index', y='Online ML',  data=df[1:], order=2,scatter=False)
#sns.regplot(x='index', y='Online TL',  data=df, order=1,scatter=False)
#sns.regplot(x='index', y='Static TL',  data=df, order=1,scatter=False)
ax.invert_yaxis()
plt.show()



ax = sns.barplot(x='Online ML', y='index', data=df)
#sns.regplot(x='index', y='Online ML',  data=df[1:], order=2,scatter=False)
#sns.regplot(x='index', y='Online TL',  data=df, order=1,scatter=False)
#sns.regplot(x='index', y='Static TL',  data=df, order=1,scatter=False)
ax.invert_yaxis()
plt.show()


plt.figure(figsize = (7,10))
ind = np.linspace(51,1,51)
ax = plt.barh(data=df,y=ind,width='Online ML', height=0.3,color='blue')
ax = plt.barh(data=df,y=ind+0.3,width='Static TL', height=0.3,color='orange')
ax = plt.barh(data=df,y=ind+0.6,width='Online TL', height=0.3,color='green')
plt.legend(['Online ML','Static TL','Online TL'])
plt.tight_layout()
plt.show()

tidy.plot( x =tidy.Index, y= tidy.Value, hue=tidy.Variable )
plt.show()




onlineml = tidy.loc[tidy['Variable'] == 'Online ML']
onlinetl = tidy.loc[tidy['Variable'] == 'Online TL']
statictl = tidy.loc[tidy['Variable'] == 'Static TL']

sns.lineplot(data = tidy, x='Index', y='Value', hue='Variable', palette=['k','gold','r'])
sns.scatterplot(data = tidy, x='Index', y='Value', hue='Variable', palette=['k','gold','r'],size=1)
plt.fill_between(onlineml.Index, onlineml.Value, onlinetl.Value, color="green", alpha=0.3)
plt.show()


x=onlineml.Index
y= onlineml.Value
y2= onlinetl.Value
y3 = statictl.Value
x_new = np.linspace(0, 50, 500)
a_BSpline = scipy.interpolate.make_interp_spline(x, y)
a_BSpline2 = scipy.interpolate.make_interp_spline(x, y2)
a_BSpline3 = scipy.interpolate.make_interp_spline(x, y3)
y_new = a_BSpline(x_new)
y_new2 = a_BSpline2(x_new)
y_new3 = a_BSpline3(x_new)
plt.plot(x_new, y_new,'k--')
plt.plot(x_new, y_new2,'gold')
plt.plot(x_new, y_new3,'r--')
plt.fill_between(x_new, y_new, y_new2,where=(y_new>y_new2), color="green", alpha=0.3)
plt.fill_between(x_new, y_new, y_new2,where=(y_new2>y_new) ,color="red", alpha=0.8)
plt.xlim(0,50)
plt.legend(['Online ML','Online TL','Static TL','Improvement','Worsening'])
plt.xlabel('Weeks')
plt.ylabel('MAE [°C]')
plt.show()


import plotly.graph_objects as go

df['index'] = df['index']+1



fig = go.Figure(data=[go.Candlestick(x=df['index'],
                open=df['Online TL'],
                high=df['Online ML'],
                low=df['Online TL'],
                close=df['Online ML'])])

fig.update_layout(xaxis_rangeslider_visible=False,  yaxis_title='MAE [°C]', xaxis_title = 'Weeks', font_family="Times New Roman", template="simple_white")
fig.show()