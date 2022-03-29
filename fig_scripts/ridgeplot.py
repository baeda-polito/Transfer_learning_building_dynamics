import pandas as pd
import os
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_theme(style="ticks")
plt.rcParams["font.family"] = "Times New Roman"
########## dataset loading
zone = 'CONFROOM_BOT_1'
clm ='3C'
eff = 'High'
year = 'TMY3'
occ = 'run_3'
df = pd.read_csv('data/{}_{}_{}_{}_{}.csv'.format(zone, clm, eff, year, occ), encoding='latin1')
# df = pd.read_csv('data/'+zone+'_'+clm+'_'+eff+'_'+year+'_'+occ+'.csv', encoding='latin1')
del df['Unnamed: 0']
del df[zone+' ZN VAV TERMINAL:Zone Air Terminal VAV Damper Position[]']
del df['Environment:Site Outdoor Air Relative Humidity[%]']
df = pd.DataFrame(df.iloc[:,4])
a,b,c,d,e,f,g,h,i,l,m,n = np.split(df, 12)
dataframes = [a,b,c,d,e,f,g,h,i,l,m,n]
for el in dataframes:
    el = el.reset_index(inplace=True,drop=True)
df_ridge = pd.concat([a,b,c,d,e,f,g,h,i,l,m,n],axis=0,ignore_index=True)
clm ='1A'
df = pd.read_csv('data/{}_{}_{}_{}_{}.csv'.format(zone, clm, eff, year, occ), encoding='latin1')
# df = pd.read_csv('data/'+zone+'_'+clm+'_'+eff+'_'+year+'_'+occ+'.csv', encoding='latin1')
del df['Unnamed: 0']
del df[zone+' ZN VAV TERMINAL:Zone Air Terminal VAV Damper Position[]']
del df['Environment:Site Outdoor Air Relative Humidity[%]']
df = pd.DataFrame(df.iloc[:,4])
a,b,c,d,e,f,g,h,i,l,m,n = np.split(df, 12)
dataframes = [a,b,c,d,e,f,g,h,i,l,m,n]
for el in dataframes:
    el = el.reset_index(inplace=True,drop=True)
df_ridge2 = pd.concat([a,b,c,d,e,f,g,h,i,l,m,n],axis=0,ignore_index=True)
clm ='5A'
df = pd.read_csv('data/{}_{}_{}_{}_{}.csv'.format(zone, clm, eff, year, occ), encoding='latin1')
# df = pd.read_csv('data/'+zone+'_'+clm+'_'+eff+'_'+year+'_'+occ+'.csv', encoding='latin1')
del df['Unnamed: 0']
del df[zone+' ZN VAV TERMINAL:Zone Air Terminal VAV Damper Position[]']
del df['Environment:Site Outdoor Air Relative Humidity[%]']
df = pd.DataFrame(df.iloc[:,4])
a,b,c,d,e,f,g,h,i,l,m,n = np.split(df, 12)
dataframes = [a,b,c,d,e,f,g,h,i,l,m,n]
j=0
for el in dataframes:
    el['mese']=j
    j=j+1
    el = el.reset_index(inplace=True,drop=True)
df_ridge3 = pd.concat([a,b,c,d,e,f,g,h,i,l,m,n],axis=0,ignore_index=True)


df_tot = pd.concat([df_ridge,df_ridge2,df_ridge3],axis=1)
df_tot.columns=['1','2','3','mese']


subs = {0:'January',1:'February',2:'March',3:'April',4:'May',5:'June',6:'July',7:'August',8:'September',9:'October',10:'November',11:'December'}
df_tot.mese = list(map(subs.get, df_tot.mese))

df_tot2=df_tot.groupby("mese",sort=False)

###########à joylot
import joypy
fig,ax = joypy.joyplot(df_tot2,by='mese',alpha=0.6,ylim='own',overlap=0.5,color=["#eab676","#9d403f","#1e81b0"])
plt.xlabel('Temperature [°C]',fontsize=13)
plt.ylabel('Month')
#plt.ylim([0,0.5])
#plt.xlim([-10,30])
plt.savefig('fig_scripts/img/joyplot.png',dpi=400)
plt.show()
