import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from functions import import_file, min_max_T, normalization, create_data, split_multistep_sequences, split_sequences, mean_absolute_percentage_error, define_period
from NN_architectures import LSTM
from training_testing_functions import train_model, test_model

# _______________________________________________IMPORT DATASET_________________________________________________________
# ZONES: CONFROOM_BOT_1, CONFROOM_MID_2, ENCLOSEDOFFICE_BOT_2, OPENOFFICE_BOT_3
# EFF: Low, Standard, High
# OCC: run_1, run_2, run_3, run_4, run_5
#
# zone = 'CONFROOM_BOT_1'
# clm = '1A'
# eff = 'Low'
# list_year = ['2005', 'TMY3']
# occ = 'run_3'
# df = import_file(zone, clm, eff, list_year, occ)

zone = 'CONFROOM_BOT_1'
clm ='5A'
eff = 'High'
year = 'TMY3'
occ = 'run_3'
df = pd.read_csv('data/{}_{}_{}_{}_{}.csv'.format(zone, clm, eff, year, occ), encoding='latin1')
# df = pd.read_csv('data/'+zone+'_'+clm+'_'+eff+'_'+year+'_'+occ+'.csv', encoding='latin1')
del df['Unnamed: 0']
del df[zone+' ZN VAV TERMINAL:Zone Air Terminal VAV Damper Position[]']
del df['Environment:Site Outdoor Air Relative Humidity[%]']


max_T1, min_T1 = min_max_T(df, zone+' ZN:Zone Mean Air Temperature[C]')

df = normalization(df)


# Files path____________________________________________________________________________________________________________
source_period = '1_year'
train_period = '1_year'
test_period = '1_month1year'

# wi
train_metrics_path_wi = 'results\\ML\\' + source_period + '\\' + test_period + '\\' +zone+'_'+clm+'_'+eff+'_'+occ+'_wi_'+source_period+'_'+test_period+'_train_metrics.csv'
loss_path_wi = 'results\\ML\\' + source_period + '\\' + test_period + '\\' +zone+'_'+clm+'_'+eff+'_'+occ+'_wi_'+source_period+'_'+test_period+'_train_loss.csv'
test_metrics_path_wi = 'results\\ML\\' + source_period + '\\' + test_period + '\\' +zone+'_'+clm+'_'+eff+'_'+occ+'_wi_'+source_period+'_'+test_period+'_test_metrics.csv'
model_path_wi = 'models\\ML\\'+source_period+'\\'+test_period+'\\'+ zone+'_'+clm+'_'+eff+'_'+occ+'_wi_'+source_period+'_'+test_period+'_weights.pth'

# fe
train_metrics_path_fe = 'results\\ML\\' + source_period + '\\' + test_period + '\\' +zone+'_'+clm+'_'+eff+'_'+occ+'_fe_'+source_period+'_'+test_period+'_train_metrics.csv'
loss_path_fe = 'results\\ML\\' + source_period + '\\' + test_period + '\\' +zone+'_'+clm+'_'+eff+'_'+occ+'_fe_'+source_period+'_'+test_period+'_train_loss.csv'
test_metrics_path_fe = 'results\\ML\\' + source_period + '\\' + test_period + '\\' +zone+'_'+clm+'_'+eff+'_'+occ+'_fe_'+source_period+'_'+test_period+'_test_metrics.csv'
model_path_fe = 'models\\ML\\'+source_period+'\\'+test_period+'\\'+ zone+'_'+clm+'_'+eff+'_'+occ+'_fe_'+source_period+'_'+test_period+'_weights.pth'


# ______________________________________Datasets_preprocessing__________________________________________________________

l_train, l_test = define_period(df, train_time=train_period, test_period=test_period)


period = 6
train_df1, test_df1 = create_data(df=df, col_name=zone+' ZN:Zone Mean Air Temperature[C]', l_train=l_train, period=period, l_test=l_test)
train_df1, test_df1 = train_df1.to_numpy(), test_df1.to_numpy()


# Split medium office
n_steps1 = 48
train_X1, train_Y1 = split_multistep_sequences(sequences=train_df1, n_steps=n_steps1)
test_X1, test_Y1 = split_multistep_sequences(sequences=test_df1, n_steps=n_steps1)


# Convert medium office to tensors
train_X1 = torch.from_numpy(train_X1)
train_Y1 = torch.from_numpy(train_Y1)
test_X1 = torch.from_numpy(test_X1)
test_Y1= torch.from_numpy(test_Y1)


print(type(train_X1), train_X1.shape)
print(type(train_Y1), train_Y1.shape)
print(type(test_X1), test_X1.shape)
print(type(test_Y1), test_Y1.shape)


# ================================================= LSTM Structure =====================================================

# HYPER PARAMETERS
lookback = 48
num_layers = 3
num_hidden = 175

n_features = train_X1.shape[2] # 8
n_timesteps = train_X1.shape[1] # 48
n_outputs = train_Y1.shape[1] # 6


# ____________________________________________________LOAD THE MODEL____________________________________________________
model_wi = LSTM(num_classes=n_outputs, input_size=n_features, hidden_size=num_hidden, num_layers=num_layers)
model_fe = LSTM(num_classes=n_outputs, input_size=n_features, hidden_size=num_hidden, num_layers=num_layers)



#cambiare il path
model_load_wi = 'models\\ML\\'+source_period+'\\'+'1_year'+'\\'+ zone+'_'+clm+'_'+eff+'_'+occ+'_wi_'+source_period+'_'+'1_year'+'_weights.pth'
model_wi.load_state_dict(torch.load(model_load_wi))

model_load_fe = 'models\\ML\\'+source_period+'\\'+'1_year'+'\\'+ zone+'_'+clm+'_'+eff+'_'+occ+'_fe_'+source_period+'_'+ '1_year' +'_weights.pth'
model_fe.load_state_dict(torch.load(model_load_fe))

# Define training and validation dataloaders
if test_period == '1_week':
    train_batch_size, test_batch_size = 400, 400
if test_period == '1_month' or '1_month1year':
    train_batch_size, test_batch_size = 900, 900
if test_period == '1_year':
    train_batch_size, test_batch_size = 900, 900


test_data1 = TensorDataset(test_X1, test_Y1)
test_dl1 = DataLoader(test_data1, shuffle=False, batch_size=test_batch_size, drop_last=True)

# Define training and validation dataloaders
if test_period == '1_week':
    train_batch_size, test_batch_size = 400, 400
if test_period == '1_month' or '1_month1year':
    train_batch_size, test_batch_size = 900, 900
if test_period == '1_year':
    train_batch_size, test_batch_size = 900, 900

# train_batch_size = 200
train_data1 = TensorDataset(train_X1, train_Y1)
train_dl1 = DataLoader(train_data1, batch_size=train_batch_size, shuffle=True, drop_last=True)

test_data1 = TensorDataset(test_X1, test_Y1)
test_dl1 = DataLoader(test_data1, shuffle=False, batch_size=test_batch_size, drop_last=True)


# _____________________________________________________FREEZING_PHASE___________________________________________________
def freeze_params(model):
    for param_c in model.lstm.parameters():
                param_c.requires_grad = False
    return model

# ______________________________________________________________________________________________________________________

def transfer(TL=''):

    if TL == 'wi':
        epochs1 = 80
        train_loss_wi, train_metrics_wi= train_model(model_wi, epochs=epochs1, train_dl=train_dl1, optimizer=optimizer_wi, criterion=criterion1, train_batch_size=train_batch_size, min_T=min_T1, max_T=max_T1, train_metrics_path=train_metrics_path_wi, loss_path=loss_path_wi, lr_scheduler=lr_scheduler_wi, mode='TL')
        torch.save(model_wi.state_dict(), model_path_wi)

        y_pred, y_real, test_metrics_wi = test_model(model_wi, test_dl1, max_T1, min_T1, test_batch_size, test_metrics_path_wi)

        return  train_loss_wi, train_metrics_wi,test_metrics_wi

    if TL == 'fe':
        modelfe = freeze_params(model_fe)
        epochs1 = 80
        train_loss_fe, train_metrics_fe = train_model(modelfe, epochs=epochs1, train_dl=train_dl1, optimizer=optimizer_fe, criterion=criterion1, train_batch_size=train_batch_size, min_T=min_T1, max_T=max_T1, train_metrics_path=train_metrics_path_fe, loss_path=loss_path_fe, lr_scheduler=lr_scheduler_fe, mode='TL')
        torch.save(model_fe.state_dict(), model_path_fe)

        y_pred1, y_real1, test_metrics_fe = test_model(model_fe, test_dl1, max_T1, min_T1, test_batch_size, test_metrics_path_fe)

        return  train_loss_fe, train_metrics_fe,test_metrics_fe #



train_loss_fe, train_metrics_fe,test_metrics_fe = transfer(TL='fe') #
train_loss_wi, train_metrics_wi,test_metrics_wi = transfer(TL='wi') #


#
train_results_wi = pd.read_csv(train_metrics_path_wi, encoding='latin1')
test_results_wi = pd.read_csv(test_metrics_path_wi, encoding='latin1')
loss_wi = pd.read_csv(loss_path_wi, encoding='latin1')
#
train_results_fe = pd.read_csv(train_metrics_path_fe, encoding='latin1')
test_results_fe = pd.read_csv(test_metrics_path_fe, encoding='latin1')
loss_fe = pd.read_csv(loss_path_fe, encoding='latin1')


