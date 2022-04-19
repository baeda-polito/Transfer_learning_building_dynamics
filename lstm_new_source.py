import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from functions import import_file, min_max_T, normalization, split_multistep_sequences, mean_absolute_percentage_error, create_data, define_period
from NN_architectures import LSTM
from training_testing_functions import train_model, test_model

# ___________________________________________IMPORT AND NORMALIZATION___________________________________________________

# ZONES: CONFROOM_BOT_1, CONFROOM_MID_2, ENCLOSEDOFFICE_BOT_2, OPENOFFICE_BOT_3
#
# zone = 'CONFROOM_BOT_1'
# clm = '5A'
# eff = 'Low'
# list_year = ['2009', 'TMY3']
# occ = 'run_3'
# df = import_file(zone, clm, eff, list_year, occ)


zone = 'CONFROOM_BOT_1'
clm ='5A'
eff = 'High'
year = 'TMY3'
occ = 'run_1'
df = pd.read_csv('data/{}_{}_{}_{}_{}.csv'.format(zone, clm, eff, year, occ), encoding='latin1')
# df = pd.read_csv('data/'+zone+'_'+clm+'_'+eff+'_'+year+'_'+occ+'.csv', encoding='latin1')
del df['Unnamed: 0']
del df[zone+' ZN VAV TERMINAL:Zone Air Terminal VAV Damper Position[]']
del df['Environment:Site Outdoor Air Relative Humidity[%]']

max_T, min_T = min_max_T(df=df, column=zone+' ZN:Zone Mean Air Temperature[C]')

df = normalization(df)


# Files path____________________________________________________________________________________________________________
train_period = '1_year'
test_period = '1_month1year'

train_metrics_path = 'results\\ML\\' + train_period + '\\' + test_period + '\\' +zone+'_'+clm+'_'+eff+'_'+occ+'_ML_'+train_period+'_'+test_period+'_train_metrics.csv'
loss_path = 'results\\ML\\' + train_period + '\\' + test_period + '\\' +zone+'_'+clm+'_'+eff+'_'+occ+'_ML_'+train_period+'_'+test_period+'_train_loss.csv'
test_metrics_path = 'results\\ML\\' + train_period + '\\' + test_period + '\\' +zone+'_'+clm+'_'+eff+'_'+occ+'_ML_'+train_period+'_'+test_period+'_test_metrics.csv'
model_path = 'models\\ML\\'+train_period+'\\'+test_period+'\\'+ zone+'_'+clm+'_'+eff+'_'+occ+'_ML_'+train_period+'_'+test_period+'_weights.pth'


# ______________________________________Datasets_preprocessing__________________________________________________________

l_train, l_test = define_period(df, train_time=train_period, test_period=test_period)


period = 6
train_df, test_df = create_data(df=df, col_name=zone+' ZN:Zone Mean Air Temperature[C]', l_train=l_train, period=period, l_test=l_test)
train_df, test_df = train_df.to_numpy(), test_df.to_numpy()


# ________________________________________Splitting in X, Y data________________________________________________________
n_steps = 48 # (8 hours)
train_X, train_Y = split_multistep_sequences(train_df, n_steps)
test_X, test_Y = split_multistep_sequences(test_df, n_steps)


# Convert medium office to tensors
train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).float()
test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).float()

print(type(train_X), train_X.shape)
print(type(train_Y), train_Y.shape)
print(type(test_X), test_X.shape)
print(type(test_Y), test_Y.shape)


# ================================================= LSTM Structure =====================================================
# HYPER PARAMETERS
lookback = 48
lr = 0.008 #0.008 0.005 #0.009
num_layers = 3 # per ogni modulo lstm
num_hidden = 175

# generalize the number of features, timesteps and outputs
n_features = train_X.shape[2] # 8
n_timesteps = train_X.shape[1] # 6
n_outputs = train_Y.shape[1] # 48

# __________________________________________________TRAINING PHASE______________________________________________________
if test_period == '1_week':
    train_batch_size, test_batch_size = 400, 400
if test_period == '1_month' or '1_month1year':
    train_batch_size, test_batch_size = 900, 900
if test_period == '1_year':
    train_batch_size, test_batch_size = 900, 900

train_data = TensorDataset(train_X, train_Y)
train_dl = DataLoader(train_data, batch_size=train_batch_size, shuffle=False, drop_last=True)

test_data = TensorDataset(test_X, test_Y)
test_dl = DataLoader(test_data, shuffle=False, batch_size=test_batch_size, drop_last=True)

model = LSTM(num_classes=n_outputs, input_size=n_features, hidden_size=num_hidden, num_layers=num_layers)
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
model_load_ml = 'models\\ML\\'+'1_year'+'\\'+'1_year'+'\\'+ zone+'_'+clm+'_'+eff+'_'+occ+'_ML_'+'1_year'+'_'+'1_year'+'_weights.pth'
model.load_state_dict(torch.load(model_load_ml))


# # ______________________________________________________LOAD OPTUNA MODEL_____________________________________________
# year = '2016'
# model_epochs = '90'
# model_lr = '0.007703248894562769'
# model_mape = '0.535291014239192'
# model.load_state_dict(torch.load('optuna_models/OPTUNA_LSTM_train_on_'+year+'_epochs_'+model_epochs+'_lr_'+model_lr+'_MAPE_'+model_mape+'.pth'))


# ______________________________________________________LOAD A MODEL____________________________________________________
# model = LSTM(num_classes=n_outputs, input_size=n_features, hidden_size=num_hidden, num_layers=num_layers)
# period = 'week'
# year = '2015'
# model_epochs = 190
# model_lr = 0.009
# model.load_state_dict(torch.load('train_on_'+period+'_'+year+'_epochs_'+str(model_epochs)+'_lr_'+str(model_lr)+'.pth'))
# model.load_state_dict(torch.load('train_on_10_years_2015_epochs_25_lr_0.008_batch_2000.pth'))
# model.load_state_dict(torch.load('new_source/models/ML/1_year/1_year/CONFROOM_BOT_1_3C_Standard_run_1_ML_1_year_1_year_lr_0.008_weights.pth'))


# epochs = 90
# # Training
# train_loss, train_metrics = train_model(model, epochs=epochs, train_dl=train_dl, optimizer=optimizer, criterion=criterion, train_batch_size=train_batch_size, min_T=min_T, max_T=max_T, train_metrics_path=train_metrics_path, loss_path=loss_path)
# torch.save(model.state_dict(), model_path)


# Testing
y_pred, y_lab, test_metrics = test_model(model=model, test_dl=test_dl, maxT=max_T, minT=min_T, batch_size=test_batch_size, test_metrics_path=test_metrics_path)

#train_results = pd.read_csv(train_metrics_path, encoding='latin1')
test_results = pd.read_csv(test_metrics_path, encoding='latin1')
#loss = pd.read_csv(loss_path, encoding='latin1')


