import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler
import pandas as pd
import numpy as np
from utils import import_file, min_max_T, normalization, split_multistep_sequences, mean_absolute_percentage_error, create_data, define_period
from utils_deploy import define_period_deploy
from models import LSTM
from training_testing_functions import train_model, test_model

#dataset load and normalization
zone = 'CONFROOM_BOT_1'
clm ='3C'
eff = 'High'
year = 'TMY3'
occ = 'run_2'
source_period = '1_year'


weeks = np.arange(1, 52)
#for debugging purposes
#weeks = [3,4,5]
#week = 1
result_by_epoch = pd.DataFrame()

for week in weeks:
    # ______________________________________Datasets_preprocessing__________________________________________________________

    df = pd.read_csv('data/{}_{}_{}_{}_{}.csv'.format(zone, clm, eff, year, occ), encoding='latin1')
    del df['Unnamed: 0']
    del df[zone + ' ZN VAV TERMINAL:Zone Air Terminal VAV Damper Position[]']
    del df['Environment:Site Outdoor Air Relative Humidity[%]']

    max_T, min_T = min_max_T(df=df, column=zone + ' ZN:Zone Mean Air Temperature[C]')
    df = normalization(df)
    l_train, l_test = define_period_deploy(week)
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
    # ================================================= LSTM Structure =====================================================
    # HYPER PARAMETERS
    lookback = 48
    num_layers = 3 # per ogni modulo lstm
    num_hidden = 175
    # generalize the number of features, timesteps and outputs
    n_features = train_X.shape[2] # 8
    n_timesteps = train_X.shape[1] # 6
    n_outputs = train_Y.shape[1] # 48
    train_batch_size=400
    test_batch_size = 400  #??
    train_data = TensorDataset(train_X, train_Y)
    train_dl = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, drop_last=True)
    test_data = TensorDataset(test_X, test_Y)
    test_dl = DataLoader(test_data, shuffle=False, batch_size=test_batch_size, drop_last=True)
    model = LSTM(num_classes=n_outputs, input_size=n_features, hidden_size=num_hidden, num_layers=num_layers)
    model_load_wi = 'C:\\Users\\BAEDA\\PycharmProjects\\Transfer_learning_building_dynamics\\models\\TL\\1_year\\1_week\\CONFROOM_BOT_1_3C_High_run_2_wi_1_year_1_week_weights.pth'
    model.load_state_dict(torch.load(model_load_wi))
    # DEFINE CRITERION, OPTIMIZER WITH SMALLER LR RATE AND LR SCHEDULER_____________________________________________________
    criterion = torch.nn.MSELoss()
    lr1 = 0.002
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr1)
    # Decay LR (learning rate) by a factor of 0.1 every 7 epochs
    step_size1 = 100
    lr_scheduler_wi = lr_scheduler.StepLR(optimizer, step_size=step_size1, gamma=0.1, verbose=True)
    epochs = 80
    # # Training
    train_metrics_path = 'deployment\\STL\\' + str(week) +'_train_metrics.csv'
    loss_path = 'deployment\\STL\\' + str(week) +'_train_loss.csv'
    test_metrics_path = 'deployment\\STL\\' + str(week) +'_test_metrics.csv'
    results_path = 'deployment\\STL\\' + '_results_by_epoch.csv'
    model_path = 'deployment\\STL\\' + str(week) +'_weights.pth'
    #train_loss, train_metrics = train_model(model, epochs=epochs, train_dl=train_dl, optimizer=optimizer, criterion=criterion, train_batch_size=train_batch_size, min_T=min_T, max_T=max_T, train_metrics_path=train_metrics_path, loss_path=loss_path,lr_scheduler=lr_scheduler_wi, mode='TL')
    #torch.save(model.state_dict(), model_path)
    # Testing
    #torch.load(model.state_dict(), model_path)
    y_pred, y_lab, test_metrics = test_model(model=model, test_dl=test_dl, maxT=max_T, minT=min_T, batch_size=test_batch_size, test_metrics_path=test_metrics_path)
    test_results = pd.read_csv(test_metrics_path, encoding='latin1')
    result_by_epoch =  result_by_epoch.append(test_metrics)
    results_by_epoch = result_by_epoch.to_csv(path_or_buf=results_path, sep=',', decimal='.', index=False)
