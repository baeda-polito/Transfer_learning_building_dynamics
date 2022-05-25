import numpy as np
import pandas as pd
# ____________________________________________PREPROCESSING FUNCTIONS___________________________________________________

def define_period_deploy(week):
    len_train = week*1008+48
    len_test = int(len_train+1008+48)
    return len_train, len_test

def define_period_deploy_SW(week):
    len_train = week*1008+48
    len_test = int(len_train+1008+48)
    return len_train, len_test


# create train, val, test datasets
def create_data(df, col_name, l_train, period, l_test):
    train_mx = pd.DataFrame(df[:l_train])
    # val_mx = pd.DataFrame(df[l_init_val:l_val])
    test_mx = pd.DataFrame(df[l_train:l_test])
    train_mx['out'] = train_mx[col_name]
    # val_mx['out'] = val_mx[col_name]
    test_mx['out'] = test_mx[col_name]
    train_mx[col_name] = train_mx[col_name].shift(periods=period) # shifting train_x
    # val_mx[col_name] = val_mx[col_name].shift(periods=period)
    test_mx[col_name] = test_mx[col_name].shift(periods=period)
    train_mx = train_mx.iloc[period:] # delete the Nan
    # val_mx = val_mx.iloc[period:]
    test_mx = test_mx.iloc[period:]
    train_mx = train_mx.reset_index(drop=True) # reset the index of the rows
    # val_mx = val_mx.reset_index(drop=True)
    test_mx = test_mx.reset_index(drop=True)
    return train_mx, test_mx



# create train, val, test datasets
def create_data_SW(df, col_name, l_train, period, l_test):
    train_mx = pd.DataFrame(df[max(l_train-5*1008,0):l_train])
    # val_mx = pd.DataFrame(df[l_init_val:l_val])
    test_mx = pd.DataFrame(df[l_train:l_test])
    train_mx['out'] = train_mx[col_name]
    # val_mx['out'] = val_mx[col_name]
    test_mx['out'] = test_mx[col_name]
    train_mx[col_name] = train_mx[col_name].shift(periods=period) # shifting train_x
    # val_mx[col_name] = val_mx[col_name].shift(periods=period)
    test_mx[col_name] = test_mx[col_name].shift(periods=period)
    train_mx = train_mx.iloc[period:] # delete the Nan
    # val_mx = val_mx.iloc[period:]
    test_mx = test_mx.iloc[period:]
    train_mx = train_mx.reset_index(drop=True) # reset the index of the rows
    # val_mx = val_mx.reset_index(drop=True)
    test_mx = test_mx.reset_index(drop=True)
    return train_mx, test_mx