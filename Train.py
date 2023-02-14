""" 
Project: Bigdata / LSTM / Start Date: 210103 / Ver.1 / made by JOKO
Revised: 220601
"""
# Load Libraries

#Basic
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import gc
import datetime as dt
from datetime import datetime, timedelta 
from openpyxl import load_workbook
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error
import dask.dataframe as dd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import seaborn as sns
import collections as co
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from ipywidgets import widgets, interactive
from typing import Union
from tqdm.notebook import tqdm_notebook as tqdm
from itertools import cycle
from fastprogress import master_bar, progress_bar

"""
Output 2개일 때, 새롭게 하려는 거
"""
def Set_Var(Choose_num_of_first_output_parameter, Choose_num_of_second_output_parameter, use_or_not_ProdBOE, use_or_not_DailyRate, use_or_not_CumProdBOE, 
use_or_not_OilProd, use_or_not_OilRate, use_or_not_CumOil, use_or_not_GasProd, use_or_not_GasRate, use_or_not_CumGas, 
use_or_not_WaterProd, use_or_not_CumMonth, use_or_not_ProdDays, use_or_not_CumProdDay, use_or_not_Shutin, use_or_not_Refrac):

    num_of_features = use_or_not_ProdBOE + use_or_not_DailyRate + use_or_not_CumProdBOE + use_or_not_OilProd + use_or_not_OilRate + use_or_not_CumOil + use_or_not_GasProd + use_or_not_GasRate + use_or_not_CumGas + use_or_not_WaterProd + use_or_not_CumMonth + use_or_not_ProdDays + use_or_not_CumProdDay + use_or_not_Shutin + use_or_not_Refrac
    tmp_1 = [use_or_not_ProdBOE, use_or_not_DailyRate, use_or_not_CumProdBOE, use_or_not_OilProd, use_or_not_OilRate, use_or_not_CumOil, use_or_not_GasProd, use_or_not_GasRate, use_or_not_CumGas, use_or_not_WaterProd, use_or_not_CumMonth, use_or_not_ProdDays, use_or_not_CumProdDay, use_or_not_Shutin, use_or_not_Refrac]
    tmp_2 = ["Prod_BOE", "ProdRate_BOE", "CumProd_BOE", 'LiquidsProd_BBL', 'Liq_rate', 'CumLiquids_BBL', 'GasProd_MCF', 'Gas_rate','CumGas_MCF', 'WaterProd_BBL', "TotalProdMonths", "ProducingDays", 'CumProdDay', "ShutinMonths", "Refrac"]
    used_features = []

    used_features = np.append(used_features, tmp_2[Choose_num_of_first_output_parameter - 1])
    used_features = np.append(used_features, tmp_2[Choose_num_of_second_output_parameter - 1])

    tmp_1 = np.delete(tmp_1, Choose_num_of_first_output_parameter - 1)
    tmp_2 = np.delete(tmp_2, Choose_num_of_first_output_parameter - 1)

    tmp_1 = np.delete(tmp_1, Choose_num_of_second_output_parameter - 2)
    tmp_2 = np.delete(tmp_2, Choose_num_of_second_output_parameter - 2)

    for idx, val in enumerate(tmp_1):
        if val == 1:
            used_features = np.append(used_features, tmp_2[idx])

    name_of_used_features = ''
    for name in used_features:
        name_of_used_features = name_of_used_features + '_' + name
    
    print("Input features =", used_features) 
    print("Onput features =", used_features[0], used_features[1])

    return used_features, name_of_used_features, num_of_features

"""
Output 1개일 때, 기존에 하던 거
"""

# def Set_Var(Choose_num_of_output_parameter, use_or_not_ProdBOE, use_or_not_DailyRate, use_or_not_CumProdBOE, use_or_not_OilProd, use_or_not_CumOil, use_or_not_GasProd, use_or_not_CumGas, use_or_not_WaterProd, use_or_not_CumMonth, use_or_not_ProdDays, use_or_not_Shutin, use_or_not_Refrac):
#     num_of_features = use_or_not_CumMonth + use_or_not_ProdDays + use_or_not_Shutin  + use_or_not_ProdBOE + use_or_not_DailyRate + use_or_not_CumProdBOE + use_or_not_OilProd + use_or_not_CumOil + use_or_not_GasProd + use_or_not_CumGas + use_or_not_WaterProd + use_or_not_Refrac
#     tmp_1 = [use_or_not_ProdBOE, use_or_not_DailyRate, use_or_not_CumProdBOE, use_or_not_OilProd, use_or_not_CumOil, use_or_not_GasProd, use_or_not_CumGas, use_or_not_WaterProd, use_or_not_CumMonth, use_or_not_ProdDays, use_or_not_Shutin, use_or_not_Refrac]
#     tmp_2 = ["Prod_BOE", "ProdRate_BOE", "CumProd_BOE", 'LiquidsProd_BBL', 'CumLiquids_BBL', 'GasProd_MCF', 'CumGas_MCF','WaterProd_BBL', "TotalProdMonths", "ProducingDays", "ShutinMonths", "Refrac"]
#     used_features = []

#     used_features = np.append(used_features, tmp_2[Choose_num_of_output_parameter - 1])
#     tmp_1 = np.delete(tmp_1, Choose_num_of_output_parameter - 1)
#     tmp_2 = np.delete(tmp_2, Choose_num_of_output_parameter - 1)

#     for idx, val in enumerate(tmp_1):
#         if val == 1:
#             used_features = np.append(used_features, tmp_2[idx])

#     name_of_used_features = ''
#     for name in used_features:
#         name_of_used_features = name_of_used_features + '_' + name
    
#     return used_features, name_of_used_features


def TrainModel(current_path, use_datetime, lamda, choose_scaler, used_features, name_of_used_features, num_of_features, seed,
num_of_out_features, seq_length, batch_size=128, num_epochs=300, learning_rate=1e-3, hidden_size=128, num_layers=3, num_classes=1, best_val_loss=0.1):

    pd.set_option('max_columns', 50)
    plt.style.use('bmh')
    color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    if not os.path.exists(current_path + '\\Model'):
            os.makedirs(current_path + '\\Model')

    ###  This function creates a sliding window or sequences of 28 days and one day label ####
    ###  For Multiple features                                                            ####
    # def sliding_windows_mutli_features(data, seq_length):
    #     x = []
    #     y = []
    #     for i in range((data.shape[0])-seq_length-1):
    #         _x = data[i:(i+seq_length), :] ## 3 columns for features  
    #         _y = data[i+seq_length, 0:2] ## column 0 contains the labbel
    #         x.append(_x)
    #         y.append(_y)

    #     return np.array(x), np.array(y).reshape(-1, 2)

    ###  This function creates a sliding window or sequences of 28 days and one day label ####
    ###  For Multiple features                                                            ####
    def sliding_windows_mutli_features(data, seq_length):
        x = []
        y = []
        for i in range((data.shape[0])-seq_length):
            _x_prod = data[i:(i+seq_length), 0:2] ## 3 columns for features
            _x_oper = data[i+1:(i+seq_length+1), 2:]
            _x = np.concatenate((_x_prod, _x_oper), axis=1)  
            _y = data[i+seq_length, 0:2] ## column 0 contains the labbel
            x.append(_x)
            y.append(_y)

        return np.array(x), np.array(y).reshape(-1, 2)

    class LSTM2(nn.Module):

        def __init__(self, num_classes, input_size, hidden_size, num_layers):
            super(LSTM2, self).__init__()
            
            self.num_classes = num_classes
            self.num_layers = num_layers
            self.input_size = input_size
            self.hidden_size = hidden_size
            
            self.batch_size = 1
            #self.seq_length = seq_length
            
            self.LSTM2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout = 0.2)
        
            self.fc1 = nn.Linear(hidden_size, 256)
            self.bn1 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.dp1 = nn.Dropout(0.25)
            
            self.fc2 = nn.Linear(256, 128)
            self.bn2 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.dp2 = nn.Dropout(0.2)

            self.fc3= nn.Linear(128, 2)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            h_1 = Variable(torch.zeros(
                self.num_layers, x.size(0), self.hidden_size).to(device))
            c_1 = Variable(torch.zeros(
                self.num_layers, x.size(0), self.hidden_size).to(device))
            _, (hn, cn) = self.LSTM2(x, (h_1, c_1))
        
            #print("hidden state shpe is:",hn.size())
            y = hn.view(-1, self.hidden_size)
            
            final_state = hn.view(self.num_layers, x.size(0), self.hidden_size)[-1]
            #print("final state shape is:",final_state.shape)
            
            x0 = self.fc1(final_state)
            x0 = self.bn1(x0)
            x0 = self.dp1(x0)
            x0 = self.relu(x0)
            
            x0 = self.fc2(x0)
            x0 = self.bn2(x0)
            x0 = self.dp2(x0)
            x0 = self.relu(x0)
            
            out = self.fc3(x0)
            #print(out.size())
            return out

    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    class CustomDataset(Dataset):
        def __init__(self, x, y):
            self.x_data = x
            self.y_data = y

        def __len__(self):
            return len(self.x_data)
        
        def __getitem__(self, idx):
            x = Variable(torch.Tensor(self.x_data[idx]))
            y = Variable(torch.Tensor(self.y_data[idx]))
            return x, y

    train_input_path = current_path + '\\Data\\After_Preprocess\\' + use_datetime + '\\Train\\'
    train_input_path_refrac = current_path + '\\Data\\After_Preprocess\\' + use_datetime + '\\Refrac\\Train\\'
    scale_input_path = current_path + '\\Data\\After_Preprocess\\' + use_datetime + '\\ForScale\\'
    train_file_list = os.listdir(train_input_path)
    train_file_list_refrac = os.listdir(train_input_path_refrac)
    scale_input_list = os.listdir(scale_input_path)

    np.random.seed(seed)
    np.random.shuffle(train_file_list)
    np.random.shuffle(train_file_list_refrac)

    df_tot_tmp = pd.DataFrame()
    df_tot = pd.DataFrame(columns=used_features)
    for file in scale_input_list:
        df_tmp = pd.read_csv(scale_input_path + file)
        df_tot_tmp = pd.concat([df_tot_tmp, df_tmp])

    for x in used_features:
        df_tot[x] = df_tot_tmp[x]
    # df_tot['GasProd_MCF'] = np.log(df_tot['GasProd_MCF'])
    # df_tot['CumGas_MCF'] = np.log(df_tot['CumGas_MCF'])

    if choose_scaler == 1:
        scaler = StandardScaler()
    elif choose_scaler == 2:
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif choose_scaler == 3:
        scaler = RobustScaler()
    elif choose_scaler == 4:
        scaler = QuantileTransformer()
    elif choose_scaler == 5:
        scaler = PowerTransformer()

    data_for_scale = np.array(df_tot)
    scaler.fit(data_for_scale)
    
    ## For No Refrac Data
    X = []
    Y = []
    for _, file in enumerate(train_file_list):
        df_tmp = pd.read_csv(train_input_path + file)
        df = pd.DataFrame(columns=used_features)
        for x in used_features:
            df[x] = df_tmp[x]
        data = np.array(df)
        if len(data) >= seq_length:
            train_data_normalized = scaler.transform(data.reshape(-1, num_of_features))
            x, y = sliding_windows_mutli_features(train_data_normalized, seq_length)
            if len(x) == 0:
                pass
            else:
                X = np.concatenate((X, x), axis=None)
                Y = np.concatenate((Y, y), axis=None)
        else:
            pass

    X = np.reshape(X, (-1, seq_length, num_of_features))
    Y = np.reshape(Y, (-1, num_of_out_features))

    train_size = int(len(Y) * 0.82)
    test_size = len(Y) - train_size

    trainX = np.array(X[0:train_size])
    trainY = np.array(Y[0:train_size])
    # print(trainX)
    testX = np.array(X[train_size:len(X)])
    testY = np.array(Y[train_size:len(Y)])

    ## For Refrac Data
    X_refrac = []
    Y_refrac = []
    for _, file in enumerate(train_file_list_refrac):
        df_temp = pd.read_csv(train_input_path_refrac + file)
        df = pd.DataFrame(columns=used_features)
        for x in used_features:
            df[x] = df_temp[[x]]
        data = np.array(df)
        if len(data) >= seq_length:
            train_data_normalized = scaler.transform(data.reshape(-1, num_of_features))
            x, y = sliding_windows_mutli_features(train_data_normalized, seq_length)
            if len(x) == 0:
                pass
            else:
                X_refrac = np.concatenate((X_refrac, x), axis=None)
                Y_refrac = np.concatenate((Y_refrac, y), axis=None)
        else:
            pass

    X_refrac = np.reshape(X_refrac, (-1, seq_length, num_of_features))
    Y_refrac = np.reshape(Y_refrac, (-1, num_of_out_features))

    train_size_refrac = int(len(Y_refrac) * 0.82)
    test_size_refrac = len(Y_refrac) - train_size_refrac

    trainX_refrac = np.array(X_refrac[0:train_size_refrac])
    trainY_refrac = np.array(Y_refrac[0:train_size_refrac])

    testX_refrac = np.array(X_refrac[train_size_refrac:len(X_refrac)])
    testY_refrac = np.array(Y_refrac[train_size_refrac:len(Y_refrac)])
    
    trainX = np.concatenate((trainX, trainX_refrac), axis=0)
    trainY = np.concatenate((trainY, trainY_refrac), axis=0)
    print(np.shape(trainX))

    testX = np.concatenate((testX, testX_refrac), axis=0)
    testY = np.concatenate((testY, testY_refrac), axis=0)
    print(np.shape(testX))


    testX = Variable(torch.Tensor(testX))
    testY = Variable(torch.Tensor(testY))

    """ 
    Train Model
    """
    input_size = num_of_features

    train_dataset = CustomDataset(trainX, trainY)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)

    for batch_idx, samples in enumerate(train_dataloader):
        x_train, y_train = samples

    lstm = LSTM2(num_classes, input_size, hidden_size, num_layers)
    lstm.to(device)

    lstm.apply(init_weights)

    criterion = torch.nn.MSELoss().to(device)    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor =0.5, min_lr=1e-7, eps=1e-08)

    # Train the model

    train_dataset = CustomDataset(trainX, trainY)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    # test_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    for epoch in progress_bar(range(num_epochs)):
        lstm.train()
        gc.collect()
        torch.cuda.empty_cache()
        for batch_idx, samples in enumerate(train_dataloader):
            x_train, y_train = samples
            outputs = lstm(x_train.to(device))
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1)
            
            loss = (1 - lamda) * criterion(outputs[0], y_train[0].to(device)) + (lamda) * criterion(outputs[1], y_train[1].to(device))
            # loss = criterion(outputs, y_train.to(device))
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            lstm.eval()
            valid = lstm(testX.to(device))
            val_loss = (1 - lamda) * criterion(valid[0], testY[0].to(device)) + (lamda) * criterion(valid[1], testY[1].to(device))
            # val_loss = criterion(valid, testY.to(device))
            scheduler.step(val_loss)

            if val_loss.cpu().item() < best_val_loss:
                torch.save(lstm.state_dict(), current_path + '\\Model\\Best_model_Case_' + name_of_used_features + '_' + str(seq_length) + 'mon_' + use_datetime + '_GAS' + '.pt')
                # torch.save(lstm.state_dict(), current_path + '\\Model\\Best_model_Case_' + name_of_used_features + '_DC_0.0_' + '_220324.pt')

                print("Saved best model epoch:", epoch, "val loss is:", val_loss.cpu().item(), "train loss is", loss.cpu().item())
                best_val_loss = val_loss.cpu().item()

            if epoch % 10 == 0:
                print("Epoch: %d, loss: %1.5f valid loss:  %1.5f " %(epoch, loss.cpu().item(), val_loss.cpu().item()))

            scheduler.step(val_loss)

