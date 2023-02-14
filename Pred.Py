" Project: Bigdata / LSTM / Start Date: 210103 / Ver.1 / made by JOKO / Revised: 220601"
# Load Libraries
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

def Prediction(current_path, test_input_path, use_datetime, lamda, choose_scaler, used_features, name_of_used_features, num_of_features, seq_length, use_or_not_EUR_until_Last_months, used_period_lst, EUR_standard, used_period_forBestWorst, learning_rate=1e-3, hidden_size=128, num_layers=3, num_classes=1):
    if not os.path.exists(current_path + '\\Results'):
            os.makedirs(current_path + '\\Results')
    if not os.path.exists(current_path + '\\Results\\' + use_datetime):
            os.makedirs(current_path + '\\Results\\' + use_datetime)
    if not os.path.exists(current_path + '\\Figs\\' + use_datetime):
        os.makedirs(current_path + '\\Figs\\' + use_datetime)
    if not os.path.exists(current_path + '\\Figs\\' + use_datetime + '\\TestData'):
        os.makedirs(current_path + '\\Figs\\' + use_datetime + '\\TestData')
    if not os.path.exists(current_path + '\\Figs\\' + use_datetime + '\\TestData\\Prod'):
        os.makedirs(current_path + '\\Figs\\' + use_datetime + '\\TestData\\Prod')
    if not os.path.exists(current_path + '\\Figs\\' + use_datetime + '\\TestData\\CumProd'):
        os.makedirs(current_path + '\\Figs\\' + use_datetime + '\\TestData\\CumProd')
    if not os.path.exists(current_path + '\\Figs\\' + use_datetime + '\\TestData\\ProdbyCum'):
        os.makedirs(current_path + '\\Figs\\' + use_datetime + '\\TestData\\ProdbyCum')

    pd.set_option('max_columns', 50)
    plt.style.use('bmh')
    color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # current_path = os.getcwd()

    if not os.path.exists(current_path + '\\Model'):
            os.makedirs(current_path + '\\Model')

    ###  This function creates a sliding window or sequences of 28 days and one day label ####
    ###  For Multiple features                                                            ####
    def sliding_windows_mutli_features(data, seq_length):
        x = []
        y = []

        for i in range((data.shape[0])-seq_length-1):
            _x = data[i:(i+seq_length), :] ## 3 columns for features  
            _y = data[i+seq_length, 0] ## column 0 contains the labbel
            x.append(_x)
            y.append(_y)

        return np.array(x), np.array(y).reshape(-1, 1)

    test_file_list = os.listdir(test_input_path)

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

    """ 
    Code for checking Result
    """

    input_size = num_of_features

    lstm = LSTM2(num_classes, input_size, hidden_size, num_layers)
    lstm.to(device)

    lstm.apply(init_weights)

    criterion = torch.nn.MSELoss().to(device)    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor =0.5, min_lr=1e-7, eps=1e-08)

    scale_input_path = current_path + '\\Data\\After_Preprocess\\' + use_datetime + '\\ForScale\\'
    scale_input_list = os.listdir(scale_input_path)


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

    # for model in Model_lst:
    for idx_, val in enumerate(used_period_lst):
        """
        - Control Pannel
        - Set the parameters
        """
        used_period = val   # used_period는 seq_length보다 짧아선 안됨. min of used_period = seq_length
        pred_period = EUR_standard - used_period

        #########################################

        EUR_true_lst = []
        EUR_pred_lst = []
        EUR_true_lst_cum = []
        EUR_pred_lst_cum = []
        EUR_pred_lst_comb = []
        EUR_true_lst_prodbycum = []
        EUR_pred_lst_prodbycum = []
        Wellname_lst = []


        for idx in range(len(test_file_list)):
            X = []
            Y = []
            EUR_True = 0
            EUR_Pred = 0
            EUR_sum = 0

            name_of_data = test_file_list[idx]
            df_temp = pd.read_csv(test_input_path + name_of_data)
            df_TEST = pd.DataFrame(columns=used_features)

            if use_or_not_EUR_until_Last_months == 1:
                EUR_standard = len(df_temp)
                pred_period = EUR_standard - used_period
            else: pass
            
            ## 처음부터 데이터 가져올 때, 생산량 외의 정보는 t+1 시점으로 맞춰줘야함
            for x in used_features:
                df_TEST[x] = df_temp[x]
            data = np.array(df_TEST)
            origin_data = data[:used_period, 0]
            origin_data_cum = data[:used_period, 1]

            test_data_normalized = scaler.transform(data.reshape(-1, num_of_features))

            if seq_length <= used_period:
                X_test_prod = test_data_normalized[used_period - seq_length:used_period, 0:2]
                X_test_oper = test_data_normalized[used_period - seq_length+1:used_period+1, 2:]
                X_test = np.concatenate((X_test_prod, X_test_oper), axis=1) 
                Y_test = test_data_normalized[used_period:used_period + pred_period, :]
            else:
                print(" used_period is shorter than the length of seq_length, it should be longer.")

            X_test = np.reshape(X_test, (-1, seq_length, num_of_features))
            Y_test = np.reshape(Y_test, (-1, pred_period, num_of_features))

            Xtest = Variable(torch.Tensor(np.array(X_test)))
            Ytest = Variable(torch.Tensor(np.array(Y_test)))
            Ytest_plot = Ytest.data.numpy()

            predict = []
            Xtest_tmp = Xtest

            for pred in range(pred_period):
                lstm.load_state_dict(torch.load(current_path + '\\Model\\Best_model_Case_' + name_of_used_features + '_' + str(seq_length) + 'mon_' + use_datetime + '_GAS' + '.pt'))
                lstm.eval()
                test_predict = lstm(Xtest_tmp.to(device)) ## seq_len 넣어서 다음거 하나 나옴, 다시 얘를 포함해서 생산량(Output이랑 동일) 6개 맞추고, 나머지 차원애들은 참값을 불러와줘야함.
                data_predict = test_predict.cpu().data.numpy()
                cascade_lst = []
                cascade_lst = np.append(cascade_lst, data_predict[0][0])
                cascade_lst = np.append(cascade_lst, data_predict[0][1])
                
                if pred < pred_period-1:
                    for idx in range(num_of_features-2):
                        cascade_lst = np.append(cascade_lst, Ytest[:, pred+1, idx+2])

                    predict = np.append(predict, np.array(cascade_lst))
                    Xtest_tmp = np.append(Xtest_tmp, cascade_lst)
                    Xtest_tmp = np.reshape(Xtest_tmp, (-1, seq_length+1, num_of_features))
                    Xtest_tmp = Xtest_tmp[:, 1:, :]
                    Xtest_tmp = Variable(torch.Tensor(np.array(Xtest_tmp)))

            for idx in range(num_of_features-2):
                cascade_lst = np.append(cascade_lst, Ytest[:, pred, idx+2])

            predict = np.append(predict, np.array(cascade_lst))
            predict = np.reshape(predict, (-1, pred_period, num_of_features))

            # Inverse Normalizaiton
            predict = scaler.inverse_transform(predict.reshape(-1, num_of_features))
            true = scaler.inverse_transform(Ytest_plot.reshape(-1, num_of_features))

            # Cal EUR by Prod
            data_plot_true = np.append(origin_data, true[:, 0])
            data_plot_pred = np.append(origin_data, predict[:, 0])
            name = name_of_data.strip(".csv")

            EUR_sum = EUR_sum + np.abs(np.sum(predict[:, 0]) - np.sum(true[:, 0]))
            EUR_true = np.sum(true[:, 0]) + np.sum(origin_data)
            EUR_pred = np.sum(predict[:, 0]) + np.sum(origin_data)

            EUR_true_lst = np.append(EUR_true_lst, EUR_true)
            EUR_pred_lst = np.append(EUR_pred_lst, EUR_pred)
            Wellname_lst = np.append(Wellname_lst, name)

            # Cal EUR by CumProd
            data_plot_true_cum = np.append(origin_data_cum, true[:, 1])
            data_plot_pred_cum = np.append(origin_data_cum, predict[:, 1])

            EUR_true_cum = data_plot_true_cum[-1]
            EUR_pred_cum = data_plot_pred_cum[-1]

            EUR_true_lst_cum = np.append(EUR_true_lst_cum, EUR_true_cum)
            EUR_pred_lst_cum = np.append(EUR_pred_lst_cum, EUR_pred_cum)

            # Cal EUR by combination with Prod and CumProd by weight for Learning
            EUR_pred_lst_comb = lamda * EUR_pred_lst + (1-lamda) * EUR_pred_lst_cum

            # Cal EUR by Prod by Cum
            data_plot_true_prodbycum = np.zeros(len(data_plot_true_cum))
            data_plot_pred_prodbycum = np.zeros(len(data_plot_pred_cum))
            origin_data_prodbycum = np.zeros(len(origin_data_cum))

            for idx, value in enumerate(data_plot_true_cum):
                if idx == 0:
                    data_plot_true_prodbycum[idx] = value
                else:
                    data_plot_true_prodbycum[idx] = (value - data_plot_true_cum[idx-1])

            for idx, value in enumerate(data_plot_pred_cum):
                if idx == 0:
                    data_plot_pred_prodbycum[idx] = value
                else:
                    data_plot_pred_prodbycum[idx] = (value - data_plot_pred_cum[idx-1])

            for idx, value in enumerate(origin_data_cum):
                if idx == 0:
                    origin_data_prodbycum[idx] = value
                else:
                    origin_data_prodbycum[idx] = (value - origin_data_cum[idx-1])

            # EUR_sum = EUR_sum + np.abs(np.sum(predict[:, 0]) - np.sum(true[:, 0]))
            EUR_true_prodbycum = np.sum(data_plot_true_prodbycum) + np.sum(origin_data_prodbycum)
            EUR_pred_prodbycum = np.sum(data_plot_pred_prodbycum) + np.sum(origin_data_prodbycum)

            EUR_true_lst_prodbycum = np.append(EUR_true_lst_prodbycum, EUR_true_prodbycum)
            EUR_pred_lst_prodbycum = np.append(EUR_pred_lst_prodbycum, EUR_pred_prodbycum)

            if used_period == 3:
                # Plot for Prod
                figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')
                plt.plot(data_plot_true, 'b')
                plt.plot(data_plot_pred, 'r')
                plt.plot(origin_data, 'k')
                plt.legend(['True', 'Prediction'], fontsize=21)
                plt.suptitle(name, fontsize=23)
                plt.xticks(fontsize=21)
                plt.yticks(fontsize=21)
                plt.ylabel(ylabel = 'Production', fontsize=21)
                plt.xlabel(xlabel = 'Month', fontsize=21)
                plt.savefig(current_path + '\\Figs\\' + use_datetime + '\\TestData\\Prod\\' + name + '.png')
                plt.close()

                # Plot for CumProd
                figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')
                plt.plot(data_plot_true_cum, 'b')
                plt.plot(data_plot_pred_cum, 'r')
                plt.plot(origin_data_cum, 'k')
                plt.legend(['True', 'Prediction'], fontsize=21)
                plt.suptitle(name, fontsize=23)
                plt.xticks(fontsize=21)
                plt.yticks(fontsize=21)
                plt.ylabel(ylabel = 'Production', fontsize=21)
                plt.xlabel(xlabel = 'Month', fontsize=21)
                plt.savefig(current_path + '\\Figs\\' + use_datetime + '\\TestData\\CumProd\\' + name + '.png')
                plt.close()

                # Plot for ProdbyCum            
                figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')
                plt.plot(data_plot_true_prodbycum, 'b')
                plt.plot(data_plot_pred_prodbycum, 'r')
                plt.plot(origin_data_prodbycum, 'k')
                plt.legend(['True', 'Prediction'], fontsize=21)
                plt.suptitle(name, fontsize=23)
                plt.xticks(fontsize=21)
                plt.yticks(fontsize=21)
                plt.ylabel(ylabel = 'Production', fontsize=21)
                plt.xlabel(xlabel = 'Month', fontsize=21)
                plt.savefig(current_path + '\\Figs\\' + use_datetime + '\\TestData\\ProdbyCum\\' + name + '.png')
                plt.close()

        colums_EUR = ['name', 'True', 'Pred']
        df_EUR = pd.DataFrame(columns=colums_EUR)
        df_EUR['name'] = Wellname_lst
        df_EUR['True'] = EUR_true_lst
        df_EUR['Pred'] = EUR_pred_lst
        df_EUR.to_csv(current_path + '/Results/' + use_datetime + '/EUR_scatterplot_' + str(val) + '.csv')

        df_EUR_cum = pd.DataFrame(columns=colums_EUR)
        df_EUR_cum['name'] = Wellname_lst
        df_EUR_cum['True'] = EUR_true_lst_cum
        df_EUR_cum['Pred'] = EUR_pred_lst_cum
        df_EUR_cum.to_csv(current_path + '/Results/' + use_datetime + '/EUR_scatterplot_cum_' + str(val) + '.csv')

        df_EUR_comb = pd.DataFrame(columns=colums_EUR)
        df_EUR_comb['name'] = Wellname_lst
        df_EUR_comb['True'] = EUR_true_lst_cum
        df_EUR_comb['Pred'] = EUR_pred_lst_comb
        df_EUR_comb.to_csv(current_path + '/Results/' + use_datetime + '/EUR_scatterplot_comb_' + str(val) + '.csv')

        df_EUR_prodbycum = pd.DataFrame(columns=colums_EUR)
        df_EUR_prodbycum['name'] = Wellname_lst
        df_EUR_prodbycum['True'] = EUR_true_lst_prodbycum
        df_EUR_prodbycum['Pred'] = EUR_pred_lst_prodbycum
        df_EUR_prodbycum.to_csv(current_path + '/Results/' + use_datetime + '/EUR_scatterplot_prodbycum_' + str(val) + '.csv')







    # ### Best 10, Worst 10 찾기
    # df_EUR_error = (df_EUR['True'] - df_EUR['Pred']) / df_EUR['True']
    # lst_EUR_error = list(abs(df_EUR_error))
    # n = 20

    # def get_top_n(num_array):
    #     num_tmp = num_array[:]   # 얕은 복사를 통해 num_tmp에 num_array 값 저장
    #     top_n_num = []; top_n_idx = [] 

    #     num_array.sort()  # num_array를 오름차순으로 정렬                   
    #     for i in range(n):
    #         top_n_num.append(num_array.pop())    # pop은 맨 뒤의 요소를 반환하니까, 가장 큰 값부터 반환
    #         top_n_idx.append(num_tmp.index(top_n_num[i]))  

    #     return top_n_idx, top_n_num

    # Worst_10 = get_top_n(lst_EUR_error)

    # lst_EUR_error = list(abs(df_EUR_error))

    # def get_down_n(num_array):
    #     num_tmp = num_array[:]   # 얕은 복사를 통해 num_tmp에 num_array 값 저장
    #     down_n_num = []; down_n_idx = [] 

    #     num_array.sort(reverse=True)  # num_array를 내림차순으로 정렬                    
    #     for i in range(n):
    #         down_n_num.append(num_array.pop())    # pop은 맨 뒤의 요소를 반환하니까, 가장 작은 값부터 반환
    #         down_n_idx.append(num_tmp.index(down_n_num[i]))  

    #     return down_n_idx, down_n_num

    # Best_10 = get_down_n(lst_EUR_error)

    # #############################################################################################################
    # if not os.path.exists(current_path + '\\Figs\\' + use_datetime):
    #     os.makedirs(current_path + '\\Figs\\' + use_datetime)
    # if not os.path.exists(current_path + '\\Figs\\' + use_datetime + '\\TestData'):
    #     os.makedirs(current_path + '\\Figs\\' + use_datetime + '\\TestData')
    # if not os.path.exists(current_path + '\\Figs\\' + use_datetime + '\\Best_Case\\'):
    #     os.makedirs(current_path + '\\Figs\\' + use_datetime + '\\Best_Case\\')
    # if not os.path.exists(current_path + '\\Figs\\' + use_datetime + '\\Worst_Case\\'):
    #     os.makedirs(current_path + '\\Figs\\' + use_datetime + '\\Worst_Case\\')

    # for numbering, idx in enumerate(Best_10[0][:]):
    #     name_of_data = test_file_list[idx]
    #     name_of_data_graph = name_of_data.strip(".csv")
    #     df_temp = pd.read_csv(test_input_path + name_of_data)
    #     df_TEST = pd.DataFrame(columns=used_features)
        
    #     EUR_standard = len(df_temp)
    #     pred_period = EUR_standard - used_period_forBestWorst
        
    #     for x in used_features:
    #         df_TEST[x] = df_temp[[x]]
    #     data = np.array(df_TEST)

    #     origin_data = data[:used_period_forBestWorst, 0]
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     test_data_normalized = scaler.fit_transform(data.reshape(-1, num_of_features))

    #     if seq_length <= used_period_forBestWorst:
    #         X_test = test_data_normalized[used_period_forBestWorst - seq_length:used_period_forBestWorst, :]
    #         Y_test = test_data_normalized[used_period_forBestWorst:used_period_forBestWorst + pred_period, :]
    #     else:
    #         print(" used_period_forBestWorst is shorter than the length of seq_length, it should be longer.")

    #     X_test = np.reshape(X_test, (-1, seq_length, num_of_features))
    #     Y_test = np.reshape(Y_test, (-1, pred_period, num_of_features))

    #     Xtest = Variable(torch.Tensor(np.array(X_test)))
    #     Ytest = Variable(torch.Tensor(np.array(Y_test)))
    #     Ytest_plot = Ytest.data.numpy()

    #     predict = []
    #     Xtest_tmp = Xtest

    #     for pred in range(pred_period):
    #         lstm.load_state_dict(torch.load(current_path + '\\Model\\Best_model_Case_' + name_of_used_features + '_' + str(seq_length) + 'mon_' + use_datetime + '_GAS' + '.pt'))
    #         lstm.eval()
    #         test_predict = lstm(Xtest_tmp.to(device)) ## seq_len 넣어서 다음거 하나 나옴, 다시 얘를 포함해서 생산량(Output이랑 동일) 6개 맞추고, 나머지 차원애들은 참값을 불러와줘야함.
    #         data_predict = test_predict.cpu().data.numpy()

    #         cascade_lst = []
    #         cascade_lst = np.append(cascade_lst, data_predict)
    #         ## fearues 갯수에 따라 첫번째 feature를 제외한 나머지 값들은 true값으로 자동으로 할당되도록 함
    #         for _ in range(num_of_features-1):
    #             cascade_lst = np.append(cascade_lst, Ytest[:, pred, _+1])

    #         predict = np.append(predict, np.array(cascade_lst))
    #         Xtest_tmp = np.append(Xtest_tmp, cascade_lst)
    #         Xtest_tmp = np.reshape(Xtest_tmp, (-1, seq_length+1, num_of_features))
    #         Xtest_tmp = Xtest_tmp[:, 1:, :]
    #         Xtest_tmp = Variable(torch.Tensor(np.array(Xtest_tmp)))

    #     predict = np.reshape(predict, (-1, pred_period, num_of_features))

    #     # Inverse Normalizaiton
    #     predict = scaler.inverse_transform(predict.reshape(-1, num_of_features))
    #     true = scaler.inverse_transform(Ytest_plot.reshape(-1, num_of_features))

    #     # Plot
    #     data_plot_true = np.append(origin_data, true[:, 0])
    #     data_plot_pred = np.append(origin_data, predict[:, 0])

    #     figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')
    #     plt.plot(data_plot_true, 'b.')
    #     plt.plot(data_plot_pred, 'r.')
    #     plt.plot(origin_data, 'k.')
    #     plt.legend(['True', 'Prediction'], fontsize=21)
    #     plt.suptitle(name_of_data_graph, fontsize=23)
    #     plt.xticks(fontsize=21)
    #     plt.yticks(fontsize=21)
    #     plt.ylabel(ylabel = 'Production', fontsize=21)
    #     plt.xlabel(xlabel = 'Month', fontsize=21)
        
    #     plt.savefig(current_path + '\\Figs\\' + use_datetime + '\\Best_Case\\' + str(numbering) + '_' +  name_of_data_graph + '.png')
    #     plt.close()

    # ## Worst Case Graph by Cascading
    # #########################################
    # for numbering, idx in enumerate(Worst_10[0][:]):
    #     name_of_data = test_file_list[idx]
    #     name_of_data_graph = name_of_data.strip(".csv")
    #     df_temp = pd.read_csv(test_input_path + name_of_data)
    #     df_TEST = pd.DataFrame(columns=used_features)

    #     EUR_standard = len(df_temp)
    #     pred_period = EUR_standard - used_period_forBestWorst

    #     for x in used_features:
    #         df_TEST[x] = df_temp[[x]]
    #     data = np.array(df_TEST)

    #     origin_data = data[:used_period_forBestWorst, 0]
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     test_data_normalized = scaler.fit_transform(data.reshape(-1, num_of_features))

    #     if seq_length <= used_period_forBestWorst:
    #         X_test = test_data_normalized[used_period_forBestWorst - seq_length:used_period_forBestWorst, :]
    #         Y_test = test_data_normalized[used_period_forBestWorst:used_period_forBestWorst + pred_period, :]
    #     else:
    #         print(" used_period_forBestWorst is shorter than the length of seq_length, it should be longer.")

    #     X_test = np.reshape(X_test, (-1, seq_length, num_of_features))
    #     Y_test = np.reshape(Y_test, (-1, pred_period, num_of_features))

    #     Xtest = Variable(torch.Tensor(np.array(X_test)))
    #     Ytest = Variable(torch.Tensor(np.array(Y_test)))
    #     Ytest_plot = Ytest.data.numpy()

    #     predict = []
    #     Xtest_tmp = Xtest

    #     for pred in range(pred_period):
    #         lstm.load_state_dict(torch.load(current_path + '\\Model\\Best_model_Case_' + name_of_used_features + '_' + str(seq_length) + 'mon_' + use_datetime + '_GAS' + '.pt'))
    #         lstm.eval()
    #         test_predict = lstm(Xtest_tmp.to(device)) ## seq_len 넣어서 다음거 하나 나옴, 다시 얘를 포함해서 생산량(Output이랑 동일) 6개 맞추고, 나머지 차원애들은 참값을 불러와줘야함.
    #         data_predict = test_predict.cpu().data.numpy()

    #         cascade_lst = []
    #         cascade_lst = np.append(cascade_lst, data_predict)
    #         ## fearues 갯수에 따라 첫번째 feature를 제외한 나머지 값들은 true값으로 자동으로 할당되도록 함
    #         for _ in range(num_of_features-1):
    #             cascade_lst = np.append(cascade_lst, Ytest[:, pred, _+1])

    #         predict = np.append(predict, np.array(cascade_lst))
    #         Xtest_tmp = np.append(Xtest_tmp, cascade_lst)
    #         Xtest_tmp = np.reshape(Xtest_tmp, (-1, seq_length+1, num_of_features))
    #         Xtest_tmp = Xtest_tmp[:, 1:, :]
    #         Xtest_tmp = Variable(torch.Tensor(np.array(Xtest_tmp)))

    #     predict = np.reshape(predict, (-1, pred_period, num_of_features))

    #     # Inverse Normalizaiton
    #     predict = scaler.inverse_transform(predict.reshape(-1, num_of_features))
    #     true = scaler.inverse_transform(Ytest_plot.reshape(-1, num_of_features))

    #     # Plot
    #     data_plot_true = np.append(origin_data, true[:, 0])
    #     data_plot_pred = np.append(origin_data, predict[:, 0])

    #     figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')
    #     plt.plot(data_plot_true, 'b.')
    #     plt.plot(data_plot_pred, 'r.')
    #     plt.plot(origin_data, 'k.')
    #     plt.legend(['True', 'Prediction'], fontsize=21)
    #     plt.suptitle(name_of_data_graph, fontsize=23)
    #     plt.xticks(fontsize=21)
    #     plt.yticks(fontsize=21)
    #     plt.ylabel(ylabel = 'Production', fontsize=21)
    #     plt.xlabel(xlabel = 'Month', fontsize=21)
    #     plt.savefig(current_path + '\\Figs\\' + use_datetime + '\\Worst_Case\\' + str(numbering) + '_' + name_of_data_graph + '.png')
    #     plt.close()

    # ## Graph all of test dataset
    # #########################################
    # for numbering, idx in enumerate(range(len(test_file_list))):
    #     name_of_data = test_file_list[idx]
    #     name_of_data_graph = name_of_data.strip(".csv")
    #     df_temp = pd.read_csv(test_input_path + name_of_data)
    #     df_TEST = pd.DataFrame(columns=used_features)

    #     EUR_standard = len(df_temp)
    #     pred_period = EUR_standard - used_period_forBestWorst

    #     for x in used_features:
    #         df_TEST[x] = df_temp[[x]]
    #     data = np.array(df_TEST)

    #     origin_data = data[:used_period_forBestWorst, 0]
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     test_data_normalized = scaler.fit_transform(data.reshape(-1, num_of_features))

    #     if seq_length <= used_period_forBestWorst:
    #         X_test = test_data_normalized[used_period_forBestWorst - seq_length:used_period_forBestWorst, :]
    #         Y_test = test_data_normalized[used_period_forBestWorst:used_period_forBestWorst + pred_period, :]
    #     else:
    #         print(" used_period_forBestWorst is shorter than the length of seq_length, it should be longer.")

    #     X_test = np.reshape(X_test, (-1, seq_length, num_of_features))
    #     Y_test = np.reshape(Y_test, (-1, pred_period, num_of_features))

    #     Xtest = Variable(torch.Tensor(np.array(X_test)))
    #     Ytest = Variable(torch.Tensor(np.array(Y_test)))
    #     Ytest_plot = Ytest.data.numpy()

    #     predict = []
    #     Xtest_tmp = Xtest

    #     for pred in range(pred_period):
    #         lstm.load_state_dict(torch.load(current_path + '\\Model\\Best_model_Case_' + name_of_used_features + '_' + str(seq_length) + 'mon_' + use_datetime + '_GAS' + '.pt'))
    #         lstm.eval()
    #         test_predict = lstm(Xtest_tmp.to(device)) ## seq_len 넣어서 다음거 하나 나옴, 다시 얘를 포함해서 생산량(Output이랑 동일) 6개 맞추고, 나머지 차원애들은 참값을 불러와줘야함.
    #         data_predict = test_predict.cpu().data.numpy()

    #         cascade_lst = []
    #         cascade_lst = np.append(cascade_lst, data_predict)
    #         ## fearues 갯수에 따라 첫번째 feature를 제외한 나머지 값들은 true값으로 자동으로 할당되도록 함
    #         for _ in range(num_of_features-1):
    #             cascade_lst = np.append(cascade_lst, Ytest[:, pred, _+1])

    #         predict = np.append(predict, np.array(cascade_lst))
    #         Xtest_tmp = np.append(Xtest_tmp, cascade_lst)
    #         Xtest_tmp = np.reshape(Xtest_tmp, (-1, seq_length+1, num_of_features))
    #         Xtest_tmp = Xtest_tmp[:, 1:, :]
    #         Xtest_tmp = Variable(torch.Tensor(np.array(Xtest_tmp)))

    #     predict = np.reshape(predict, (-1, pred_period, num_of_features))

    #     # Inverse Normalizaiton
    #     predict = scaler.inverse_transform(predict.reshape(-1, num_of_features))
    #     true = scaler.inverse_transform(Ytest_plot.reshape(-1, num_of_features))

    #     # Plot
    #     data_plot_true = np.append(origin_data, true[:, 0])
    #     data_plot_pred = np.append(origin_data, predict[:, 0])

    #     figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')
    #     plt.plot(data_plot_true, 'b.')
    #     plt.plot(data_plot_pred, 'r.')
    #     plt.plot(origin_data, 'k.')
    #     plt.legend(['True', 'Prediction'], fontsize=21)
    #     plt.suptitle(name_of_data_graph, fontsize=23)
    #     plt.xticks(fontsize=21)
    #     plt.yticks(fontsize=21)
    #     plt.ylabel(ylabel = 'Production', fontsize=21)
    #     plt.xlabel(xlabel = 'Month', fontsize=21)
    #     plt.savefig(current_path + '\\Figs\\' + use_datetime + '\\TestData\\' + str(numbering) + '_' + name_of_data_graph + '.png')
    #     plt.close()

