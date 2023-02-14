from __future__ import division
import numpy as np
import pandas as pd
import os
from openpyxl import load_workbook
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
# from sklearn import preprocessing, metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import matthews_corrcoefconda 
import warnings
from itertools import cycle
from fastprogress import master_bar, progress_bar
# from pyswarms.utils.plotters import plot_contour, plot_surface
import random
import shutil


pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
# pd.set_option('max_columns', 50)
plt.style.use('bmh')
warnings.filterwarnings('ignore')
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

def MakeFolder(current_path, datetime):
    if not os.path.exists(current_path + '\\Data'):
        os.makedirs(current_path + '\\Data')
    if not os.path.exists(current_path + '\\Data\\Original'):
        os.makedirs(current_path + '\\Data\\Original')
    if not os.path.exists(current_path + '\\Data\\Before_Preprocess'):
        os.makedirs(current_path + '\\Data\\Before_Preprocess')
    if not os.path.exists(current_path + '\\Data\\Before_Preprocess\\' + datetime):
        os.makedirs(current_path + '\\Data\\Before_Preprocess\\' + datetime)
    if not os.path.exists(current_path + '\\Data\\After_Preprocess'):
        os.makedirs(current_path + '\\Data\\After_Preprocess')
    if not os.path.exists(current_path + '\\Data\\After_Preprocess\\' + datetime):
        os.makedirs(current_path + '\\Data\\After_Preprocess\\' + datetime)
    if not os.path.exists(current_path + '\\Data\\After_Preprocess\\' + datetime + '\\Train'):
        os.makedirs(current_path + '\\Data\\After_Preprocess\\' + datetime + '\\Train')
    if not os.path.exists(current_path + '\\Data\\After_Preprocess\\' + datetime + '\\Test'):
        os.makedirs(current_path + '\\Data\\After_Preprocess\\' + datetime + '\\Test')

def DivideSource(current_path):
    ## Divide source data to original data
    source_path = current_path + '\\Source\\Production\\'
    original_path = current_path + '\\Data\\Original\\'
    file_list_source = os.listdir(source_path)

    for file in file_list_source:
        excel_source = pd.read_csv(source_path + file)
        df = pd.DataFrame(excel_source)
        start_row = 0
        end_row = 0

        for idx in range(len(df)):
            if idx == 0:
                pass

            else :
                if df['API_UWI'][idx] == df['API_UWI'][idx-1]:
                    end_row = idx
                else:
                    df_tmp = df.iloc[start_row:end_row]
                    df_tmp.to_csv(original_path + df['API_UWI'][end_row] + '.csv', index=False)
                    start_row = idx

def ClassifyByReportedType(current_path, datetime):
        ## Original data classify by "ProductionReportedMethod" column
        ## Define (REPORTED + nan) are pure Reported data and (DCA + PENDING) are pure DCA data. Other case is mixed data. 

        input_path = current_path + '\\Data\\Original\\'
        file_list = os.listdir(input_path)
        ### Completion Data 열어두기
        completion_path = current_path + '\\Source\\Completion\\'
        completion_filename = os.listdir(completion_path)
        completion_source = pd.read_csv(completion_path + completion_filename[0]) ## 하나밖에없어서 첫번째 값 사용
        df_comp = pd.DataFrame(completion_source)
        Refrac_lst = [] ## Refrac 유정 확인하기 위한 리스트로, Completion이 두 번 이상 되었을 경우 저장

        if not os.path.exists(current_path + '\\Data\\Before_Preprocess\\' + datetime + '\\REPORTED'):
                os.makedirs(current_path + '\\Data\\Before_Preprocess\\' + datetime + '\\REPORTED')

        if not os.path.exists(current_path + '\\Data\\Before_Preprocess\\' + datetime + '\\REPORTED\\OIL'):
                os.makedirs(current_path + '\\Data\\Before_Preprocess\\' + datetime + '\\REPORTED\\OIL')

        if not os.path.exists(current_path + '\\Data\\Before_Preprocess\\' + datetime + '\\REPORTED\\GAS'):
                os.makedirs(current_path + '\\Data\\Before_Preprocess\\' + datetime + '\\REPORTED\\GAS')

        if not os.path.exists(current_path + '\\Data\\Before_Preprocess\\' + datetime + '\\REPORTED\\OILNGAS'):
                os.makedirs(current_path + '\\Data\\Before_Preprocess\\' + datetime + '\\REPORTED\\OILNGAS')

        if not os.path.exists(current_path + '\\Data\\Before_Preprocess\\' + datetime + '\\DCA'):
                os.makedirs(current_path + '\\Data\\Before_Preprocess\\' + datetime + '\\DCA')

        if not os.path.exists(current_path + '\\Data\\Before_Preprocess\\' + datetime + '\\MIXED'):
                os.makedirs(current_path + '\\Data\\Before_Preprocess\\' + datetime + '\\MIXED')

        for file in file_list:
                excel_source = pd.read_csv(input_path + file)
                df_tmp = pd.DataFrame(excel_source)
                criterian_Except_NaN = df_tmp['ProductionReportedMethod'].count() ## df 에서 빈칸이 if문 조건에서 인식이 안되어서 일단 count()함수로 갯수 세고, 다음 줄에서와 같이 'NaN' 값을 직접 넣어줌.
                df_tmp = df_tmp.replace(np.nan, 'NaN', regex = True)
                criteria = 0

                for idx in range(len(df_tmp['ProductionReportedMethod'])):
                        if (df_tmp['ProductionReportedMethod'][idx] == 'REPORTED') or (df_tmp['ProductionReportedMethod'][idx] == 'PENDING'):
                                criteria += 1
                        elif (df_tmp['ProductionReportedMethod'][idx] == 'DCA'):
                                criteria -= 1

                if criteria == criterian_Except_NaN:
                        file_name_tmp = file.strip('.csv')
                        lst_tmp = list(df_comp.index[df_comp['API_UWI'] == file_name_tmp])

                        if df_comp['ENVWellType'][lst_tmp[-1]] == 'OIL':
                                df_tmp.to_csv(current_path + '\\Data\\Before_Preprocess\\' + datetime + '\\REPORTED\\OIL\\' + file, index=False)
                        if df_comp['ENVWellType'][lst_tmp[-1]] == 'GAS':
                                df_tmp.to_csv(current_path + '\\Data\\Before_Preprocess\\' + datetime + '\\REPORTED\\GAS\\' + file, index=False)
                        if df_comp['ENVWellType'][lst_tmp[-1]] == 'OIL & GAS':
                                df_tmp.to_csv(current_path + '\\Data\\Before_Preprocess\\' + datetime + '\\REPORTED\\OILNGAS\\' + file, index=False)

                elif criteria == - criterian_Except_NaN:
                        df_tmp.to_csv(current_path + '\\Data\\Before_Preprocess\\' + datetime + '\\DCA\\' + file, index=False)
                else: df_tmp.to_csv(current_path + '\\Data\\Before_Preprocess\\' + datetime + '\\MIXED\\' + file, index=False)

## Preprocess : Generate new data / Cut from max value / Delete Shut-in Period
def Generate_SI(current_path, use_datetime, criteria_for_preprocess, Use_or_not_maxidx_preprocess, minval_for_maxidx, Use_or_not_shutin_preprocess, minval_for_shutin, minDataNum):
    if criteria_for_preprocess == 'GasProd_MCF':
        criteria_for_path = 'GAS'
    else: criteria_for_path = 'OIL'
        

    input_path = current_path + '\\Data\\Before_Preprocess\\' + use_datetime + '//REPORTED//' + criteria_for_path + '//'
    file_list = os.listdir(input_path)
    df_report = pd.DataFrame(columns = ['API_UWI', 'Origin_len', 'Pre-processed_len', 'Max_month', 'Minval_for_maxmonth', 'Shut-in_month', 'Pre-processed_Shut-in_len', 'Min_Prod_BOE_for_shutin'])

    if not os.path.exists(current_path + '\\Data\\Before_Refrac'):
            os.makedirs(current_path + '\\Data\\Before_Refrac')
    if not os.path.exists(current_path + '\\Data\\Before_Refrac\\' + use_datetime):
            os.makedirs(current_path + '\\Data\\Before_Refrac\\' + use_datetime)

    for idx, file in enumerate(file_list):

        excel_source = pd.read_csv(input_path + file, usecols = ['TotalProdMonths', 'ProducingDays', 'Prod_BOE', 'CumProd_BOE', 'LiquidsProd_BBL', 'CumLiquids_BBL', 'GasProd_MCF', 'CumGas_MCF', 'WaterProd_BBL'])
        df_tmp = pd.DataFrame(excel_source)
        ## 'CumLiquids_BBL', 'CumGas_MCF'
        
        ## 생산월수가 1월부터 시작하지 않으면 재할당
        if len(df_tmp) <= 0:
            pass
        else:
            if df_tmp['TotalProdMonths'][0] != 1:
                df_tmp['TotalProdMonths'] = 0
                for idx in range(len(df_tmp['TotalProdMonths'])):
                    df_tmp['TotalProdMonths'][idx] += (idx+1)

            # Cum_Liq, Gas, Cumday, Rate 계산
            Cum_Liq = []
            for idx in range(len(df_tmp)):
                if idx == 0:
                    Cum_Liq = np.append(Cum_Liq, df_tmp['LiquidsProd_BBL'][idx])
                elif idx >= 1:
                    Cum_Liq = np.append(Cum_Liq, df_tmp['CumLiquids_BBL'][idx-1] + df_tmp['LiquidsProd_BBL'][idx])

            df_tmp['CumLiquids_BBL'] = Cum_Liq

            Cum_Gas = 0
            df_tmp['CumGas_MCF'] = Cum_Gas
            for idx in range(len(df_tmp)):
                if idx == 0:
                    df_tmp['CumGas_MCF'][idx] = df_tmp['GasProd_MCF'][idx]
                else:
                    df_tmp['CumGas_MCF'][idx] = df_tmp['GasProd_MCF'][idx] + df_tmp['CumGas_MCF'][idx-1]

            Cum_ProdDay = 0
            df_tmp['CumProdDay'] = Cum_ProdDay
            for idx in range(len(df_tmp)):
                if idx == 0:
                    df_tmp['CumProdDay'][idx] = df_tmp['ProducingDays'][idx]
                else:
                    df_tmp['CumProdDay'][idx] = df_tmp['ProducingDays'][idx] + df_tmp['CumProdDay'][idx-1]

            Liq_rate = 0
            df_tmp['Liq_rate'] = Liq_rate
            for idx in range(len(df_tmp)):
                if df_tmp['ProducingDays'][idx] != 0:
                    df_tmp['Liq_rate'][idx] = float(df_tmp['LiquidsProd_BBL'][idx] / df_tmp['ProducingDays'][idx])
                else:
                    df_tmp['Liq_rate'][idx] = 0
            
            gas_rate = 0
            df_tmp['Gas_rate'] = gas_rate
            for idx in range(len(df_tmp)):
                if df_tmp['ProducingDays'][idx] != 0:
                    df_tmp['Gas_rate'][idx] = float(df_tmp['GasProd_MCF'][idx] / df_tmp['ProducingDays'][idx])
                else:
                    df_tmp['Gas_rate'][idx] = 0

            ## Assign ShutinMonths 
            ShutinMonths = 0
            df_tmp['ShutinMonths'] = ShutinMonths
            for idx in range(len(df_tmp)):
                if df_tmp[criteria_for_preprocess][idx] <= minval_for_shutin:
                    df_tmp['ShutinMonths'][idx+1] = 1

            list_shutin = []

            for idx in range(len(df_tmp)):
                if df_tmp[criteria_for_preprocess][idx] <= minval_for_shutin:
                    list_shutin = np.append(list_shutin, int(idx+1))

            len_original = len(df_tmp)
            
            ## Data pre-process 1: Cut data from the maximum index if it is not more than minval_for_maxidx. Ex. minval_for_maxidx = 5 --> max idx가 5 이하일 경우에만 자름
            if Use_or_not_maxidx_preprocess == 1:
                max_idx = df_tmp[:][criteria_for_preprocess].idxmax(axis=0)
                if max_idx <= minval_for_maxidx:
                    df = pd.DataFrame(df_tmp[max_idx:][:])
                    max_idx = max_idx + 1
                else:
                    df = pd.DataFrame(df_tmp)
                    max_idx = max_idx + 1
            else:
                df = pd.DataFrame(df_tmp)
                max_idx = df_tmp[:][criteria_for_preprocess].idxmax(axis=0) + 1
            
            len_after_maxidx = len(df)


            ## Data pre-process 2: Cut data where shut-in is happen. Note: minval_of Prod_BOE and ProducingDays are "And" Condition.
            if Use_or_not_shutin_preprocess == 1:
                df = df[df[:][criteria_for_preprocess] > minval_for_shutin][:]
                # print("Shutin Here: ", len(df))
            else:
                df = pd.DataFrame(df_tmp)
            

            if len(df) >= minDataNum:
                df.to_csv(current_path + '\\Data\\Before_Refrac\\' + use_datetime + '\\' + file, index=False)
           
            len_after_preprocess = len(df)
            len_of_deleted_shutin = len_after_maxidx - len_after_preprocess

            list_report = {
                'API_UWI' : file,
                'Origin_len' : len_original,
                'Pre-processed_len' : len_after_preprocess,
                'Max_month' : max_idx,
                'Minval_for_maxmonth' : minval_for_maxidx,
                'Shut-in_month' : list_shutin,
                'Pre-processed_Shut-in_len' : len_of_deleted_shutin,
                'Min_Prod_BOE_for_shutin' : minval_for_shutin
            }

            df_report = df_report.append(list_report, ignore_index=True)


    df_report.to_csv(current_path + '\\Data\\Pre-processed_Report' + '_' + use_datetime + '.xlsx', index=False)

def Generate_Refrac(current_path, use_datetime, criteria_for_Refrac, criteria_of_avg, criteria_max_idx, rambda_gas, rambda_water, min_criteria, min_month, EUR_criteria):
    if not os.path.exists(current_path + '\\Data\\After_Refrac\\'):
        os.makedirs(current_path + '\\Data\\After_Refrac')

    if not os.path.exists(current_path + '\\Data\\After_Refrac\\' + use_datetime):
            os.makedirs(current_path + '\\Data\\After_Refrac\\' + use_datetime)

    if not os.path.exists(current_path + '\\Data\\After_Refrac\\' + use_datetime + '\\Train\\'):
            os.makedirs(current_path + '\\Data\\After_Refrac\\' + use_datetime + '\\Train\\')

    if not os.path.exists(current_path + '\\Data\\After_Refrac\\' + use_datetime + '\\Test\\'):
            os.makedirs(current_path + '\\Data\\After_Refrac\\' + use_datetime + '\\Test\\')

    if not os.path.exists(current_path + '\\Data\\After_Refrac\\' + use_datetime + '\\Refrac\\'):
            os.makedirs(current_path + '\\Data\\After_Refrac\\' + use_datetime + '\\Refrac\\')

    if not os.path.exists(current_path + '\\Data\\After_Refrac\\' + use_datetime + '\\Refrac\\Train\\'):
            os.makedirs(current_path + '\\Data\\After_Refrac\\' + use_datetime + '\\Refrac\\Train\\')

    if not os.path.exists(current_path + '\\Data\\After_Refrac\\' + use_datetime + '\\Refrac\\Test\\'):
            os.makedirs(current_path + '\\Data\\After_Refrac\\' + use_datetime + '\\Refrac\\Test\\')

    if not os.path.exists(current_path + '\\Figs\\'):
            os.makedirs(current_path + '\\Figs\\')

    if not os.path.exists(current_path + '\\Figs\\Refrac\\'):
            os.makedirs(current_path + '\\Figs\\Refrac\\')

    if not os.path.exists(current_path + '\\Figs\\Refrac\\' + use_datetime):
            os.makedirs(current_path + '\\Figs\\Refrac\\' + use_datetime)

    input_path = current_path + '/Data/Before_Refrac/' + use_datetime + '/'
    output_path = current_path + '/Data/After_Refrac/' + use_datetime + '/'

    used_time = ['TotalProdMonths']

    file_list = os.listdir(input_path)

    for file in file_list:
        vertical_criteria_gas = 0
        vertical_criteria_water = 0
        treshold_gas = 0
        treshold_water = 0
        idx_tmp = 0
        Refrac = 0
        try:
            excel_source = pd.read_csv(input_path + file)
            df = pd.DataFrame(excel_source)

            if df['GasProd_MCF'].mean() < criteria_of_avg:
                pass
            else:
                df['Refrac'] = Refrac
                
                data_Gas = np.array(df[[criteria_for_Refrac]])
                data_Gas_tmp = []
                data_Water = np.array(df[['WaterProd_BBL']])
                data_Water_tmp = []
                data_Shutin = np.array(df[['ShutinMonths']])
                data_time = np.array(df[['TotalProdMonths']])

                scaler = MinMaxScaler(feature_range=(0, 10))
                data_Gas = scaler.fit_transform(data_Gas)
                data_Water = scaler.fit_transform(data_Water)

                for idx in range(len(df)):
                    if (idx+1) < len(df):
                        if idx < criteria_max_idx:
                            pass
                        else:
                            if df['ShutinMonths'][idx] > 0:
                                idx_tmp = int(idx/2)
                                treshold_gas = np.mean(data_Gas[idx_tmp:idx])
                                treshold_water = np.mean(data_Water[idx_tmp:idx])

                                vertical_criteria_gas = np.mean(data_Gas[idx+1:(idx+7)])
                                vertical_criteria_water = np.mean(data_Water[idx+1:(idx+7)])

                                # idx_tmp = idx

                                if vertical_criteria_gas >= treshold_gas*rambda_gas and vertical_criteria_water >= treshold_water*rambda_water and vertical_criteria_gas >=  min_criteria:
                                # if vertical_criteria_gas >= treshold_gas*rambda_gas and vertical_criteria_gas >=  min_criteria:
                                    df['Refrac'][idx] = 1
                                    if idx <= 5:
                                        pass
                                    else:
                                        for _ in range(idx-6, idx):
                                            if df['Refrac'][_] == 1:
                                                df['Refrac'][_] = 0
                
                if len(df[[criteria_for_Refrac]]) <= EUR_criteria:
                    df.to_csv(output_path + 'Train\\' + file, index=False)
                else:
                    df.to_csv(output_path + 'Test\\' + file, index=False)
                
                
                if np.sum(df['Refrac']) != 0 and len(df) >= min_month:
                    # df.to_excel(output_path + file, index=False)
                    if len(df[criteria_for_Refrac]) <= EUR_criteria:
                        df.to_csv(output_path + 'Refrac\\Train\\' + file, index=False)
                    else:
                        df.to_csv(output_path + 'Refrac\\Test\\' + file, index=False)

                    df_time = pd.DataFrame(columns=used_time)
                    df_Gas = pd.DataFrame()
                    df_Water = pd.DataFrame()
                    df_Shutin = pd.DataFrame()

                    df_Gas[criteria_for_Refrac]  = df[[criteria_for_Refrac]]
                    df_Water['WaterProd_BBL']  = df[['WaterProd_BBL']]
                    df_Shutin['ShutinMonths']  = df[['ShutinMonths']]
                    df_time['TotalProdMonths'] = df[['TotalProdMonths']]

                    data_refrac = np.array(df[['Refrac']] )
                    scaler = MinMaxScaler(feature_range=(0, 10))
                    Gas_scaled = scaler.fit_transform(data_Gas)
                    Water_scaled = scaler.fit_transform(data_Water)
                    file = file.strip(".xlsx")
                    figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')
                    plt.plot(Gas_scaled, 'k', linewidth=2)
                    plt.plot(Water_scaled, 'b')
                    plt.plot(data_Shutin, 'r')
                    plt.plot(data_refrac, 'g')

                    plt.legend([criteria_for_Refrac, 'Water Prod', 'Shut-in month'], fontsize=21)
                    plt.suptitle('%s' % file, fontsize = 21)
                    plt.xticks(fontsize=21)
                    plt.yticks(fontsize=21)
                    plt.ylabel(ylabel = 'Value', fontsize=21)
                    plt.xlabel(xlabel = 'Time', fontsize=21)
                    plt.xlim([0, len(data_time)+5])
                    # os.chdir('c:\\Lab\\Project\\Bigdata\\figs_refrac')
                    plt.savefig(current_path + '/Figs/Refrac/' + use_datetime +'/' + file + '.png')
                    plt.close()

        except Exception as e:
            print("Error become", e)
            pass

def FileListExtract(current_path, use_datetime, seed):
    if not os.path.exists(current_path + '/FileList'):
            os.makedirs(current_path + '/FileList')

    input_path_train = current_path + '/Data/After_Refrac/' + use_datetime + '/Train/'
    input_path_test = current_path + '/Data/After_Refrac/' + use_datetime + '/Test/'

    file_list_input_train = os.listdir(input_path_train)
    file_list_input_test = os.listdir(input_path_test)
    if not os.path.exists(current_path + '/Data/After_Preprocess/' + use_datetime):
            os.makedirs(current_path + '/Data/After_Preprocess/' + use_datetime)
    if not os.path.exists(current_path + '/Data/After_Preprocess/' + use_datetime + '/Train/'):
            os.makedirs(current_path + '/Data/After_Preprocess/' + use_datetime + '/Train/')
    if not os.path.exists(current_path + '/Data/After_Preprocess/' + use_datetime + '/Test/'):
        os.makedirs(current_path + '/Data/After_Preprocess/' + use_datetime + '/Test/')

    train_path = current_path + '/Data/After_Preprocess/' + use_datetime + '/Train/'
    test_path = current_path + '/Data/After_Preprocess/' + use_datetime + '/Test/'

    ## Refrac Data를 우선 4:3으로 나눔
    Refrac_path = current_path + '/Data/After_Refrac/' + use_datetime + '/Refrac/Test/'
    file_list_refrac = os.listdir(Refrac_path)
    np.random.seed(seed)
    np.random.shuffle(file_list_refrac)
    train_size = int(len(file_list_refrac) * 0.85)
    test_size = len(file_list_refrac) - train_size
    file_list_Refrac_train = np.array(file_list_refrac[0:train_size])
    file_list_Refrac_test = np.array(file_list_refrac[train_size:len(file_list_refrac)])

    if not os.path.exists(current_path + '/Data/After_Preprocess/' + use_datetime + '/Refrac/'):
        os.makedirs(current_path + '/Data/After_Preprocess/' + use_datetime + '/Refrac/')
    if not os.path.exists(current_path + '/Data/After_Preprocess/' + use_datetime + '/Refrac/Train'):
        os.makedirs(current_path + '/Data/After_Preprocess/' + use_datetime + '/Refrac/Train')
    if not os.path.exists(current_path + '/Data/After_Preprocess/' + use_datetime + '/Refrac/Test'):
        os.makedirs(current_path + '/Data/After_Preprocess/' + use_datetime + '/Refrac/Test')

    Refrac_path_train = current_path + '/Data/After_Preprocess/' + use_datetime + '/Refrac/Train/'
    Refrac_path_test = current_path + '/Data/After_Preprocess/' + use_datetime + '/Refrac/Test/'

    for idx, file in enumerate(file_list_Refrac_train):
        excel_source = pd.read_csv(Refrac_path + file)
        df = pd.DataFrame(excel_source)
        df.to_csv(Refrac_path_train + file, index=False)

    for idx, file in enumerate(file_list_Refrac_test):
        excel_source = pd.read_csv(Refrac_path + file)
        df = pd.DataFrame(excel_source)
        df.to_csv(Refrac_path_test + file, index=False)

    for i in range(len(file_list_Refrac_train)):
        file_list_Refrac_train[i] = file_list_Refrac_train[i].strip(".csv")
        file_list_Refrac_train[i] = '\'' + file_list_Refrac_train[i] + '\''

    for i in range(len(file_list_Refrac_test)):
        file_list_Refrac_test[i] = file_list_Refrac_test[i].strip(".csv")
        file_list_Refrac_test[i] = '\'' + file_list_Refrac_test[i] + '\''

    colums_lst = ['Train']
    df_ = pd.DataFrame(columns=colums_lst)
    df_['Train'] = file_list_Refrac_train
    df_.to_csv(current_path + '/FileList/filelist_refrac_Train_' + use_datetime + '.csv')

    colums_lst = ['Test']
    df_ = pd.DataFrame(columns=colums_lst)
    df_['Test'] = file_list_Refrac_test
    df_.to_csv(current_path + '/FileList/filelist_refrac_Test_' + use_datetime + '.csv')

    ## 이번엔 Refrac 데이터를 제외한 데이터로 일정 수를 맞춤
    # file_list_except_refrac = list(set(file_list_input) - set(file_list_refrac))
    file_list_except_refrac = [x for x in file_list_input_test if x not in file_list_refrac]

    np.random.shuffle(file_list_except_refrac)
    train_size = int(len(file_list_except_refrac) * 0.85)
    test_size = len(file_list_except_refrac) - train_size
    file_list_trainX = np.array(file_list_except_refrac[0:train_size])
    file_list_testX = np.array(file_list_except_refrac[train_size:len(file_list_except_refrac)])

    for idx, file in enumerate(file_list_trainX):
        excel_source = pd.read_csv(input_path_test + file)
        df = pd.DataFrame(excel_source)
        df.to_csv(train_path + file, index=False)

    for idx, file in enumerate(file_list_testX):
        excel_source = pd.read_csv(input_path_test + file)
        df = pd.DataFrame(excel_source)
        df.to_csv(test_path + file, index=False)

    for i in range(len(file_list_trainX)):
        file_list_trainX[i] = file_list_trainX[i].strip(".csv")
        file_list_trainX[i] = '\'' + file_list_trainX[i] + '\''
        
    for i in range(len(file_list_testX)):
        file_list_testX[i] = file_list_testX[i].strip(".csv")
        file_list_testX[i] = '\'' + file_list_testX[i] + '\''

    colums_lst = ['Train']
    df_ = pd.DataFrame(columns=colums_lst)
    df_['Train'] = file_list_trainX
    df_.to_csv(current_path + '/FileList/filelist_except_refrac_Train_' + use_datetime + '.csv')

    colums_lst = ['Test']
    df_ = pd.DataFrame(columns=colums_lst)
    df_['Test'] = file_list_testX
    df_.to_csv(current_path + '/FileList/filelist_except_refrac_Test_' + use_datetime + '.csv')

# Output 2개일 때
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

def set_data_by_filelist(current_path, use_datetime):
    filelist_path = current_path + '\\FileList\\'
    input_path = current_path + '\\Data\\After_Refrac\\'
    output_path = current_path + '\\Data\\After_Preprocess\\'

    for f in os.listdir(output_path + use_datetime):
        shutil.rmtree(output_path + use_datetime + '/' + f) ## 기존에 있던 파일 지우기

    if not os.path.exists(current_path + '\\Data\\After_Preprocess\\' + use_datetime):
            os.makedirs(current_path + '\\Data\\After_Preprocess\\' + use_datetime)
    if not os.path.exists(current_path + '\\Data\\After_Preprocess\\' + use_datetime + '\\Train'):
            os.makedirs(current_path + '\\Data\\After_Preprocess\\' + use_datetime + '\\Train')
    if not os.path.exists(current_path + '\\Data\\After_Preprocess\\' + use_datetime + '\\Test'):
            os.makedirs(current_path + '\\Data\\After_Preprocess\\' + use_datetime + '\\Test')
    if not os.path.exists(current_path + '\\Data\\After_Preprocess\\' + use_datetime + '\\Refrac'):
            os.makedirs(current_path + '\\Data\\After_Preprocess\\' + use_datetime + '\\Refrac')
    if not os.path.exists(current_path + '\\Data\\After_Preprocess\\' + use_datetime + '\\Refrac\\Train'):
            os.makedirs(current_path + '\\Data\\After_Preprocess\\' + use_datetime + '\\Refrac\\Train')
    if not os.path.exists(current_path + '\\Data\\After_Preprocess\\' + use_datetime + '\\Refrac\\Test'):
            os.makedirs(current_path + '\\Data\\After_Preprocess\\' + use_datetime + '\\Refrac\\Test')

    val_tmp = ['Train', 'Test']

    for val in val_tmp:
        file_name = 'filelist_except_refrac_' + val + '_' + use_datetime + '.csv'
        df_tmp = pd.read_csv(filelist_path + file_name)
        for idx in range(len(df_tmp)):
            name = df_tmp[val][idx].strip("'") + '.csv'
            df = pd.read_csv(input_path + use_datetime + '\\Test\\' + name)
            df.to_csv(output_path + use_datetime + '\\' + val + '\\' + name)

    for val in val_tmp:
        file_name = 'filelist_refrac_' + val + '_' + use_datetime + '.csv'
        df_tmp = pd.read_csv(filelist_path + file_name)
        for idx in range(len(df_tmp)):
            name = df_tmp[val][idx].strip("'") + '.csv'
            df = pd.read_csv(input_path + use_datetime + '\\Refrac\\Test\\' + name)
            df.to_csv(output_path + use_datetime + '\\Refrac\\' + val + '\\' + name)

    # file_name = 'filelist_refrac_' + 'Test_' + use_datetime + '.csv'
    # df_tmp = pd.read_csv(filelist_path + file_name)
    # for idx in range(len(df_tmp)):
    #     name = df_tmp[val][idx].strip("'") + '.csv'
    #     df = pd.read_csv(input_path + use_datetime + '\\Refrac\\Test\\' + name)
    #     df.to_csv(output_path + use_datetime + '\\Refrac\\Test\\' + name)
