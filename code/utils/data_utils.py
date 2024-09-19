import numpy as np
import os
import torch
import pandas as pd


def read_data(datapath,dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join(datapath, dataset, 'train/')
        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()
        return train_data
    else:
        test_data_dir = os.path.join(datapath,dataset, 'test/')
        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()
        return test_data

def std_data(array):
    array_r = array.astype(float)
    for col_idx in range(array.shape[1]):
        col = array[:, col_idx]  
        if len(set(col)) != 2:
            col = (col - np.mean(col)) / np.std(col) # standardization
        array_r[:, col_idx] = col
    return array


def read_data_tabular(datapath, dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join(datapath, dataset, 'train/')
        train_file = train_data_dir + str(idx) + '_train.csv'
        train_tmp = pd.read_csv(train_file, index_col=0)
        X = train_tmp.drop(columns='label').to_numpy()
        X = std_data(X)
        y = train_tmp['label'].to_numpy()
        train_data = {'x':X, 'y':y}
        return train_data
    else:
        test_data_dir = os.path.join(datapath,dataset, 'test/')
        test_file = test_data_dir + str(idx) + '_test.csv'
        test_tmp=pd.read_csv(test_file, index_col=0)
        X = test_tmp.drop(columns='label').to_numpy()
        X = std_data(X)
        y = test_tmp['label'].to_numpy()
        test_data = {'x':X, 'y':y}
        return test_data
    
def read_client_data_tabular(datapath, dataset, idx, is_train=True):
    if is_train:
        train_data = read_data_tabular(datapath, dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data_tabular(datapath, dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

def read_adult_data(idx, num_clients, is_train=True):
    col_names = ['education_11th', 'education_12th', 'education_1st-4th',
       'education_5th-6th', 'education_7th-8th', 'education_9th',
       'education_Assoc-acdm', 'education_Assoc-voc', 'education_Bachelors',
       'education_Doctorate', 'education_HS-grad', 'education_Masters',
       'education_Preschool', 'education_Prof-school',
       'education_Some-college', 'marital.status_Married-AF-spouse',
       'marital.status_Married-civ-spouse',
       'marital.status_Married-spouse-absent', 'marital.status_Never-married',
       'marital.status_Separated', 'marital.status_Widowed',
       'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White',
       'workclass_Local-gov', 'workclass_Private', 'workclass_Self-emp-inc',
       'workclass_Self-emp-not-inc', 'workclass_State-gov',
       'workclass_Without-pay']
    if is_train:
        train_data_dir = f'../../data/adult/C{num_clients}_new/'
        train_file = train_data_dir + 'train_S' + str(idx + 1) + '.csv'
        train_tmp = pd.read_csv(train_file)
        X = train_tmp[['workclass', 'education', 'marital.status', 'race']]
        X = pd.get_dummies(X, dtype='int', drop_first=True)

        feature_diff = set(col_names) - set(X.columns)
        if feature_diff:
            dummy_df = pd.DataFrame(data=np.zeros((X.shape[0], len(feature_diff))), columns=list(feature_diff), index=X.index)
            X = X.join(dummy_df)
        X = X.sort_index(axis=1)
        y = train_tmp['income.new'].to_numpy()
        train_data = {'x': X, 'y': y, 'sensitive_feature': train_tmp['gender'].map({'Male': 1, 'Female': 0})}
        X_train = torch.Tensor(np.array(train_data['x'])).type(torch.float32)
        y_train = torch.Tensor(np.array(train_data['y'])).type(torch.int64)
        sensitive_feature = torch.Tensor(np.array(train_data['sensitive_feature'])).type(torch.float32)
        train_data = [(x, y, z) for x, y, z in zip(X_train, y_train, sensitive_feature)]
        return train_data
    else:
        test_data_dir = f'../../data/adult/C{num_clients}_new/'
        test_file = test_data_dir + 'val_S' + str(idx + 1) + '.csv'
        test_tmp = pd.read_csv(test_file)
        X = test_tmp[['workclass', 'education', 'marital.status', 'race']]
        X = pd.get_dummies(X, dtype='int', drop_first=True)

        feature_diff = set(col_names) - set(X.columns)
        if feature_diff:
            dummy_df = pd.DataFrame(data=np.zeros((X.shape[0], len(feature_diff))), columns=list(feature_diff), index=X.index)
            X = X.join(dummy_df)
        X = X.sort_index(axis=1)
        y = test_tmp['income.new'].to_numpy()
        test_data = {'x': X, 'y': y, 'sensitive_feature': test_tmp['gender'].map({'Male': 1, 'Female': 0})}
        X_test = torch.Tensor(np.array(test_data['x'])).type(torch.float32)
        y_test = torch.Tensor(np.array(test_data['y'])).type(torch.int64)
        sensitive_feature = torch.Tensor(np.array(test_data['sensitive_feature'])).type(torch.float32)
        test_data = [(x, y, z) for x, y, z in zip(X_test, y_test, sensitive_feature)]
        return test_data

def read_US_data(idx, is_train=True, setting='hete1'):
    data_dir = f'~/Aug22/US_{setting}/'
    if is_train:
        train_file = data_dir + f'C{idx + 1}_train.csv'
        train_tmp = pd.read_csv(train_file, index_col=0)
        X = train_tmp[["AGE","cause_cardiac","witnessed","init_rhythm","BCPR","resp_time","PHDEPIN"]]
        X = pd.get_dummies(X, dtype='int', drop_first=True)
        X = X.sort_index(axis=1)
        y = train_tmp['outcome_neurological'].to_numpy()
        train_data = {'x': X, 'y': y, 'sensitive_feature': train_tmp['SEX'].map({'Male': 1, 'Female': 0})}
        X_train = torch.Tensor(np.array(train_data['x'])).type(torch.float32)
        y_train = torch.Tensor(np.array(train_data['y'])).type(torch.int64)
        sensitive_feature = torch.Tensor(np.array(train_data['sensitive_feature'])).type(torch.float32)
        train_data = [(x, y, z) for x, y, z in zip(X_train, y_train, sensitive_feature)]
        return train_data
    else:
        test_file = data_dir + f'C{idx + 1}_test.csv'
        test_tmp = pd.read_csv(test_file, index_col=0)
        X = test_tmp[["AGE","cause_cardiac","witnessed","init_rhythm","BCPR","resp_time","PHDEPIN"]]
        X = pd.get_dummies(X, dtype='int', drop_first=True)
        X = X.sort_index(axis=1)
        y = test_tmp['outcome_neurological'].to_numpy()
        test_data = {'x': X, 'y': y, 'sensitive_feature': test_tmp['SEX'].map({'Male': 1, 'Female': 0})}
        X_test = torch.Tensor(np.array(test_data['x'])).type(torch.float32)
        y_test = torch.Tensor(np.array(test_data['y'])).type(torch.int64)
        sensitive_feature = torch.Tensor(np.array(test_data['sensitive_feature'])).type(torch.float32)
        test_data = [(x, y, z) for x, y, z in zip(X_test, y_test, sensitive_feature)]
        return test_data

def read_USSG_data(idx, is_train=True):
    data_dir = f'~/Aug22/USSG/'
    if is_train:
        if idx == 0:
            train_file = data_dir + f'US_train.csv'
        elif idx == 1:
            train_file = data_dir + f'SG_train.csv'
        train_tmp = pd.read_csv(train_file, index_col=0)
        X = train_tmp[["AGE", "cause_cardiac", "witnessed", "init_rhythm", "BCPR", "resp_time", "PHDEPIN"]]
        X = pd.get_dummies(X, dtype='int', drop_first=True)
        X = X.sort_index(axis=1)
        y = train_tmp['outcome_neurological'].to_numpy()
        train_data = {'x': X, 'y': y, 'sensitive_feature': train_tmp['SEX'].map({'Male': 1, 'Female': 0})}
        X_train = torch.Tensor(np.array(train_data['x'])).type(torch.float32)
        y_train = torch.Tensor(np.array(train_data['y'])).type(torch.int64)
        sensitive_feature = torch.Tensor(np.array(train_data['sensitive_feature'])).type(torch.float32)
        train_data = [(x, y, z) for x, y, z in zip(X_train, y_train, sensitive_feature)]
        return train_data
    else:
        if idx == 0:
            test_file = data_dir + f'US_test.csv'
        elif idx == 1:
            test_file = data_dir + f'SG_test.csv'
        test_tmp = pd.read_csv(test_file, index_col=0)
        X = test_tmp[["AGE", "cause_cardiac", "witnessed", "init_rhythm", "BCPR", "resp_time", "PHDEPIN"]]
        X = pd.get_dummies(X, dtype='int', drop_first=True)
        X = X.sort_index(axis=1)
        y = test_tmp['outcome_neurological'].to_numpy()
        test_data = {'x': X, 'y': y, 'sensitive_feature': test_tmp['SEX'].map({'Male': 1, 'Female': 0})}
        X_test = torch.Tensor(np.array(test_data['x'])).type(torch.float32)
        y_test = torch.Tensor(np.array(test_data['y'])).type(torch.int64)
        sensitive_feature = torch.Tensor(np.array(test_data['sensitive_feature'])).type(torch.float32)
        test_data = [(x, y, z) for x, y, z in zip(X_test, y_test, sensitive_feature)]
        return test_data

def read_MIMIC_data(idx, num_clients, is_train=True):
    if is_train:
        train_data_dir = f'~/Documents/nBox/white_nonwhite/C{num_clients}/'
        train_file = train_data_dir + 'trains_S' + str(idx + 1) + '.csv'
        train_tmp = pd.read_csv(train_file)
        X = train_tmp.drop(['label', 'Race'], axis=1)
        X['GENDER'] = X['GENDER'].map({'M': 1, 'F': 0})
        X = X.sort_index(axis=1)
        y = train_tmp['label'].to_numpy()
        train_data = {'x': X, 'y': y, 'sensitive_feature': train_tmp['Race']}
        X_train = torch.Tensor(np.array(train_data['x'])).type(torch.float32)
        y_train = torch.Tensor(np.array(train_data['y'])).type(torch.int64)
        sensitive_feature = torch.Tensor(np.array(train_data['sensitive_feature'])).type(torch.float32)
        train_data = [(x, y, z) for x, y, z in zip(X_train, y_train, sensitive_feature)]
        return train_data
    else:
        test_data_dir = f'~/Documents/nBox/white_nonwhite/C{num_clients}/'
        test_file = test_data_dir + 'tests_S' + str(idx + 1) + '.csv'
        test_tmp = pd.read_csv(test_file)
        X = test_tmp.drop(['label', 'Race'], axis=1)
        X['GENDER'] = X['GENDER'].map({'M': 1, 'F': 0})
        X = X.sort_index(axis=1)
        y = test_tmp['label'].to_numpy()
        test_data = {'x': X, 'y': y, 'sensitive_feature': test_tmp['Race']}
        X_test = torch.Tensor(np.array(test_data['x'])).type(torch.float32)
        y_test = torch.Tensor(np.array(test_data['y'])).type(torch.int64)
        sensitive_feature = torch.Tensor(np.array(test_data['sensitive_feature'])).type(torch.float32)
        test_data = [(x, y, z) for x, y, z in zip(X_test, y_test, sensitive_feature)]
        return test_data

def read_client_data(datapath, dataset, idx, is_train=True):
    if dataset[:2] == "ag" or dataset[:2] == "SS":
        return read_client_data_text(datapath,dataset, idx, is_train)
    elif dataset[:2] == "sh":
        return read_client_data_shakespeare(datapath,dataset, idx)
    elif 'Categorical' in dataset:
        return read_client_data_tabular(datapath, dataset, idx)
    elif 'adult' in dataset:
        return read_adult_data(idx, num_clients=5, is_train=is_train)
    elif 'MIMIC' in dataset:
        return read_MIMIC_data(idx, num_clients=5, is_train=is_train)
    elif 'US_homo' in dataset:
        return read_US_data(idx, is_train=is_train, setting="homo")
    elif 'US_hete' in dataset:
        setting = dataset.split('_')[-1]
        return read_US_data(idx, is_train=is_train, setting=setting)
    elif 'USSG' in dataset:
        return read_USSG_data(idx, is_train=is_train)
    
    if is_train:
        train_data = read_data(datapath,dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(datapath,dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_client_data_text(datapath,dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(datapath,dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']
        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)
        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(datapath,dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']
        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_shakespeare(datapath,dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(datapath,dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(datapath,dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data
