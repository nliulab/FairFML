import os
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from extract_output import extract_FL_result, find_best_gamma_value

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
    data_dir = f'../data/adult/C{num_clients}/'
    if is_train:
        train_file = data_dir + 'train_S' + str(idx + 1) + '.csv'
        train_tmp = pd.read_csv(train_file, index_col=0).reset_index(drop=True)
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
        test_file = data_dir + 'tests_S' + str(idx + 1) + '.csv'
        test_tmp = pd.read_csv(test_file, index_col=0).reset_index(drop=True)
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

class BinaryLogisticRegression(nn.Module):
    # build the constructor
    def __init__(self, n_inputs):
        super(BinaryLogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, 1)

    # make predictions
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


def train_federated_model_with_gamma(best_lambda_values, strategy, total_clients=5, metric='accuracy'):
    best_gamma_dict = {}
    gamma_length_1, gamma_length_2 = 10, 10
    for lambda_value in best_lambda_values:
        if not os.path.exists(f'outputs/adult/FL/{strategy}/lambda_{lambda_value}'):
            os.mkdir(f'outputs/adult/FL/{strategy}/lambda_{lambda_value}')
        gamma_list_1 = np.linspace(1e-4, 0.1, gamma_length_1)
        with Pool(10) as pool:
            args = [(lambda_value, gamma_value, total_clients) for gamma_value in gamma_list_1]
            if strategy == 'FedAvg':
                pool.starmap(run_FedAvg, args)
            elif strategy == 'PerAvg':
                pool.starmap(run_PerAvg, args)
        extract_FL_result(f'outputs/adult/FL/{strategy}/lambda_{lambda_value}')
        result_df = find_best_gamma_value(f'outputs/adult/FL/{strategy}/lambda_{lambda_value}', metric=metric)
        result_df = result_df.loc[(result_df['lambda'] == lambda_value) & (result_df['gamma'] >= min(gamma_list_1))
                                  & (result_df['gamma'] <= max(gamma_list_1))]
        print("After first round:\n", result_df)
        if metric == 'accuracy':
            best_idx = result_df['accuracy'].idxmax()
        elif metric == 'mse':
            best_idx = result_df['mse'].idxmin()
        else:
            raise ValueError("Metric must be either 'accuracy' or 'mse'")
        print('Best gamma value:', result_df['gamma'][best_idx])
        if best_idx == 0:
            gamma_list_2 = np.linspace(0, result_df['gamma'][1], gamma_length_2)
        elif best_idx == result_df.shape[0] - 1:
            diff = result_df['gamma'][-1] - result_df['gamma'][-2]
            gamma_list_2 = np.linspace(result_df['gamma'][-2], result_df['gamma'][-1] + diff, gamma_length_2)
        else:
            gamma_list_2 = np.linspace(result_df['gamma'][best_idx - 1], result_df['gamma'][best_idx + 1], gamma_length_2)
        with Pool(10) as pool:
            args = [(lambda_value, gamma_value, total_clients) for gamma_value in gamma_list_2]
            if strategy == 'FedAvg':
                pool.starmap(run_FedAvg, args)
            elif strategy == 'PerAvg':
                pool.starmap(run_PerAvg, args)
        extract_FL_result(f'outputs/adult/FL/{strategy}/lambda_{lambda_value}')
        result_df = find_best_gamma_value(f'outputs/adult/FL/{strategy}/lambda_{lambda_value}', metric=metric)
        result_df = result_df.loc[(result_df['lambda'] == lambda_value)
                                  & (result_df['gamma'] >= min(gamma_list_2)) & (result_df['gamma'] <= max(gamma_list_2))]
        print("After second round:\n", result_df)
        if metric == 'accuracy':
            best_idx = result_df['accuracy'].idxmax()
        elif metric == 'mse':
            best_idx = result_df['mse'].idxmin()
        else:
            raise ValueError("Metric must be either 'accuracy' or 'mse'")
        best_gamma_dict[lambda_value] = result_df.loc[best_idx].to_dict()
    best_gamma_dict = dict(sorted(best_gamma_dict.items()))
    return best_gamma_dict

def run_PerAvg(lambda_value, gamma_value, total_clients):
    if not os.path.exists(f'outputs/adult/FL/PerAvg/lambda_{lambda_value}/FL_group_lambda{lambda_value}_gamma{gamma_value}.txt'):
        cmd = (f'python -u main.py -datp ../data/adult/C{total_clients} -data adult -m lr -algo PerAvg -lr 0.2 '
               f'-gr 10 -nb 2 -nc {total_clients} -fn True -fnl group -flambda {lambda_value} -fgamma {gamma_value} '
               f'| tee outputs/adult/FL/PerAvg/lambda_{lambda_value}/FL_group_lambda{lambda_value}_gamma{gamma_value}.txt')
        print(cmd)
        os.system(cmd)

def run_FedAvg(lambda_value, gamma_value, total_clients):
    if not os.path.exists(f'outputs/adult/FL/FedAvg/lambda_{lambda_value}/FL_group_lambda{lambda_value}_gamma{gamma_value}.txt'):
        cmd = (f'python -u main.py -datp ../data/adult/C{total_clients} -data adult -m lr -algo FedAvg -lr 0.2 '
               f'-gr 10 -nb 2 -nc {total_clients} -fn True -fnl group -flambda {lambda_value} -fgamma {gamma_value} '
               f'| tee outputs/adult/FL/FedAvg/lambda_{lambda_value}/FL_group_lambda{lambda_value}_gamma{gamma_value}.txt')
        print(cmd)
        os.system(cmd)

if __name__ == '__main__':
    # Finetune gamma values for the selected lambda values
    strategy = sys.argv[1]
    lambda_values = [1, 2, 3, 4, 5]
    train_federated_model_with_gamma(lambda_values, strategy=strategy, total_clients=5, metric='accuracy')
    os.system(f'python utils/save_PFL_client_models.py {strategy}')
