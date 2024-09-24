import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, mean_squared_error
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference, equalized_odds_ratio
from aif360.sklearn.metrics import consistency_score, generalized_entropy_error
sys.path.append('..')

class BinaryLogisticRegression(nn.Module):
    # build the constructor
    def __init__(self, n_inputs):
        super(BinaryLogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, 1)

    # make predictions
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

def read_adult_data(idx, num_clients):
    col_names = ['education_11th', 'education_12th', 'education_1st-4th',
                 'education_5th-6th', 'education_7th-8th', 'education_9th',
                 'education_Assoc-acdm', 'education_Assoc-voc', 'education_Bachelors',
                 'education_Doctorate', 'education_HS-grad', 'education_Masters',
                 'education_Preschool', 'education_Prof-school', 'education_Some-college',
                 'marital.status_Married-AF-spouse', 'marital.status_Married-civ-spouse',
                 'marital.status_Married-spouse-absent', 'marital.status_Never-married',
                 'marital.status_Separated', 'marital.status_Widowed', 'race_Asian-Pac-Islander',
                 'race_Black', 'race_Other', 'race_White', 'workclass_Local-gov', 'workclass_Private',
                 'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc', 'workclass_State-gov', 'workclass_Without-pay']
    data_dir = f'../../data/adult/C{num_clients}/'
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

def fairness_metrics(y_true, y_pred, sensitive_feature, X):
    res_dict = {}
    res_dict['demographic_parity_difference'] = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_feature)
    res_dict['demographic_parity_ratio'] = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_feature, method='to_overall')
    res_dict['equalized_odds_difference'] = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_feature)
    res_dict['equalized_odds_ratio'] = equalized_odds_ratio(y_true, y_pred, sensitive_features=sensitive_feature, method='to_overall')
    res_dict['consistency'] = consistency_score(X, y_pred, n_neighbors=5)
    res_dict['generalized_entropy_error'] = generalized_entropy_error(y_true, y_pred)
    return res_dict

def evaluate_server_models(strategy):
    result_lst = []
    for site in range(5):
        test_data = read_adult_data(site, num_clients=5)
        test_loader = DataLoader(dataset=test_data, batch_size=1024, shuffle=False)
        model_directory = f'../outputs/adult/models/group/{strategy}'
        for root, dirs, files in os.walk(model_directory):
            for file in files:
                if not file.endswith('.pt') or 'server' not in file:
                    continue
                model_path_split = os.path.join(root, file).split('/')
                lambda_val, gamma_val = float(model_path_split[6].split('_')[-1]), float(model_path_split[7].split('_')[-1])
                round_number = int('_'.join(file.split('.')[0].split('_')[-1]))
                if round_number != 10:
                    continue
                model = torch.load(os.path.join(root, file))
                model.eval()
                y_prob, y_true, y_pred, all_sensitive_feature, X = [], [], [], [], []
                for x, y, sensitive_feature in test_loader:
                    X.extend(x.numpy())
                    output = model(x).reshape(1, -1)[0].type(torch.float32)
                    predicted = output >= 0.5
                    y_prob.append(output.detach())
                    y_pred.append(predicted.detach())
                    y_true.append(y.detach())
                    all_sensitive_feature.append(sensitive_feature.detach())

                y_prob = np.concatenate(y_prob, axis=0)
                y_true = np.concatenate(y_true, axis=0)
                y_pred = np.concatenate(y_pred, axis=0)
                all_sensitive_feature = np.concatenate(all_sensitive_feature, axis=0)

                accuracy = accuracy_score(y_true, y_pred)
                auc = roc_auc_score(y_true, y_prob)
                mse = mean_squared_error(y_true, y_prob)
                fairness = fairness_metrics(y_true, y_pred, all_sensitive_feature, X)
                result_lst.append([site, lambda_val, gamma_val, round_number, accuracy, auc, mse, fairness['demographic_parity_difference'], fairness['demographic_parity_ratio'],
                                   fairness['equalized_odds_difference'], fairness['equalized_odds_ratio'], fairness['consistency'], fairness['generalized_entropy_error']])
    result_df = pd.DataFrame.from_records(result_lst, columns=['site', 'lambda', 'gamma', 'epoch', 'accuracy', 'AUC', 'MSE', 'DPD', 'DPR', 'EOD', 'EOR', 'consistency', 'generalized_entropy_error'])
    result_df = result_df.sort_values(by=['site', 'lambda', 'gamma', 'epoch']).reset_index(drop=True)
    print(result_df)
    result_df.to_csv(f'../outputs/adult/FL/{strategy}/test_results_server_model.csv', index=False)
    split_result(model_type='server', strategy=strategy)

def evaluate_client_models(strategy):
    result_lst = []
    for site in range(5):
        test_data = read_adult_data(site, num_clients=5)
        test_loader = DataLoader(dataset=test_data, batch_size=1024, shuffle=False)
        model_directory = f'../outputs/adult/models/group/{strategy}'
        for root, dirs, files in os.walk(model_directory):
            for file in files:
                if not file.endswith('.pt') or f'client{site}' not in file:
                    continue
                model_path_split = os.path.join(root, file).split('/')
                lambda_val, gamma_val = float(model_path_split[6].split('_')[-1]), float(model_path_split[7].split('_')[-1])
                round_number = int('_'.join(file.split('.')[0].split('_')[-1]))
                if round_number != 10:
                    continue
                model = BinaryLogisticRegression(n_inputs=31)
                model.load_state_dict(torch.load(os.path.join(root, file)))
                model.eval()
                y_prob, y_true, y_pred, all_sensitive_feature, X = [], [], [], [], []
                for x, y, sensitive_feature in test_loader:
                    X.extend(x.numpy())
                    output = model(x).reshape(1, -1)[0].type(torch.float32)
                    predicted = output >= 0.5
                    y_prob.append(output.detach())
                    y_pred.append(predicted.detach())
                    y_true.append(y.detach())
                    all_sensitive_feature.append(sensitive_feature.detach())

                y_prob = np.concatenate(y_prob, axis=0)
                y_true = np.concatenate(y_true, axis=0)
                y_pred = np.concatenate(y_pred, axis=0)
                all_sensitive_feature = np.concatenate(all_sensitive_feature, axis=0)

                accuracy = accuracy_score(y_true, y_pred)
                auc = roc_auc_score(y_true, y_prob)
                mse = mean_squared_error(y_true, y_prob)
                fairness = fairness_metrics(y_true, y_pred, all_sensitive_feature, X)
                result_lst.append([site, lambda_val, gamma_val, round_number, accuracy, auc, mse, fairness['demographic_parity_difference'], fairness['demographic_parity_ratio'],
                                   fairness['equalized_odds_difference'], fairness['equalized_odds_ratio'], fairness['consistency'], fairness['generalized_entropy_error']])
    result_df = pd.DataFrame.from_records(result_lst, columns=['site', 'lambda', 'gamma', 'epoch', 'accuracy', 'AUC', 'MSE', 'DPD', 'DPR', 'EOD', 'EOR', 'consistency', 'generalized_entropy_error'])
    result_df = result_df.sort_values(by=['site', 'lambda', 'gamma', 'epoch']).reset_index(drop=True)
    print(result_df)
    result_df.to_csv(f'../outputs/adult/FL/{strategy}/test_results_client_model.csv', index=False)
    split_result(model_type='client', strategy=strategy)

def split_result(model_type, strategy):
    data = pd.read_csv(f'../outputs/adult/FL/{strategy}/test_results_{model_type}_model.csv')
    for lambda_val in set(data['lambda']):
        data_part = data[data['lambda'] == lambda_val]
        if int(lambda_val) != lambda_val:
            data_part.to_csv(f'../outputs/adult/FL/{strategy}/lambda_{lambda_val}/test_result_lambda{lambda_val}_{model_type}_model.csv', index=False)
        else:
            data_part.to_csv(f'../outputs/adult/FL/{strategy}/lambda_{int(lambda_val)}/test_result_lambda{int(lambda_val)}_{model_type}_model.csv', index=False)


if __name__ == '__main__':
    strategy = sys.argv[1]
    if strategy == 'FedAvg':
        evaluate_server_models(strategy)
    elif strategy == 'PerAvg':
        evaluate_server_models(strategy)
        evaluate_client_models(strategy)

