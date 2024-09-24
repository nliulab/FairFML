from multiprocessing import Pool
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fairness_loss import GroupFairnessLoss, fairness_metrics
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error


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

def train_model_with_local_lambda(site, total_clients, fairness_lambda, metric='accuracy', out_file=None):
    print(f"Client {site}, Fairness penalty strength (lambda): {fairness_lambda}", file=out_file)
    train_data = read_adult_data(idx=site, num_clients=total_clients, is_train=True)
    test_data = read_adult_data(idx=site, num_clients=total_clients, is_train=False)
    batch_size = 128
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    model = BinaryLogisticRegression(len(train_data[0][0]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    criterion = GroupFairnessLoss(fairness_lambda=fairness_lambda, L2_gamma=0)
    epochs = 100
    acc_list, auc_list, mse_list, fairness_list = [], [], [], []
    for epoch in range(epochs):
        for i, (x, y, sensitive_feature) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x).reshape(1, -1)[0].type(torch.float32)
            df = pd.DataFrame(x.numpy())
            df['gender'] = sensitive_feature
            df['y'] = y.numpy()
            loss = criterion(output, y.type(torch.float32), dataset=df, sensitive_feature='gender', model=model)
            loss.backward()
            optimizer.step()

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
        acc_list.append(accuracy)
        auc_list.append(auc)
        mse_list.append(mse)
        print('Epoch: {}. Loss: {}. Accuracy: {}. AUROC: {}. MSE:{}'.format(epoch, loss.item(), accuracy, auc, mse), file=out_file)
        fairness = fairness_metrics(y_true, y_pred, all_sensitive_feature, X)
        fairness_list.append(fairness)
        print(fairness, file=out_file)
    for p in model.parameters():
        print(p, file=out_file)
    if metric == 'accuracy':
        idx = max(enumerate(acc_list), key=lambda x: x[1])[0]
        print(f"\nMax Accuracy: {acc_list[idx]}, Iteration: {idx}, AUC: {auc_list[idx]}, MSE: {mse_list[idx]}, "
              f"Fairness metrics: {fairness_list[idx]}\n", file=out_file)
    elif metric == 'mse':
        idx = min(enumerate(mse_list), key=lambda x: x[1])[0]
        print(f"\nMin MSE: {mse_list[idx]}, Iteration: {idx}, AUC: {auc_list[idx]}, Accuracy: {acc_list[idx]}, "
              f"Fairness metrics: {fairness_list[idx]}\n", file=out_file)
    else:
        raise ValueError("Metric must be either 'accuracy' or 'mse'")
    return acc_list[idx], auc_list[idx], mse_list[idx], fairness_list[idx]

def find_best_local_lambda(site, total_clients, metric_diff=0.01, metric='accuracy', out_file=None):
    result_lst = []
    base_acc, base_auc, base_mse, base_fairness = train_model_with_local_lambda(site, total_clients=total_clients, fairness_lambda=0, metric=metric, out_file=out_file)
    result_lst.append([site, 0, base_acc, base_auc, base_mse, base_fairness['demographic_parity_difference'],
                       base_fairness['equalized_odds_difference'], base_fairness['consistency'], base_fairness['generalized_entropy_error']])
    for fairness_lambda in np.linspace(5, 100, 20):
        acc, auc, mse, fairness = train_model_with_local_lambda(site, total_clients=total_clients, fairness_lambda=fairness_lambda, out_file=out_file)
        result_lst.append([site, fairness_lambda, acc, auc, mse, fairness['demographic_parity_difference'],
                           fairness['equalized_odds_difference'], fairness['consistency'], fairness['generalized_entropy_error']])
        if (metric == 'accuracy' and acc <= (1 - metric_diff) * base_acc) or (metric == 'mse' and mse >= (1 + metric_diff) * base_mse):
            df = pd.DataFrame.from_records(result_lst, columns=['site', 'lambda', 'acc', 'auc', 'mse', 'dpd', 'eod', 'consistency', 'generalized_entropy_error'])
            return df
    df = pd.DataFrame.from_records(result_lst, columns=['site', 'lambda', 'acc', 'auc', 'mse', 'dpd', 'eod', 'consistency', 'generalized_entropy_error'])
    return df

def find_lambda_for_one_site(site, total_clients):
    filename = f'outputs/adult/local/local_lambda_site{site}.txt'
    csv_name = f'outputs/adult/local/local_lambda_site{site}.csv'
    with open(filename, 'w') as f:
        df = find_best_local_lambda(site=site, total_clients=total_clients, out_file=f)
        print(df, file=f)
        df.to_csv(csv_name, index=False)


if __name__ == '__main__':
    # Find local lambda values for each site
    args = [(x, 5) for x in range(5)]
    with Pool(5) as pool:
        pool.starmap(find_lambda_for_one_site, args)
