import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_ratio, demographic_parity_difference, equalized_odds_ratio, equalized_odds_difference
from aif360.sklearn.metrics import consistency_score, generalized_entropy_error

class IndividualFairnessLoss(nn.Module):
    def __init__(self, fairness_lambda=1, L2_gamma=0):
        super(IndividualFairnessLoss, self).__init__()
        self.fairness_lambda = fairness_lambda
        self.L2_gamma = L2_gamma

    def forward(self, y_pred, y_true, dataset, sensitive_feature, model):
        if self.fairness_lambda > 0:
            fairness_loss = 0
            pos_data, neg_data = dataset[dataset[sensitive_feature] == 1], dataset[dataset[sensitive_feature] == 0]
            pos_target, neg_target = pos_data['y'].reset_index(drop=True), neg_data['y'].reset_index(drop=True)
            pos_data, neg_data = pos_data.drop(['y', sensitive_feature], axis=1), neg_data.drop(['y', sensitive_feature],
                                                                                                axis=1)
            pos_len, neg_len = len(pos_data), len(neg_data)
            pos_data, neg_data = torch.tensor(pos_data.values).to(torch.float), torch.tensor(neg_data.values).to(
                torch.float)
            pos_data_output, neg_data_output = [model(pos_data[i]) for i in range(len(pos_data))], [model(neg_data[i]) for i in range(len(neg_data))]
            selected_pairs = np.random.choice(pos_len * neg_len, 2 * min(pos_len, neg_len), replace=False)
            idx = 0
            for i in range(pos_len):
                for j in range(neg_len):
                    if idx in selected_pairs and pos_target[i] == neg_target[j]:
                        fairness_loss += (pos_data_output[i] - neg_data_output[j]) ** 2
                    idx += 1
            scaled_fairness_loss = self.fairness_lambda * fairness_loss / (2 * min(pos_len, neg_len))
        else:
            scaled_fairness_loss = 0
        L2_penalty = self.L2_gamma * sum([(p**2).sum() for p in model.parameters()])
        return F.binary_cross_entropy(y_pred, y_true) + scaled_fairness_loss + L2_penalty

class GroupFairnessLoss(nn.Module):
    def __init__(self, fairness_lambda=1, L2_gamma=0):
        super(GroupFairnessLoss, self).__init__()
        self.fairness_lambda = fairness_lambda
        self.L2_gamma = L2_gamma

    def forward(self, y_pred, y_true, dataset, sensitive_feature, model):
        if self.fairness_lambda > 0:
            fairness_loss = 0
            pos_data, neg_data = dataset[dataset[sensitive_feature] == 1], dataset[dataset[sensitive_feature] == 0]
            pos_target, neg_target = pos_data['y'].reset_index(drop=True), neg_data['y'].reset_index(drop=True)
            pos_data, neg_data = pos_data.drop(['y', sensitive_feature], axis=1), neg_data.drop(['y', sensitive_feature], axis=1)
            pos_len, neg_len = len(pos_data), len(neg_data)
            pos_data, neg_data = torch.tensor(pos_data.values).to(torch.float), torch.tensor(neg_data.values).to(torch.float)
            pos_data_output, neg_data_output = [model(pos_data[i]) for i in range(len(pos_data))], [model(neg_data[i]) for i in range(len(neg_data))]
            selected_pairs = np.random.choice(pos_len * neg_len, 2 * min(pos_len, neg_len), replace=False)
            idx = 0
            for i in range(pos_len):
                for j in range(neg_len):
                    if idx in selected_pairs and pos_target[i] == neg_target[j]:
                        fairness_loss += (pos_data_output[i] - neg_data_output[j]) ** 2
                    idx += 1
            scaled_fairness_loss = self.fairness_lambda * (fairness_loss / (2 * min(pos_len, neg_len))) ** 2
        else:
            scaled_fairness_loss = 0
        L2_penalty = self.L2_gamma * sum([(p**2).sum() for p in model.parameters()])
        return F.binary_cross_entropy(y_pred, y_true) + scaled_fairness_loss + L2_penalty

class HybridFairnessLoss(nn.Module):
    def __init__(self, fairness_lambda=1, L2_gamma=0):
        super(HybridFairnessLoss, self).__init__()
        self.fairness_lambda = fairness_lambda
        self.L2_gamma = L2_gamma

    def forward(self, y_pred, y_true, dataset, sensitive_feature, model):
        if self.fairness_lambda > 0:
            pos_fairness_loss, neg_fairness_loss = 0, 0
            pos_data, neg_data = dataset[dataset[sensitive_feature] == 1], dataset[dataset[sensitive_feature] == 0]
            pos_data_pos, pos_data_neg, neg_data_pos, neg_data_neg = pos_data[pos_data['y'] == 1], pos_data[pos_data['y'] == 0], neg_data[neg_data['y'] == 1], neg_data[neg_data['y'] == 0]
            n11, n10, n21, n20 = len(pos_data_pos), len(pos_data_neg), len(neg_data_pos), len(neg_data_neg)
            pos_data_pos_data, pos_data_neg_data, neg_data_pos_data, neg_data_neg_data = (
                pos_data_pos.drop(['y', sensitive_feature], axis=1), pos_data_neg.drop(['y', sensitive_feature], axis=1),
                neg_data_pos.drop(['y', sensitive_feature], axis=1), neg_data_neg.drop(['y', sensitive_feature], axis=1))
            pos_data_pos_data, pos_data_neg_data, neg_data_pos_data, neg_data_neg_data = (
                torch.tensor(pos_data_pos_data.values).to(torch.float), torch.tensor(pos_data_neg_data.values).to(torch.float),
                torch.tensor(neg_data_pos_data.values).to(torch.float), torch.tensor(neg_data_neg_data.values).to(torch.float))
            pos_data_pos_output, pos_data_neg_output, neg_data_pos_output, neg_data_neg_output = (
                [model(pos_data_pos_data[i]) for i in range(len(pos_data_pos_data))], [model(pos_data_neg_data[i]) for i in range(len(pos_data_neg_data))],
                [model(neg_data_pos_data[i]) for i in range(len(neg_data_pos_data))], [model(neg_data_neg_data[i]) for i in range(len(neg_data_neg_data))])
            pos_selected_pairs, neg_selected_pairs = (np.random.choice(n11 * n21, 2 * min(n11, n21), replace=False),
                                                      np.random.choice(n10 * n20, 2 * min(n10, n20), replace=False))
            pos_idx, neg_idx = 0, 0
            for i in range(n11):
                for j in range(n21):
                    if pos_idx in pos_selected_pairs:
                        pos_fairness_loss += pos_data_pos_output[i] - neg_data_pos_output[j]
                    pos_idx += 1
            for i in range(n10):
                for j in range(n20):
                    if neg_idx in neg_selected_pairs:
                        neg_fairness_loss += pos_data_neg_output[i] - neg_data_neg_output[j]
                    neg_idx += 1
            fairness_loss = (pos_fairness_loss / (2 * min(n11, n21))) ** 2 + (neg_fairness_loss / (2 * min(n10, n20))) ** 2
        else:
            fairness_loss = 0
        L2_penalty = self.L2_gamma * sum([(p ** 2).sum() for p in model.parameters()])
        return F.binary_cross_entropy(y_pred, y_true) + self.fairness_lambda * fairness_loss + L2_penalty

def fairness_metrics(y_true, y_pred, sensitive_feature, X):
    res_dict = {}
    res_dict['demographic_parity_difference'] = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_feature)
    # res_dict['demographic_parity_ratio'] = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_feature, method='to_overall')
    res_dict['equalized_odds_difference'] = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_feature)
    # res_dict['equalized_odds_ratio'] = equalized_odds_ratio(y_true, y_pred, sensitive_features=sensitive_feature, method='to_overall')
    res_dict['consistency'] = consistency_score(X, y_pred, n_neighbors=5)
    res_dict['generalized_entropy_error'] = generalized_entropy_error(y_true, y_pred)
    return res_dict
