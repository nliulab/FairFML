import os
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from extract_output import extract_FL_result, find_best_gamma_value

class BinaryLogisticRegression(nn.Module):
    # build the constructor
    def __init__(self, n_inputs):
        super(BinaryLogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, 1)

    # make predictions
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

def tune_FL_model_gamma_step1(lambda_value, strategy, total_clients=5, metric='accuracy'):
    gamma_length_1 = 3
    if not os.path.exists(f'outputs/adult/FL/{strategy}/lambda_{lambda_value}'):
        os.mkdir(f'outputs/adult/FL/{strategy}/lambda_{lambda_value}')
    gamma_list_1 = np.linspace(1e-4, 0.1, gamma_length_1)
    with Pool(3) as pool:
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
    return result_df

def tune_FL_model_gamma_step2(lambda_value, strategy, best_idx, total_clients=5, metric='accuracy'):
    result_df = pd.read_csv(f'outputs/adult/FL/{strategy}/lambda_{lambda_value}/FL_lambda_gamma_{metric}_result.csv')
    gamma_length_2 = 3
    if best_idx == 0:
        gamma_list_2 = np.linspace(0, result_df['gamma'][1], gamma_length_2)
    elif best_idx == result_df.shape[0] - 1:
        diff = result_df['gamma'][-1] - result_df['gamma'][-2]
        gamma_list_2 = np.linspace(result_df['gamma'][-2], result_df['gamma'][-1] + diff, gamma_length_2)
    else:
        gamma_list_2 = np.linspace(result_df['gamma'][best_idx - 1], result_df['gamma'][best_idx + 1], gamma_length_2)
    with Pool(3) as pool:
        args = [(lambda_value, gamma_value, total_clients) for gamma_value in gamma_list_2]
        if strategy == 'FedAvg':
            pool.starmap(run_FedAvg, args)
        elif strategy == 'PerAvg':
            pool.starmap(run_PerAvg, args)
    extract_FL_result(f'outputs/adult/FL/{strategy}/lambda_{lambda_value}')
    new_result_df = find_best_gamma_value(f'outputs/adult/FL/{strategy}/lambda_{lambda_value}', metric=metric)
    new_result_df = new_result_df.loc[(new_result_df['lambda'] == lambda_value)
                              & (new_result_df['gamma'] >= min(gamma_list_2)) & (new_result_df['gamma'] <= max(gamma_list_2))]
    print("After second round:\n", new_result_df)
    return new_result_df

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
    strategy = sys.argv[1]
    lambda_values = [1]
    # Finetune gamma value for selected lambda values (step 1)
    for lambda_value in lambda_values:
        result_df1 = tune_FL_model_gamma_step1(lambda_value, strategy=strategy, total_clients=5, metric='accuracy')
    # Find the best gamma value for each lambda value from step 1 result
    best_idx_dict = {1: 2}
    # Finetune gamma value in a narrow range (step 2)
    for lambda_value in lambda_values:
        result_df2 = tune_FL_model_gamma_step2(lambda_value, strategy=strategy, best_idx=best_idx_dict[lambda_value], total_clients=5, metric='accuracy')
    os.system(f'python utils/save_PFL_client_models.py')
