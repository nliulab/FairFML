import sys
import os
import numpy as np
import pandas as pd

def extract_PFL_result(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not (file.startswith("PFL") and file.endswith(".txt")):
                continue
            lambda_value, gamma_value = file.split("_")[2], file.split("_")[3]
            out_filename = f'result_PFL_group_{lambda_value}_{gamma_value}'
            with open(os.path.join(root, file), 'r') as f_in, open(os.path.join(root, out_filename), 'w') as f_out:
                f_out.write("Experiment setup options:\n\n")
                lines = f_in.readlines()
                for line in lines:
                    if 'Local learning rate:' in line:
                        lr = line.split(':')[-1].strip('\n')
                        f_out.write('Learning rate:' + lr + '\n')
                    elif 'Local batch size' in line:
                        bs = line.split(':')[-1].strip('\n')
                        f_out.write('Batch size:' + bs + '\n')
                    elif 'Total number of clients' in line:
                        nc = line.split(':')[-1].strip('\n')
                        f_out.write('Number of clients:' + nc + '\n')
                    elif 'Dataset' in line or 'Global rounds' in line:
                        f_out.write(line)
                    elif 'fairness' in line.lower():
                        f_out.write(line)
                    elif 'Round number' in line:
                        round_num = line.strip('-\n').split(':')[-1].strip('-\n')
                        f_out.write('\nRound' + round_num + ':\n')
                    elif 'before SGD' in line:
                        f_out.write('Before personalized one-step SGD:\n')
                    elif 'Evaluate global model with one step update' in line:
                        f_out.write('After personalized one-step SGD:\n')
                    elif 'Mean test AUC' in line:
                        auc = line.strip('-').split(':')[-1]
                        f_out.write('Mean test AUC:' + auc)
                    elif 'Mean test accuracy' in line:
                        accuracy = line.strip('-').split(':')[-1]
                        f_out.write('Mean test accuracy:' + accuracy)
                    elif 'Mean MSE' in line:
                        mse = line.strip('-').split(':')[-1]
                        f_out.write('Mean MSE:' + mse)
                    elif 'Mean demographic parity difference' in line:
                        dpd = line.strip('-').split(':')[-1]
                        f_out.write('Mean demographic parity difference: ' + dpd)
                    elif 'Mean equalized odds difference' in line:
                        eod = line.strip('-').split(':')[-1]
                        f_out.write('Mean equalized odds difference:' + eod)
                    elif 'Mean consistency' in line:
                        consistency = line.strip('-').split(':')[-1]
                        f_out.write('Mean consistency:' + consistency)
                    elif 'Mean generalized entropy error' in line:
                        gee = line.strip('-').split(':')[-1]
                        f_out.write('Mean generalized entropy error:' + gee)

def find_best_gamma_value(directory, metric='accuracy'):
    result_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not (file.startswith("result_PFL") and file.endswith(".txt")):
                continue
            filename_split = file.split("_")
            lambda_value, gamma_value = float(filename_split[3][6:]), float(filename_split[4][5:-4])
            with open(os.path.join(root, file), 'r') as f:
                lines = f.readlines()
                if metric == 'accuracy':
                    max_accuracy = 0
                    for i in range(len(lines)):
                        if 'Mean test accuracy' in lines[i]:
                            accuracy = float(lines[i].strip('\n').split(':')[-1].strip('-\n'))
                            if accuracy > max_accuracy:
                                max_accuracy = accuracy
                                AUC = float(lines[i + 1].strip('\n').split(':')[-1].strip('-\n'))
                                MSE = float(lines[i + 2].strip('\n').split(':')[-1].strip('-\n'))
                                DPD = float(lines[i + 3].strip('\n').split(':')[-1].strip('-\n'))
                                EOD = float(lines[i + 4].strip('\n').split(':')[-1].strip('-\n'))
                                consistency = float(lines[i + 5].strip('\n').split(':')[-1].strip('-\n'))
                                gee = float(lines[i + 6].strip('\n').split(':')[-1].strip('-\n'))
                    # result_list.append([lambda_value, gamma_value, max_accuracy, AUC, MSE, DPD, EOD])
                    result_list.append([lambda_value, gamma_value, max_accuracy, AUC, MSE, DPD, EOD, consistency, gee])
                elif metric == 'mse':
                    min_mse = np.inf
                    for i in range(len(lines)):
                        if 'Mean MSE' in lines[i]:
                            mse = float(lines[i].strip('\n').split(':')[-1].strip('-\n'))
                            if mse < min_mse:
                                min_mse = mse
                                accuracy = float(lines[i - 2].strip('\n').split(':')[-1].strip('-\n'))
                                AUC = float(lines[i - 1].strip('\n').split(':')[-1].strip('-\n'))
                                DPD = float(lines[i + 1].strip('\n').split(':')[-1].strip('-\n'))
                                EOD = float(lines[i + 2].strip('\n').split(':')[-1].strip('-\n'))
                                consistency = float(lines[i + 3].strip('\n').split(':')[-1].strip('-\n'))
                                gee = float(lines[i + 4].strip('\n').split(':')[-1].strip('-\n'))
                    # result_list.append([lambda_value, gamma_value, max_accuracy, AUC, MSE, DPD, EOD])
                    result_list.append([lambda_value, gamma_value, min_mse, accuracy, AUC, DPD, EOD, consistency, gee])
    if metric == 'accuracy':
        # result_df = pd.DataFrame.from_records(result_list, columns=['lambda', 'gamma', 'accuracy', 'AUC', 'MSE', 'DPD', 'EOD'])
        result_df = pd.DataFrame.from_records(result_list, columns=['lambda', 'gamma', 'accuracy', 'AUC', 'MSE', 'DPD', 'EOD', 'consistency', 'generalized_entropy_error'])
    elif metric == 'mse':
        # result_df = pd.DataFrame.from_records(result_list, columns=['lambda', 'gamma', 'accuracy', 'AUC', 'MSE', 'DPD', 'EOD'])
        result_df = pd.DataFrame.from_records(result_list, columns=['lambda', 'gamma', 'MSE', 'accuracy', 'AUC', 'DPD', 'EOD', 'consistency', 'generalized_entropy_error'])
    else:
        raise ValueError(f'Metric must be either accuracy or mse')
    result_df = result_df.sort_values(by=['lambda', 'gamma']).reset_index(drop=True)
    result_df.to_csv(os.path.join(directory, 'FL_lambda_gamma_' + metric + '_result.csv'), index=False)
    return result_df

def extract_FedAvg_result(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not (file.startswith("FL") and file.endswith(".txt")):
                continue
            lambda_value, gamma_value = file.split("_")[2], file.split("_")[3]
            out_filename = f'result_FL_group_{lambda_value}_{gamma_value}'
            with open(os.path.join(root, file), 'r') as f_in, open(os.path.join(root, out_filename), 'w') as f_out:
                f_out.write("Experiment setup options:\n\n")
                lines = f_in.readlines()
                for line in lines:
                    if 'Local learning rate:' in line:
                        lr = line.split(':')[-1].strip('\n')
                        f_out.write('Learning rate:' + lr + '\n')
                    elif 'Local batch size' in line:
                        bs = line.split(':')[-1].strip('\n')
                        f_out.write('Batch size:' + bs + '\n')
                    elif 'Total number of clients' in line:
                        nc = line.split(':')[-1].strip('\n')
                        f_out.write('Number of clients:' + nc + '\n')
                    elif 'Dataset' in line or 'Global rounds' in line:
                        f_out.write(line)
                    elif 'fairness' in line.lower():
                        f_out.write(line)
                    elif 'Round number' in line:
                        round_num = line.strip('-\n').split(':')[-1].strip('-\n')
                        f_out.write('\nRound' + round_num + ':\n')
                    elif 'Mean test AUC' in line:
                        auc = line.strip('-').split(':')[-1]
                        f_out.write('Mean test AUC:' + auc)
                    elif 'Mean test accuracy' in line:
                        accuracy = line.strip('-').split(':')[-1]
                        f_out.write('Mean test accuracy:' + accuracy)
                    elif 'Mean MSE' in line:
                        mse = line.strip('-').split(':')[-1]
                        f_out.write('Mean MSE:' + mse)
                    elif 'Mean demographic parity difference' in line:
                        dpd = line.strip('-').split(':')[-1]
                        f_out.write('Mean demographic parity difference: ' + dpd)
                    elif 'Mean equalized odds difference' in line:
                        eod = line.strip('-').split(':')[-1]
                        f_out.write('Mean equalized odds difference:' + eod)
                    elif 'Mean consistency' in line:
                        consistency = line.strip('-').split(':')[-1]
                        f_out.write('Mean consistency:' + consistency)
                    elif 'Mean generalized entropy error' in line:
                        gee = line.strip('-').split(':')[-1]
                        f_out.write('Mean generalized entropy error:' + gee)

def find_best_gamma_value_FedAvg(directory, metric='accuracy'):
    result_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not (file.startswith("result_FL") and file.endswith(".txt")):
                continue
            filename_split = file.split("_")
            lambda_value, gamma_value = float(filename_split[3][6:]), float(filename_split[4][5:-4])
            with open(os.path.join(root, file), 'r') as f:
                lines = f.readlines()
                if metric == 'accuracy':
                    max_accuracy = 0
                    for i in range(len(lines)):
                        if 'Mean test accuracy' in lines[i]:
                            accuracy = float(lines[i].strip('\n').split(':')[-1].strip('-\n'))
                            if accuracy > max_accuracy:
                                max_accuracy = accuracy
                                AUC = float(lines[i + 1].strip('\n').split(':')[-1].strip('-\n'))
                                MSE = float(lines[i + 2].strip('\n').split(':')[-1].strip('-\n'))
                                DPD = float(lines[i + 3].strip('\n').split(':')[-1].strip('-\n'))
                                EOD = float(lines[i + 4].strip('\n').split(':')[-1].strip('-\n'))
                                consistency = float(lines[i + 5].strip('\n').split(':')[-1].strip('-\n'))
                                gee = float(lines[i + 6].strip('\n').split(':')[-1].strip('-\n'))
                    # result_list.append([lambda_value, gamma_value, max_accuracy, AUC, MSE, DPD, EOD])
                    result_list.append([lambda_value, gamma_value, max_accuracy, AUC, MSE, DPD, EOD, consistency, gee])
                elif metric == 'mse':
                    min_mse = np.inf
                    for i in range(len(lines)):
                        if 'Mean MSE' in lines[i]:
                            mse = float(lines[i].strip('\n').split(':')[-1].strip('-\n'))
                            if mse < min_mse:
                                min_mse = mse
                                accuracy = float(lines[i - 2].strip('\n').split(':')[-1].strip('-\n'))
                                AUC = float(lines[i - 1].strip('\n').split(':')[-1].strip('-\n'))
                                DPD = float(lines[i + 1].strip('\n').split(':')[-1].strip('-\n'))
                                EOD = float(lines[i + 2].strip('\n').split(':')[-1].strip('-\n'))
                                consistency = float(lines[i + 3].strip('\n').split(':')[-1].strip('-\n'))
                                gee = float(lines[i + 4].strip('\n').split(':')[-1].strip('-\n'))
                    # result_list.append([lambda_value, gamma_value, max_accuracy, AUC, MSE, DPD, EOD])
                    result_list.append([lambda_value, gamma_value, min_mse, accuracy, AUC, DPD, EOD, consistency, gee])
    if metric == 'accuracy':
        # result_df = pd.DataFrame.from_records(result_list, columns=['lambda', 'gamma', 'accuracy', 'AUC', 'MSE', 'DPD', 'EOD'])
        result_df = pd.DataFrame.from_records(result_list, columns=['lambda', 'gamma', 'accuracy', 'AUC', 'MSE', 'DPD', 'EOD', 'consistency', 'generalized_entropy_error'])
    elif metric == 'mse':
        # result_df = pd.DataFrame.from_records(result_list, columns=['lambda', 'gamma', 'accuracy', 'AUC', 'MSE', 'DPD', 'EOD'])
        result_df = pd.DataFrame.from_records(result_list, columns=['lambda', 'gamma', 'MSE', 'accuracy', 'AUC', 'DPD', 'EOD', 'consistency', 'generalized_entropy_error'])
    else:
        raise ValueError(f'Metric must be either accuracy or mse')
    result_df = result_df.sort_values(by=['lambda', 'gamma']).reset_index(drop=True)
    result_df.to_csv(os.path.join(directory, 'FedAvg_lambda_gamma_' + metric + '_result.csv'), index=False)
    return result_df


if __name__ == '__main__':
    directory = sys.argv[1]
    extract_FedAvg_result(directory)
    find_best_gamma_value_FedAvg(directory, metric='accuracy')
    find_best_gamma_value_FedAvg(directory, metric='mse')
