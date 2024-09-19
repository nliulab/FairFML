import torch
from torch import tensor
from collections import OrderedDict
import os
import sys

def save_PFL_client_models(setting):
    if setting == 'US_hete1' or setting == 'US_hete2':
        num_clients = 4
    elif setting == 'US_hete3' or setting == 'US_hete4':
        num_clients = 6
    else:
        raise ValueError('Unrecognized setting {}'.format(setting))
    directory = f'../outputs/{setting}/group/FL_{num_clients}_sites/PerAvg/'
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not (file.startswith('PFL') and file.endswith('.txt')):
                continue
            filename_split = file[:-4].split('_')
            lambda_val, gamma_val = float(filename_split[2][6:]), float(filename_split[3][5:])
            with open(os.path.join(root, file), 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if 'Intermediate coefficient for client' in line:
                        state_dict = OrderedDict()
                        line_split = line.split(' ')
                        client_id, round_num = int(line_split[4].strip()), int(line_split[-1].strip(':\n'))
                        if not ('fc.weight' in lines[i + 2] and 'tensor' in lines[i + 2]):
                            continue
                        model_coef = lines[i + 2].strip('\n')[10:]
                        j = i + 3
                        while not 'fc.bias' in lines[j]:
                            model_coef += lines[j].strip(' \n')
                            j += 1
                        state_dict['linear.weight'] = eval(model_coef)
                        state_dict['linear.bias'] = eval(lines[j + 1].strip('\n')[8:])
                        output_filename = f'../outputs/{setting}/models/group/PerAvg/lambda_{lambda_val}/gamma_{gamma_val}/PerAvg_client{client_id}_{round_num}.pt'
                        torch.save(state_dict, output_filename)


if __name__ == '__main__':
    setting = sys.argv[1]
    save_PFL_client_models(setting)