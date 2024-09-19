import h5py
import numpy as np
import os


def average_data(datapath="",algorithm="", dataset="", goal="", times=10):
    test_acc = get_all_results_for_one_algo(datapath,algorithm, dataset, goal, times)

    max_accuracy = []
    for i in range(times):
        max_accuracy.append(test_acc[i].max())

    print("std for best accuracy:", np.std(max_accuracy))
    print("mean for best accuracy:", np.mean(max_accuracy))
    print("all accuracy:",test_acc)


def get_all_results_for_one_algo(datapath="", algorithm="", dataset="", goal="", times=10):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(datapath, dataset, file_name, delete=False)))

    return test_acc


def read_data_then_delete(filepath, dataset, file_name, delete=False):
    #file_path = os.path.join('..','results', file_name + '.h5')
    file_path = filepath + dataset + "/results/" + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc