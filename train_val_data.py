import numpy as np
from natsort import natsorted
import os
import torch
from pandas import *

def get_train_val_data(directory, dir2, target_dir):
    lead1 = list()
    limit = None
    j = 0
    sorted_files = natsorted(os.listdir(directory))

    for i in sorted_files:
        # print(i)
        ecg_temp = np.loadtxt(dir2 + i)
        lead1.append(ecg_temp[:, 0].T * (1 / 1000))  # lead 1 of ecg dataset
        j = j + 1
        # print(ecg_temp[:,0])
        if j == limit:
            break
    lead1 = np.array(lead1)
    lead1 = torch.Tensor(lead1)
    print(lead1[0])
    print(lead1[2])

    #####################################
    # getting target value of Ventrate:##
    ####################################
    # reading CSV file
    data = read_csv(target_dir, nrows=len(lead1))
    vent_rt = data['VentRate'].tolist()
    # print(np.shape(vent_rt))
    vent_rt = np.array(vent_rt)
    vent_rt = torch.Tensor(vent_rt)
    # print(vent_rt)
    print(vent_rt.shape)




    return lead1, vent_rt.view(vent_rt.shape[0], 1)

