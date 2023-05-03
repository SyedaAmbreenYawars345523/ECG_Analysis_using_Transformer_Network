### Ground truth and ecg extraction lead 1

import numpy as np
from natsort import natsorted
import os
import torch
from pandas import *

def get_train_val_data(directory, dir2, target_dir):
    lead1 = list()
    limit = 13000
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
    #vent_rt = data['VentRate'].tolist()
    QT_interval = data['qt'].tolist()
    #QRS_duration = data['qrs'].tolist()
    #R_amplitude = data['R_PeakAmpl_i']
    

    # print(np.shape(vent_rt))
    
    QT_interval = np.array(QT_interval)
    QT_interval = torch.Tensor(QT_interval)
    
    
    #R_amplitude = np.array(R_amplitude)
    #R_amplitude = torch.Tensor(R_amplitude)
    
    #QRS_duration = np.array(QRS_duration)
    #QRS_duration= torch.Tensor(QRS_duration)
    
    #vent_rt = torch.Tensor(vent_rt)
    #vent_rt = np.array(vent_rt)
    
    
    
    print(QT_interval.shape)



    return lead1, QT_interval.view(QT_interval.shape[0], 1)

