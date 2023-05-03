# ground truth and lead 1 extraction


import numpy as np
from natsort import natsorted
import os
import torch
from pandas import *

def get_train_val_data(directory, dir2, target_dir):
    lead1 = list()
    limit = 5000
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

    tgt = []

    # reading CSV file
    data = read_csv(target_dir, nrows=len(lead1))
    vent_rt = data['VentRate'].tolist()
    #QT_interval = data['qt'].tolist()
    #QRS_duration = data['qrs'].tolist()
    #R_amplitude = data['R_PeakAmpl_i'].tolist()

    for i in range(len(lead1)):
        #final_temp_v = []
        temp_v = []
        temp_v.append(vent_rt[i])
        #final_temp_v.append(temp_v)

        #temp_QT = []
        #temp_QT.append(QT_interval[i])
        #final_temp_v.append(temp_QT)

        #temp_QRS = []
        #temp_QRS.append(QRS_duration[i])
        #final_temp_v.append(temp_QRS)


        #temp_R = []
        #temp_R.append(R_amplitude[i])
        #final_temp_v.append(temp_R)

        #print(final_temp_v)
        tgt.append(temp_v)
    #print(tgt)
    tgt = np.array(tgt)
    tgt = torch.Tensor(tgt)
    print("target shape")
    print(tgt.shape)



    return lead1, tgt.view(tgt.shape[0],1,1)


