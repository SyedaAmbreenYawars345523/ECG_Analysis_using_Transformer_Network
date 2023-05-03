import torch
import os
import numpy as np
from pandas import *
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from transformer_encoder_decoder import EcgModel
#from transformer_encoder.py import EcgModel


from positionalencoding import PositionalEncoding
from selfattentionpooling import SelfAttentionPooling

# # parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = 'BEST_checkpoint_ecg100epoch.pth.tar'






# load model:
checkpoint = torch.load(checkpoint,map_location=torch.device('cpu'))

model = checkpoint['model']
model = model.to(device)
model.eval()
criterion = nn.MSELoss().to(device)

#load ecg and vent_rt


##############################
# getting ecgs in data array:
#############################
lead1 = []

j = 0
for i in os.listdir('test'):

    ecg_temp = np.loadtxt("test/" + i)
    lead1.append(ecg_temp[:, 0].T * (1 / 1000))  # lead 1 of ecg dataset


    # print(ecg_temp[:,0])
    # print(np.shape(ecg_temp[:,0]))
# print(numpy.shape(data))
# print(np.shape(data[0]))
# print(numpy.shape(data))
lead1 = np.array(lead1)
lead1 = torch.Tensor(lead1)
# lead1 = lead1.T*(2/1000)
# print(lead1.shape)
# print(data[1][1].shape)
# print(lead1[0].shape)
print(lead1[0])
print(len(lead1))


#####################################
# getting target value of Ventrate:##
####################################
# reading CSV file
df =read_csv("filtered_all_normals_121977_ground_truth.csv", skiprows = lambda x: 0<x<=121927, nrows=121977-121927)
vent_rt_test = df['VentRate'].tolist()
vent_rt_test = np.array(vent_rt_test)
vent_rt_test = torch.Tensor(vent_rt_test)
vent_rt_test = np.array(vent_rt_test)
vent_rt_test = torch.Tensor(vent_rt_test)
print(vent_rt_test)
print(vent_rt_test.shape)


X, y = lead1, vent_rt_test.view(vent_rt_test.shape[0], 1)
print(y.shape)

test_ds = TensorDataset(X, y)
test_loader = DataLoader(test_ds,batch_size=10, shuffle=False, pin_memory=True)
losses = 0

for batch, (X,y) in enumerate(test_loader):
    batch_loss = 0
    pred = model(X)

    print('/n Prediction is : ' + str(pred))
    print( 'target was :' + str(y))
    loss  = criterion(pred, y)
    print('loss is '+str(loss))
    k=+1
    batch_loss = + loss
    losses =+ batch_loss




avg_losses = losses / len(test_loader)
print('average loss : ', str(avg_losses))
