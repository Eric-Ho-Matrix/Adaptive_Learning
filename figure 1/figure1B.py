# # -*- coding: utf-8 -*-
# """
# Created on Sat Sep 17 19:17:31 2022
# verified on 2025-04-18

# @author: Cris and Qin
# """

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

#we define the functions needed for the analysis
def linear_reg(X, Y):
    
    beta = np.dot((np.linalg.inv(np.dot(X.T,X))), np.dot(X.T,Y))

    return beta

def Load_vecterized_data(npz_file):
    load_data = np.load(npz_file, allow_pickle=True)

    model_data = load_data["arr_0"][0]

    # print(model_data.shape)    # (478,32,1)  

    model_response_ind = np.zeros((model_data.shape[1],model_data.shape[0]))

    for i in range(model_data.shape[1]):
        model_response_ind[i] = model_data[:,i,:].reshape(model_data.shape[0],)
    
    return model_response_ind      ### (32,478)



head_mat = sc.io.loadmat('McGuireNassar2014data.mat')
human_data = np.genfromtxt('data2014.csv', delimiter=',')
human_data = np.delete(human_data, (0), axis=0)
# outcomes = human_data[np.where((human_data[:,3] == 1)),2]
means    = human_data[np.where((human_data[:,3] == 1)),0].reshape(-1,1)


mat = head_mat['allDataStruct']

nSubjs         = np.max(mat['subjNum'][0][0])
subjects       = mat['subjNum'][0][0]
hazard         = mat['currentHazard'][0][0]
prediction     = mat['currentPrediction'][0][0]
outcome        = mat['currentOutcome'][0][0]
standard_dev   = mat['blockStds'][0][0]
block_num      = mat['blkNum'][0][0]
blk_comp_trial = mat['blockCompletedTrials'][0][0]
changepoint    = mat['isChangeTrial'][0][0]


npz_file = "Model_simulations_for_LR_dynamics.npz"
model_data = Load_vecterized_data(npz_file)

# nSim = len(model_data)
nSim = 1

trial_number = 8

sim_vec = np.linspace(0,10,num=11)

rec_betas    = np.empty((nSim,trial_number))
rec_betas[:] = np.NaN

rec_betas_H    = np.empty((nSim,trial_number))
rec_betas_H[:] = np.NaN

TAC = []

for i in range(nSim):    ## participant 17 and 18 adjustment    17 blk1 and 18 blk3 only have 119 trials    so we use 119 trials of blk1 and blk3 across each ppant

    # idx   = np.where(subjects == i+1)   #index of the ppant
    idx = np.where(subjects == 1)
        
    pred  = prediction[idx[0]]          #prediction of the ppant
    haz   = hazard[idx[0]]              #hazard rate 
    out   = outcome[idx[0]]             #outcome of the experimental design
    newB  = blk_comp_trial[idx[0]]      #listing of the ongoing trials
    stdv  = standard_dev[idx[0]]
    b_num = block_num[idx[0]]
    cp    = changepoint[idx[0]]
    


#plot subject one and simulation one!
cp    = changepoint[idx[0]]

sub_num = 1
idx  = np.where(subjects == sub_num)
pred = prediction[idx[0]].reshape(-1,1)           #prediction of the ppant
out  = outcome[idx[0]].reshape(-1,1) 
mod_data = model_data[sub_num-1].reshape(-1,1) 
hu_data = np.array(mod_data)

delete_index = np.array([119,359])
out = np.delete(out, delete_index, None)
pred = np.delete(pred, delete_index, None)
cp    = np.delete(cp, delete_index, None)

plt.figure(figsize=(15,5))
for i in range(len(cp)):
    if cp[i] == 1:
        plt.axvline(x=i, color='grey', linestyle='--', linewidth=0.3)
for i in [0, 119, 239, 358]:
    plt.axvline(x=i, color='blue', linestyle='--', linewidth=1.0)
plt.plot(out, 'o', color = "grey", markersize=10, label = 'True hidden state')
plt.plot(pred, '-', color = "red", linewidth = 7, label = 'Participant')
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.legend(loc='upper right', fontsize = 15)
# plt.savefig('LEIA_human_data.svg', dpi=500, format="svg",bbox_inches='tight')
plt.show()
