##### version 2 re-implement lr computation
##### try to locate the change point index bug

import numpy as np
import pandas as pd
import random as rd
import scipy as sc
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy
from get_beliefs import *   
import statsmodels.api as sm
import math
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

def smooth_lr_avg_lr(index_neg_10, index_pos_10, PE_10, UP_10, LR_10, SEM_10):
    if ((len(index_neg_10) == 0) & (len(index_pos_10)!=0)):
        LR_10[index_pos_10] = UP_10[index_pos_10] / np.mean(PE_10[index_pos_10])
        SEM_10[index_pos_10] = np.std(LR_10[index_pos_10]) / np.sqrt(len(LR_10[index_pos_10]))
        LR_10[index_pos_10] = np.mean(LR_10[index_pos_10])


    if ((len(index_neg_10) != 0) & (len(index_pos_10) == 0)):
        LR_10[index_neg_10] = UP_10[index_neg_10] / np.mean(PE_10[index_neg_10])
        SEM_10[index_neg_10] = np.std(LR_10[index_neg_10]) / np.sqrt(len(LR_10[index_neg_10]))
        LR_10[index_neg_10] = np.mean(LR_10[index_neg_10])
        PE_10[index_neg_10] = -PE_10[index_neg_10]


    if ((len(index_neg_10) != 0) & (len(index_pos_10) != 0)):
        index_all = np.concatenate([index_pos_10, index_neg_10])
        UP_10[index_neg_10] = -UP_10[index_neg_10]
        PE_10[index_neg_10] = -PE_10[index_neg_10]
        LR_10[index_all] = UP_10[index_all] / np.mean(PE_10[index_all])
        SEM_10[index_all] = np.std(LR_10[index_all]) / np.sqrt(len(LR_10[index_all]))
        LR_10[index_all] = np.mean(LR_10[index_all])

def smooth_lr_avg_update(index_neg_10, index_pos_10, PE_10, UP_10, LR_10):
    if ((len(index_neg_10) == 0) & (len(index_pos_10)!=0)):
        if np.mean(UP_10[index_pos_10]) == 0 or np.mean(PE_10[index_pos_10]) == 0:
            LR_10[index_pos_10] = 0
        else:
            LR_10[index_pos_10] = np.mean(UP_10[index_pos_10]) / np.mean(PE_10[index_pos_10])

    if ((len(index_neg_10) != 0) & (len(index_pos_10) == 0)):
        if np.mean(UP_10[index_neg_10]) == 0 or np.mean(PE_10[index_neg_10]) == 0:

            LR_10[index_neg_10] = 0
        else:
            LR_10[index_neg_10] = np.mean(UP_10[index_neg_10]) / np.mean(PE_10[index_neg_10])
        PE_10[index_neg_10] = -PE_10[index_neg_10]


    if ((len(index_neg_10) != 0) & (len(index_pos_10) != 0)):
        index_all = np.concatenate([index_pos_10, index_neg_10])
        # index_all = np.unique(index_all)
        UP_10[index_neg_10] = -UP_10[index_neg_10]
        PE_10[index_neg_10] = -PE_10[index_neg_10]
        if np.mean(UP_10[index_all]) == 0 or np.mean(PE_10[index_all]) == 0:
            LR_10[index_all] = 0
        else:
            LR_10[index_all] = np.mean(UP_10[index_all]) / np.mean(PE_10[index_all])

def get_pos_neg_index(PE_10, i, step):
    index_neg_10 = np.where((PE_10>i) & (PE_10<=i+step))[0]
    if i == -1:
        index_pos_10 = np.where(((PE_10<-i) & (PE_10>-i-step)))[0]
    else:
        index_pos_10 = np.where(((PE_10<-i) & (PE_10>=-i-step)))[0]

    return index_neg_10, index_pos_10

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
standard_dev   = mat['currentStd'][0][0]
block_num      = mat['blkNum'][0][0]
blk_comp_trial = mat['blockCompletedTrials'][0][0]
changepoint    = mat['isChangeTrial'][0][0]


### start loop over all data
nSim = 32

MSE = np.zeros((31, 32))
REC_BETAS = []

npz_file = "adaptive_tolerance_real.npz"
model_data = Load_vecterized_data(npz_file)  # (32,478)
mse_model = []

trial_number = 8

sim_vec = np.linspace(0,10,num=11)

rec_betas_B_10    = np.empty((nSim,trial_number))
rec_betas_B_10[:] = np.NaN

rec_betas_10    = np.empty((nSim,trial_number))
rec_betas_10[:] = np.NaN

rec_betas_H_10    = np.empty((nSim,trial_number))
rec_betas_H_10[:] = np.NaN


rec_betas_B_25    = np.empty((nSim,trial_number))
rec_betas_B_25[:] = np.NaN

rec_betas_25    = np.empty((nSim,trial_number))
rec_betas_25[:] = np.NaN

rec_betas_H_25    = np.empty((nSim,trial_number))
rec_betas_H_25[:] = np.NaN


UP_10 = []
UP_25 = []

PE_10 = []
PE_25 = []


UP_10_H = []
UP_25_H = []

PE_10_H = []
PE_25_H = []


UP_10_B = []
UP_25_B = []

PE_10_B = []
PE_25_B = []


count = 0

for i in range(nSim):    ## participant 17 and 18 adjustment    17 blk1 and 18 blk3 only have 119 trials    so we use 119 trials of blk1 and blk3 across each ppant

    idx   = np.where(subjects == i+1)   #index of the ppant

    pred  = prediction[idx[0]]          #prediction of the ppant
    haz   = hazard[idx[0]]              #hazard rate 
    out   = outcome[idx[0]]             #outcome of the experimental design
    newB  = blk_comp_trial[idx[0]]      #listing of the ongoing trials
    stdv  = standard_dev[idx[0]]
    b_num = block_num[idx[0]]
    cp    = changepoint[idx[0]]

    ## benchmark reduced bayesian model
    newB_index = np.where(newB==1)[0]
    bayesian_pred = []
    for j in range(len(newB_index)):

        beliefs = def_blf()
        if j != 3:
            get_beliefs(out[newB_index[j]:newB_index[j+1]], beliefs, stdv[newB_index[j]][0])

        else:
            get_beliefs(out[newB_index[j]:], beliefs, stdv[newB_index[j]][0])
        bayesian_pred.extend(copy.deepcopy(beliefs.pred[:-1]))
    bayesian_pred = np.array(bayesian_pred)
    ## end calculate
    
    model = model_data[i]
    model_pred = np.array(model)

    model_pred[model_pred>350] = 0
    bayesian_pred[bayesian_pred>350] = 0
    
    newBlock   = (newB < 2).astype(int)
    newB_idx   = np.where(newBlock == 1)[0]
    newB_idx_a = np.where(newBlock == 1)[0] + 1
    newBlock[newB_idx_a,0] = 1

    newB_idx_for_model_pred = np.array([0,119,239,358])

    ### adjust trial need to be deleted
    if i+1 != 17:
        newB_idx = np.insert(newB_idx, 1, 119)
    else:
        pass    
    if i+1 != 18:
        if i+1 == 17:
            newB_idx = np.insert(newB_idx, 3, 358)
        else:
            newB_idx = np.insert(newB_idx, 3, 359)
    else:
        pass

    model_pred  = np.delete(model_pred, newB_idx_for_model_pred, None)
    bayesian_pred = np.delete(bayesian_pred, newB_idx, None)
    haz         = np.delete(haz, newB_idx, None)
    out         = np.delete(out, newB_idx, None)
    newBlock    = np.delete(newBlock, newB_idx, None)
    b_num       = np.delete(b_num, newB_idx, None)
    cp          = np.delete(cp, newB_idx, None)
    stdv        = np.delete(stdv, newB_idx, None)
    pred        = np.delete(pred, newB_idx, None)
    
    up_B       = np.empty(len(out))
    up_B[:]    = np.NaN  
    up_B[0:-1] = bayesian_pred[1:] - bayesian_pred[0:-1]

    up       = np.empty(len(out))
    up[:]    = np.NaN  
    up[0:-1] = model_pred[1:] - model_pred[0:-1]
    
    up_H     = np.empty(len(out))
    up_H[:]    = np.NaN  
    up_H[0:-1] = pred[1:] - pred[0:-1]

    pe_B       = out - bayesian_pred
    xes_B      = np.empty((len(pe_B),2))
    xes_B[:,0] = 1
    xes_B[:,1] = pe_B
    
    pe       = out - model_pred
    xes      = np.empty((len(pe),2))
    xes[:,0] = 1
    xes[:,1] = pe
    
    pe_H       = out - pred
    xes_H      = np.empty((len(pe_H),2))
    xes_H[:,0] = 1
    xes_H[:,1] = pe_H
    
    #filter the nans
    idx_nan = np.where(np.isnan(up))[0]
    up      = np.delete(up, idx_nan, 0)
    xes     = np.delete(xes, idx_nan, 0)
    cp      = np.delete(cp, idx_nan, 0)
    stdv    = np.delete(stdv, idx_nan, 0)
    up_H    = np.delete(up_H, idx_nan, 0)
    xes_H   = np.delete(xes_H, idx_nan, 0)
    up_B    = np.delete(up_B, idx_nan, 0)
    xes_B   = np.delete(xes_B, idx_nan, 0)
    newBlock = np.delete(newBlock, idx_nan, 0)
    
    
    #filter UP due to errors
    idx_up_H = np.where((up_H > 150) | (up_H < -150))[0]
    up      = np.delete(up, idx_up_H, 0)
    xes     = np.delete(xes, idx_up_H, 0)
    cp      = np.delete(cp, idx_up_H, 0)
    stdv    = np.delete(stdv, idx_up_H, 0)
    up_H    = np.delete(up_H, idx_up_H, 0)
    xes_H   = np.delete(xes_H, idx_up_H, 0)
    up_B    = np.delete(up_B, idx_up_H, 0)
    xes_B   = np.delete(xes_B, idx_up_H, 0)
    newBlock   = np.delete(newBlock, idx_up_H, 0)

    index_10 = np.where(stdv==10)[0]
    index_25 = np.where(stdv==25)[0]
    
    ## calculate LEIA model LR
    nb_ind = np.where(newBlock == 1)[0] - 1  # take away cross blk calculation
    index_10 = np.setdiff1d(index_10, nb_ind)
    index_25 = np.setdiff1d(index_25, nb_ind)

    UP_10.extend(up[index_10])
    PE_10.extend(xes[index_10][:,1])

    UP_25.extend(up[index_25])
    PE_25.extend(xes[index_25][:,1])

    UP_10_B.extend(up_B[index_10])
    PE_10_B.extend(xes_B[index_10][:,1])

    UP_25_B.extend(up_B[index_25])
    PE_25_B.extend(xes_B[index_25][:,1])

    UP_10_H.extend(up_H[index_10])
    PE_10_H.extend(xes_H[index_10][:,1])

    UP_25_H.extend(up_H[index_25])
    PE_25_H.extend(xes_H[index_25][:,1])


PE_10 = np.array(PE_10)
UP_10 = np.array(UP_10)

index = np.where((PE_10 <=1) & (PE_10>-1))[0]
print(PE_10[index])
print(UP_10[index])



start = -200
end = 200
step = 1

SEM = 0*np.ones_like(PE_10)
SEM_25 = 0*np.ones_like(PE_25)

SEM_B = 0*np.ones_like(PE_10_B)
SEM_B_25 = 0*np.ones_like(PE_25_B)

SEM_H = 0*np.ones_like(PE_10_H)
SEM_H_25 = 0*np.ones_like(PE_25_H)


LR_10 = np.full_like(PE_10, np.nan)
SEM_10 = np.full_like(PE_10, np.nan)

LR_25 = np.full_like(PE_25, np.nan)
SEM_25 = np.full_like(PE_25, np.nan)

LR_10_B = np.full_like(PE_10_B, np.nan)
SEM_10_B = np.full_like(PE_10_B, np.nan)

LR_25_B = np.full_like(PE_25_B, np.nan)
SEM_25_B = np.full_like(PE_25_B, np.nan)

LR_10_H = np.full_like(PE_10_H, np.nan)
SEM_10_H = np.full_like(PE_10_H, np.nan)

LR_25_H = np.full_like(PE_25_H, np.nan)
SEM_25_H = np.full_like(PE_25_H, np.nan)

for i in range(start, 0, step):
    print(i)
    PE_10 = np.array(PE_10)
    UP_10 = np.array(UP_10)

    PE_25 = np.array(PE_25)
    UP_25 = np.array(UP_25)

    index_neg_10, index_pos_10 = get_pos_neg_index(PE_10, i, step)
    index_neg_25, index_pos_25 = get_pos_neg_index(PE_25, i, step)

    smooth_lr_avg_update(index_neg_10, index_pos_10, PE_10, UP_10, LR_10)
    smooth_lr_avg_update(index_neg_25, index_pos_25, PE_25, UP_25, LR_25)


for i in range(start, 0, step):
    PE_10_B = np.array(PE_10_B)
    UP_10_B = np.array(UP_10_B)

    PE_25_B = np.array(PE_25_B)
    UP_25_B = np.array(UP_25_B)

    index_neg_10_B, index_pos_10_B = get_pos_neg_index(PE_10_B, i, step)
    index_neg_25_B, index_pos_25_B = get_pos_neg_index(PE_25_B, i, step)

    smooth_lr_avg_update(index_neg_10_B, index_pos_10_B, PE_10_B, UP_10_B, LR_10_B)
    smooth_lr_avg_update(index_neg_25_B, index_pos_25_B, PE_25_B, UP_25_B, LR_25_B)


for i in range(start, 0, step):
    PE_10_H = np.array(PE_10_H)
    UP_10_H = np.array(UP_10_H)

    PE_25_H = np.array(PE_25_H)
    UP_25_H = np.array(UP_25_H)

    index_neg_10_H, index_pos_10_H = get_pos_neg_index(PE_10_H, i, step)
    index_neg_25_H, index_pos_25_H = get_pos_neg_index(PE_25_H, i, step)

    smooth_lr_avg_update(index_neg_10_H, index_pos_10_H, PE_10_H, UP_10_H, LR_10_H)
    smooth_lr_avg_update(index_neg_25_H, index_pos_25_H, PE_25_H, UP_25_H, LR_25_H)


plt.figure()
plt.scatter(PE_10, LR_10, color='r', label='LEIA std 10')
plt.scatter(PE_25, LR_25, color='cornflowerblue', label='LEIA std 25')
plt.ylim(0,1.1)
plt.tick_params('x', labelsize=20)
plt.tick_params('y', labelsize=20)
plt.legend(fontsize=15)
# plt.savefig('LR_PR_LEIA.svg', dpi=100, format="svg", bbox_inches='tight')
plt.show()

plt.figure()
plt.scatter(PE_10_B, LR_10_B, color='r', label='Reduced Bayesian std 10')
plt.scatter(PE_25_B, LR_25_B, color='cornflowerblue', label='Reduced Bayesian std 25')
plt.ylim(0,1.1)
plt.tick_params('x', labelsize=20)
# plt.tick_params('y', labelsize=20)
plt.yticks([])
plt.legend(fontsize=15)
# plt.savefig('LR_PR_BAYESIAN.svg', dpi=100, format="svg", bbox_inches='tight')
plt.show()

plt.figure()
plt.scatter(PE_10_H, LR_10_H, color='r', label='Human std 10')
plt.scatter(PE_25_H, LR_25_H, color='cornflowerblue', label='Human std 25')
plt.ylim(0,1.1)
plt.tick_params('x', labelsize=20)
# plt.tick_params('y', labelsize=20)
plt.yticks([])
plt.xticks([0, 50,100,150,200])
plt.legend(fontsize=15)
# plt.savefig('LR_PR_HUMAN.svg', dpi=100, format="svg", bbox_inches='tight')
plt.show()
