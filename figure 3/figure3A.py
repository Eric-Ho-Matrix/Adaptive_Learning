# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 19:17:31 2022

@author: Cris and Qin
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

##### just calculate once for human data
nSim = 32
trial_number = 8

rec_betas_H    = np.empty((nSim,trial_number))
rec_betas_H[:] = np.NaN

TAC = []

for i in range(nSim):    ## participant 17 and 18 adjustment    17 blk1 and 18 blk3 only have 119 trials    so we use 119 trials of blk1 and blk3 across each ppant

    idx   = np.where(subjects == i+1)   #index of the ppant
        
    pred  = prediction[idx[0]]          #prediction of the ppant
    haz   = hazard[idx[0]]              #hazard rate 
    out   = outcome[idx[0]]             #outcome of the experimental design
    newB  = blk_comp_trial[idx[0]]      #listing of the ongoing trials
    stdv  = standard_dev[idx[0]]
    b_num = block_num[idx[0]]
    cp    = changepoint[idx[0]]
    
    newBlock   = (newB < 2).astype(int)
    newB_idx   = np.where(newBlock == 1)[0]
    newB_idx_a = np.where(newBlock == 1)[0] + 1
    newBlock[newB_idx_a,0] = 1


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

    out         = np.delete(out, newB_idx, None)
    # newBlock    = np.delete(newBlock, newB_idx, None)
    cp          = np.delete(cp, newB_idx, None)
    pred        = np.delete(pred, newB_idx, None)
    
    up_H     = np.empty(len(out))
    up_H[:]    = np.NaN  
    up_H[0:-1] = pred[1:] - pred[0:-1]
    
    pe_H       = out - pred
    xes_H      = np.empty((len(pe_H),2))
    xes_H[:,0] = 1
    xes_H[:,1] = pe_H
    
    #filter UP due to errors
    idx_up_H = np.where((up_H > 150) | (up_H < -150))[0]
    cp      = np.delete(cp, idx_up_H, 0)
    up_H    = np.delete(up_H, idx_up_H, 0)
    xes_H   = np.delete(xes_H, idx_up_H, 0)

    idx_nan = np.where(np.isnan(up_H))[0]
    up_H    = np.delete(up_H, idx_nan, 0)
    xes_H   = np.delete(xes_H, idx_nan, 0)
    cp      = np.delete(cp, idx_nan, 0)
    
    tac = np.zeros(len(cp))
    
    for j in range(len(cp)-1):
        
        if cp[j+1] == 1:
            tac[j+1] = 0
        else:
            tac[j+1] = tac[j] + 1
    
    TAC.append(tac)
    
    modelSubCE_H    = np.empty(trial_number)
    modelSubCE_H[:] = np.NaN
    
    for t in range(trial_number):
        
        prob   = np.where(tac == t)[0]
        
        x_H      = xes_H[prob]
        y_H      = np.empty((len(xes_H[prob]),1))
        y_H[:,0] = up_H[prob]

        c_H    = linear_reg(x_H, y_H)
        
        modelSubCE_H[t] = c_H[1,0]
    
    rec_betas_H[i,:] = modelSubCE_H

###### end calculate human data


###### start calculate model data
num_of_inh_th = 31 # search through 30 different threshold for entropy
GAIN = [12]
REC_BETAS = []
for gain_on_mem in GAIN:
    for j in range(num_of_inh_th):
    # for j in [1,2]:
        ### need to read data from 30 dataset
        dir = "grid_"+str(gain_on_mem)+"_bias_inh_"+str(round(float(j*0.2+1),1))
        npz_file = dir+"/Model_simulations_for_LR_dynamics_grid_"+str(gain_on_mem)+"_bias_"+str(round(float(j*0.2+1),1))+".npz"
        model_data = Load_vecterized_data(npz_file)

        nSim = len(model_data)

        trial_number = 8

        rec_betas    = np.empty((nSim,trial_number))
        rec_betas[:] = np.NaN

        TAC = []

        for i in range(nSim):    ## participant 17 and 18 adjustment    17 blk1 and 18 blk3 only have 119 trials    so we use 119 trials of blk1 and blk3 across each ppant

            idx   = np.where(subjects == i+1)   #index of the ppant
                
            pred  = prediction[idx[0]]          #prediction of the ppant
            haz   = hazard[idx[0]]              #hazard rate 
            out   = outcome[idx[0]]             #outcome of the experimental design
            newB  = blk_comp_trial[idx[0]]      #listing of the ongoing trials
            stdv  = standard_dev[idx[0]]
            b_num = block_num[idx[0]]
            cp    = changepoint[idx[0]]
            
            
            model = model_data[i]
            model_pred = np.array(model)
            
            newBlock   = (newB < 2).astype(int)
            newB_idx   = np.where(newBlock == 1)[0]
            newB_idx_a = np.where(newBlock == 1)[0] + 1
            newBlock[newB_idx_a,0] = 1

            newB_idx_for_model_pred = np.array([0,119,239,358])

            ### adjust trial number 
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
            haz         = np.delete(haz, newB_idx, None)
            out         = np.delete(out, newB_idx, None)
            # newBlock    = np.delete(newBlock, newB_idx, None)
            stdv        = np.delete(stdv, newB_idx, None)
            b_num       = np.delete(b_num, newB_idx, None)
            cp          = np.delete(cp, newB_idx, None)
            pred        = np.delete(pred, newB_idx, None)
            
            up       = np.empty(len(out))
            up[:]    = np.NaN  
            up[0:-1] = model_pred[1:] - model_pred[0:-1]
            
            up_H     = np.empty(len(out))
            up_H[:]    = np.NaN  
            up_H[0:-1] = pred[1:] - pred[0:-1]
            
            pe       = out - model_pred
            xes      = np.empty((len(pe),2))
            xes[:,0] = 1
            xes[:,1] = pe
            
            #filter the nans
            idx_nan = np.where(np.isnan(up))[0]
            up      = np.delete(up, idx_nan, 0)
            xes     = np.delete(xes, idx_nan, 0)
            cp      = np.delete(cp, idx_nan, 0)
            
            #filter UP due to errors
            idx_up_H = np.where((up_H > 150) | (up_H < -150))[0]
            up      = np.delete(up, idx_up_H, 0)
            xes     = np.delete(xes, idx_up_H, 0)
            cp      = np.delete(cp, idx_up_H, 0)
            
            tac = np.zeros(len(cp))
            
            for j in range(len(cp)-1):
                
                if cp[j+1] == 1:
                    tac[j+1] = 0
                else:
                    tac[j+1] = tac[j] + 1
            
            TAC.append(tac)
            
            modelSubCE    = np.empty(trial_number)
            modelSubCE[:] = np.NaN
            
            for t in range(trial_number):
                
                prob   = np.where(tac == t)[0]
                x      =  xes[prob]
                y      = np.empty((len(xes[prob]),1))
                y[:,0] = up[prob]
                
                c      = linear_reg(x, y)
                
                modelSubCE[t]   = c[1,0]
            
            rec_betas[i,:]   = modelSubCE
        
        REC_BETAS.append(rec_betas)



    #prepare list for plot
    error    = np.std(rec_betas, axis=0)
    error_H  = np.std(rec_betas_H, axis=0)


    plot_list = []
    for plot in range(trial_number):
        plot_list.append(str(plot+1))

        
    xx = np.arange(0,trial_number,1)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    sm = plt.cm.ScalarMappable(cmap=pl.cm.jet, norm=plt.Normalize(vmin=1, vmax=7))
    sm.set_array([])  # Set an empty array
    for cc,rec_betas in enumerate(REC_BETAS):
        colors = pl.cm.jet(np.linspace(0, 1, len(REC_BETAS)))
        plt.plot(xx,np.mean(rec_betas,axis=0), '--', linewidth = 3 , color = colors[cc])
    plt.xticks( np.arange(trial_number), plot_list, fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylim([0.0, 1])

    # plt.legend(loc='upper right', fontsize = 13)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    clb = fig.colorbar(sm, cax=cax)
    clb.ax.tick_params(labelsize=25)
    clb.set_ticks([1,2,3,4,5,6,7])

    # Manually create legend entries for the box plots
    legend_labels = [ 'Model']
    # Specify custom colors and linestyles for each legend entry
    legend_colors = ['grey']
    legend_linestyles = ['-','--']  # Set the desired linestyle here
    # Create proxy artists with specified colors and linestyles for each legend entry
    legend_entries = [plt.Line2D([0], [0], color=color, linestyle=linestyle, lw=2)
                    for color, linestyle in zip(legend_colors, legend_linestyles)]
    # Add the custom legend with specified colors and linestyles
    plt.legend(legend_entries, legend_labels)

    # plt.savefig('LR_all_data_combined.svg', dpi=300, format='svg',bbox_inches='tight')
    plt.show()
        
