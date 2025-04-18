import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy import integrate
import copy
import statsmodels.api as sm
plt.rcParams['font.family'] = 'Times New Roman'


# os.chdir('/Users/qinhe/Desktop/localfile/')

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


nSim = 32

for i in range(nSim):    ## participant 17 and 18 adjustment    17 blk1 and 18 blk3 only have 119 trials    so we use 119 trials of blk1 and blk3 across each ppant

    best_threshold_file = 'best_threshold_for_mse_lr.npz'
    load_best_threshold = np.load(best_threshold_file, allow_pickle=True)
    Best_threshold_each_ppant = load_best_threshold["arr_0"]

    subscript = round(float(Best_threshold_each_ppant[i]),1)

    # if i == 0:
    #     subscript = 4.4

    npz_file = "grid_12_bias_inh_{}/Model_simulations_for_LR_dynamics_grid_12_bias_{}.npz".format(subscript, subscript)
    model_data = Load_vecterized_data(npz_file)

    trial_number = 8

    sim_vec = np.linspace(0,10,num=11)

    rec_betas    = np.empty((1,trial_number))
    rec_betas[:] = np.NaN

    rec_betas_H    = np.empty((1,trial_number))
    rec_betas_H[:] = np.NaN

    # TAC = []

    idx   = np.where(subjects == i+1)   #index of the ppant
    # idx = np.where(subjects == 1)
        
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
    
    pe_H       = out - pred
    xes_H      = np.empty((len(pe_H),2))
    xes_H[:,0] = 1
    xes_H[:,1] = pe_H
    
    #filter the nans
    idx_nan = np.where(np.isnan(up))[0]
    up      = np.delete(up, idx_nan, 0)
    xes     = np.delete(xes, idx_nan, 0)
    cp      = np.delete(cp, idx_nan, 0)
    up_H    = np.delete(up_H, idx_nan, 0)
    xes_H   = np.delete(xes_H, idx_nan, 0)
    
    #filter UP due to errors
    idx_up_H = np.where((up_H > 150) | (up_H < -150))[0]
    up      = np.delete(up, idx_up_H, 0)
    xes     = np.delete(xes, idx_up_H, 0)
    cp      = np.delete(cp, idx_up_H, 0)
    up_H    = np.delete(up_H, idx_up_H, 0)
    xes_H   = np.delete(xes_H, idx_up_H, 0)
    
    tac = np.zeros(len(cp))
    
    for j in range(len(cp)-1):
        
        if cp[j+1] == 1:
            tac[j+1] = 0
        else:
            tac[j+1] = tac[j] + 1
    
    # TAC.append(tac)
    
    modelSubCE    = np.empty(trial_number)
    modelSubCE[:] = np.NaN
    
    modelSubCE_H    = np.empty(trial_number)
    modelSubCE_H[:] = np.NaN

    error    = np.empty(trial_number)
    error[:] = np.NaN

    error_H    = np.empty(trial_number)
    error_H[:] = np.NaN
    
    for t in range(trial_number):
        
        prob   = np.where(tac == t)[0]
        x      =  xes[prob]
        y      = np.empty((len(xes[prob]),1))
        y[:,0] = up[prob]
        
        x_H      = xes_H[prob]
        y_H      = np.empty((len(xes_H[prob]),1))
        y_H[:,0] = up_H[prob]
        

        model_ena = sm.OLS(y, x).fit()
        model_H = sm.OLS(y_H, x_H).fit()

        c = model_ena.params[1]
        c_H = model_H.params[1]

        if c<0: c = 0
        if c_H<0: c_H = 0

        modelSubCE[t]   = copy.deepcopy(c)
        modelSubCE_H[t] = copy.deepcopy(c_H)

        error[t]    = model_ena.bse[1]
        error_H[t]  = model_H.bse[1]

    
    rec_betas[0,:]   = modelSubCE
    rec_betas_H[0,:] = modelSubCE_H


    plot_list = []
    for plot in range(trial_number):
        plot_list.append(str(plot+1))

    xx = np.arange(0,trial_number,1)
    plt.figure(figsize=(7,5))
    plt.plot(xx,np.mean(rec_betas,axis=0), '--', linewidth = 5 , color = 'purple', label = 'Model simulation' )
    plt.fill_between(xx, np.mean(rec_betas,axis=0)-error, np.mean(rec_betas,axis=0)+error, alpha=0.5, facecolor = 'purple')
    plt.plot(xx,np.mean(rec_betas_H,axis=0), '--', linewidth = 5 , color = 'blue', label = 'Human data' )
    plt.fill_between(xx, np.mean(rec_betas_H,axis=0)-error_H, np.mean(rec_betas_H,axis=0)+error_H, alpha=0.5, facecolor = 'cyan')
    plt.xticks( np.arange(trial_number), plot_list, fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.ylim([-0.25, 1.2])
    plt.legend(loc='lower right', fontsize = 20)
    # if i == 5 or i == 26:
    #     plt.savefig('single_lr_ppant_{}.svg'.format(i+1), dpi=500, format='svg', bbox_inches='tight')
    plt.show()



for ppp in range(32):
    index = ppp
    sub_num = index + 1
    idx  = np.where(subjects == sub_num)
    pred = prediction[idx[0]].reshape(-1,1)           #prediction of the ppant
    out  = outcome[idx[0]].reshape(-1,1) 

    # reload_model_data
    subscript = round(float(Best_threshold_each_ppant[index]),1)
    npz_file = "grid_12_bias_inh_{}/Model_simulations_for_LR_dynamics_grid_12_bias_{}.npz".format(subscript, subscript)
    model_data = Load_vecterized_data(npz_file)

    mod_data = model_data[index].reshape(-1,1) 
    hu_data = np.array(mod_data)

    cp    = changepoint[idx[0]]

    delete_index = np.array([119,359])
    out = np.delete(out, delete_index, None)
    pred = np.delete(pred, delete_index, None)
    cp = np.delete(cp, delete_index, None)


    plt.figure(figsize=(15,5))
    for i in range(len(cp)):
        if cp[i] == 1:
            plt.axvline(x=i, color='grey', linestyle='--', linewidth=0.3)
    for i in [0, 119, 239, 358]:
        plt.axvline(x=i, color='blue', linestyle='--', linewidth=1.0)
    plt.plot(out, 'o', color = "grey", markersize=10, label = 'True hidden state')
    plt.plot(mod_data, '-', linewidth = 5, color = 'purple', label = 'LEIA model')
    plt.plot(pred, '-', color = "red", linewidth = 5, label = 'Participant')
    plt.yticks(fontsize = 30)
    plt.xticks(fontsize = 30)
    plt.show()