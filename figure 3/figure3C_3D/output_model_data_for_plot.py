import numpy as np
import scipy as sc
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.io
plt.rcParams['font.family'] = 'Times New Roman'


def Load_vecterized_data(npz_file):
    load_data = np.load(npz_file, allow_pickle=True)

    model_data = load_data["arr_0"][0]

    # print(model_data.shape)    # (478,32,1)  

    model_response_ind = np.zeros((model_data.shape[1],model_data.shape[0]))

    for i in range(model_data.shape[1]):
        model_response_ind[i] = model_data[:,i,:].reshape(model_data.shape[0],)
    
    return model_response_ind      ### (32,478)

def regression(x,y):
    # Create a LinearRegression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(x, y)

    # Output the coefficients and intercept
    coefficients = model.coef_

    return coefficients



##### import human data
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
##### end import human data

##### import model data
# import the best threshold for each ppant
best_threshold_file = 'best_threshold_for_mse_lr.npz'
load_best_threshold = np.load(best_threshold_file, allow_pickle=True)
Best_threshold_each_ppant = load_best_threshold["arr_0"]
##### end import model data


## define array
nSim = 32
trial_number = 8
sim_vec = np.linspace(0,10,num=11)
rec_betas    = np.empty((nSim,trial_number))
rec_betas[:] = np.NaN
rec_betas_H    = np.empty((nSim,trial_number))
rec_betas_H[:] = np.NaN


subjNum              = []
currentHazard        = []
currentPrediction    = []
currentOutcome       = []
blockCompletedTrials = []
blockStds            = []
blkNum               = []
isChangeTrial        = []

## start loop
for i in range(nSim):    ## participant 17 and 18 adjustment    17 blk1 and 18 blk3 only have 119 trials    so we use 119 trials of blk1 and blk3 across each ppant

    npz_file = "grid_12_bias_inh_"+ str(round(float(Best_threshold_each_ppant[i]),1)) +\
                "/Model_simulations_for_LR_dynamics_grid_12_bias_" + str(round(float(Best_threshold_each_ppant[i]),1)) +".npz"
    model_data = Load_vecterized_data(npz_file)

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

    newB_idx = []

    newBlock   = (newB < 2).astype(int)


    ### adjust trial need to be deleted
    if i+1 != 17:
        newB_idx.append(119)
    else:
        pass    
    if i+1 != 18:
        if i+1 == 17:
            newB_idx.append(358)
        else:
            newB_idx.append(359)
    else:
        pass

    haz         = np.delete(haz, newB_idx, None)
    out         = np.delete(out, newB_idx, None)
    newBlock    = np.delete(newBlock, newB_idx, None)
    stdv        = np.delete(stdv, newB_idx, None)
    b_num       = np.delete(b_num, newB_idx, None)
    cp          = np.delete(cp, newB_idx, None)
    pred        = np.delete(pred, newB_idx, None)

    subjNum.extend(i+1 for j in range(len(model_pred)))
    currentHazard.extend(haz)
    currentPrediction.extend(model_pred)
    currentOutcome.extend(out)

    index_newb = np.where(newBlock == 1)[0]
    start = 1
    blk = 0
    for j in range(len(model_pred)):
        if all(j != index_newb):
            blockCompletedTrials.append(start)
            start += 1

            blkNum.append(blk)
            isChangeTrial.append(0)
        else:
            start = 1
            blockCompletedTrials.append(start)
            start += 1

            blk += 1
            blkNum.append(blk)

            isChangeTrial.append(1)
    
    blockStds.extend(stdv)


allDataStruct = {
    'subjNum'              : np.array(subjNum).reshape(-1,1)             ,  
    'currentHazard'        : np.array(currentHazard).reshape(-1,1)       ,
    'currentPrediction'    : np.array(currentPrediction).reshape(-1,1)   ,
    'currentOutcome'       : np.array(currentOutcome).reshape(-1,1)      ,
    'blockCompletedTrials' : np.array(blockCompletedTrials).reshape(-1,1),
    'blockStds'            : np.array(blockStds).reshape(-1,1)           ,
    'blkNum'               : np.array(blkNum).reshape(-1,1)              ,
    'isChangeTrial'        : np.array(isChangeTrial).reshape(-1,1)       
}

data = {
    'allDataStruct' : allDataStruct
}

# Specify the filename for the .mat file
file_name = 'Model_data.mat'

# Save the data to the .mat file
scipy.io.savemat(file_name, data)



# also output csv version 
import pandas as pd

allDataStruct = {
    'subjNum'              : np.array(subjNum).reshape(-1,)             ,  
    'currentHazard'        : np.array(currentHazard).reshape(-1,)       ,
    'currentPrediction'    : np.array(currentPrediction).reshape(-1,)   ,
    'currentOutcome'       : np.array(currentOutcome).reshape(-1,)      ,
    'blockCompletedTrials' : np.array(blockCompletedTrials).reshape(-1,),
    'blockStds'            : np.array(blockStds).reshape(-1,)           ,
    'blkNum'               : np.array(blkNum).reshape(-1,)              ,
    'isChangeTrial'        : np.array(isChangeTrial).reshape(-1,)       
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(allDataStruct)

# Define the name of the CSV file
csv_file = 'Model_data.csv'

# Write the DataFrame to a CSV file
df.to_csv(csv_file, index=False)  # Use index=False to exclude row numbers

print(f'Data has been written to {csv_file}')





