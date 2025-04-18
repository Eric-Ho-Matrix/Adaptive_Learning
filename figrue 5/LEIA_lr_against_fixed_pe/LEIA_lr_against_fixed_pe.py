import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy.optimize import curve_fit
import uncertainties as unc
import copy
plt.rcParams['font.family'] = 'Times New Roman'

# Custom sigmoid function with three parameters (a, b, d)
def custom_sigmoid(x, a, b, d):
    return d / (1 + np.exp(-(a * x + b))) 



for i in range(10):

    # Load data std5
    npz_file_std5 = "std5_LR_against_PE_prediction_{}.npz".format(i*100+100)
    model_data_std5 = np.load(npz_file_std5, allow_pickle=True)["arr_0"]   
    model_data_std5 = np.squeeze(model_data_std5, axis=1) ### (10, 12, 100, 1)

    # Load real data
    real_data_file_std5 = "std5_LR_against_PE_real_outcome_{}.npz".format(i*100+100)
    load_data_std5 = np.load(real_data_file_std5, allow_pickle=True)
    real_outcome_std5 = load_data_std5["arr_0"]     # (10, 100, 12)

    ## concat all array
    if i == 0:
        final_model_data_std5 = copy.deepcopy(model_data_std5)
        final_real_outcome_std5 = copy.deepcopy(real_outcome_std5)
    else:
        final_model_data_std5 = np.concatenate([final_model_data_std5, model_data_std5], axis=2)
        final_real_outcome_std5 = np.concatenate([final_real_outcome_std5, real_outcome_std5], axis=1)

    
    # Load data std10
    npz_file_std10 = "std10_LR_against_PE_prediction_{}.npz".format(i*100+100)
    model_data_std10 = np.load(npz_file_std10, allow_pickle=True)["arr_0"]   
    model_data_std10 = np.squeeze(model_data_std10, axis=1) ### (10, 12, 100, 1)

    # Load real data
    real_data_file_std10 = "std10_LR_against_PE_real_outcome_{}.npz".format(i*100+100)
    load_data_std10 = np.load(real_data_file_std10, allow_pickle=True)
    real_outcome_std10 = load_data_std10["arr_0"]     # (10, 100, 12)

    ## concat all array
    if i == 0:
        final_model_data_std10 = copy.deepcopy(model_data_std10)
        final_real_outcome_std10 = copy.deepcopy(real_outcome_std10)
    else:
        final_model_data_std10 = np.concatenate([final_model_data_std10, model_data_std10], axis=2)
        final_real_outcome_std10 = np.concatenate([final_real_outcome_std10, real_outcome_std10], axis=1)



    # Load data std25
    npz_file_std25 = "std25_LR_against_PE_prediction_{}.npz".format(i*100+100)
    model_data_std25 = np.load(npz_file_std25, allow_pickle=True)["arr_0"]   
    model_data_std25 = np.squeeze(model_data_std25, axis=1) ### (10, 12, 100, 1)

    # Load real data
    real_data_file_std25 = "std25_LR_against_PE_real_outcome_{}.npz".format(i*100+100)
    load_data_std25 = np.load(real_data_file_std25, allow_pickle=True)
    real_outcome_std25 = load_data_std25["arr_0"]     # (10, 100, 12)

    ## concat all array
    if i == 0:
        final_model_data_std25 = copy.deepcopy(model_data_std25)
        final_real_outcome_std25= copy.deepcopy(real_outcome_std25)
    else:
        final_model_data_std25 = np.concatenate([final_model_data_std25, model_data_std25], axis=2)
        final_real_outcome_std25 = np.concatenate([final_real_outcome_std25, real_outcome_std25], axis=1)

nb_ppants = 1000

LR_std5 = np.zeros((nb_ppants,10))
LR_std10 = np.zeros((nb_ppants,10))
LR_std25 = np.zeros((nb_ppants,10))


# calculate lr
for i in range(10):
    # std 5 model
    crt_mod_data_std5 = final_model_data_std5[i]  #(9, 10, 1)
    crt_real_data_std5 = final_real_outcome_std5[i] # (10, 9)

    PE = (crt_real_data_std5[:,10] - crt_mod_data_std5[10].reshape(-1,)).reshape(-1,1)
    model_update_std5 = crt_mod_data_std5[11] - crt_mod_data_std5[10]
    lr = model_update_std5 / PE
    lr[lr<=0] = 0

    LR_std5[:,i] = lr.reshape(-1,)


    # std 10 model
    crt_mod_data_std10 = final_model_data_std10[i]  #(9, 10, 1)
    crt_real_data_std10 = final_real_outcome_std10[i] # (10, 9)

    PE = (crt_real_data_std10[:,10] - crt_mod_data_std10[10].reshape(-1,)).reshape(-1,1)
    model_update_std10 = crt_mod_data_std10[11] - crt_mod_data_std10[10]
    lr = model_update_std10 / PE
    lr[lr<=0] = 0

    LR_std10[:,i] = lr.reshape(-1,)



    # std 25 model
    crt_mod_data_std25 = final_model_data_std25[i]  #(9, 10, 1)
    crt_real_data_std25 = final_real_outcome_std25[i] # (10, 9)

    PE = (crt_real_data_std25[:,10] - crt_mod_data_std25[10].reshape(-1,)).reshape(-1,1)
    model_update_std25 = crt_mod_data_std25[11] - crt_mod_data_std25[10]
    lr = model_update_std25 / PE
    lr[lr<=0] = 0

    LR_std25[:,i] = lr.reshape(-1,)



# Fit the custom sigmoid function to the data
xx = [i * 10+10 for i in range(10)]

a_fit_std5 = np.zeros((nb_ppants, 1))
b_fit_std5 = np.zeros((nb_ppants, 1))
d_fit_std5 = np.zeros((nb_ppants, 1))

a_fit_std10 = np.zeros((nb_ppants, 1))
b_fit_std10 = np.zeros((nb_ppants, 1))
d_fit_std10 = np.zeros((nb_ppants, 1))

a_fit_std25 = np.zeros((nb_ppants, 1))
b_fit_std25 = np.zeros((nb_ppants, 1))
d_fit_std25 = np.zeros((nb_ppants, 1))

for i in range(nb_ppants):

    try:
        params, covariance = curve_fit(custom_sigmoid, xx, LR_std5[i,:])
    except RuntimeError:
        continue
    # Extract the optimized parameters std10
    a_fit_ind, b_fit_ind, d_fit_ind = params
    a_fit_std5[i,:] = a_fit_ind
    b_fit_std5[i,:] = b_fit_ind
    d_fit_std5[i,:] = d_fit_ind

    try:
        params, covariance = curve_fit(custom_sigmoid, xx, LR_std10[i,:])
    except RuntimeError:
        continue
    # Extract the optimized parameters std10
    a_fit_ind, b_fit_ind, d_fit_ind = params
    a_fit_std10[i,:] = a_fit_ind
    b_fit_std10[i,:] = b_fit_ind
    d_fit_std10[i,:] = d_fit_ind


    try:
        params, covariance = curve_fit(custom_sigmoid, xx, LR_std25[i,:])
    except RuntimeError:
        continue
    # Extract the optimized parameters std10
    a_fit_ind, b_fit_ind, d_fit_ind = params
    a_fit_std25[i,:] = a_fit_ind
    b_fit_std25[i,:] = b_fit_ind
    d_fit_std25[i,:] = d_fit_ind


fitted_LR_std5 = np.zeros((nb_ppants, 10)) 
fitted_LR_std10 = np.zeros((nb_ppants, 10)) 
fitted_LR_std25 = np.zeros((nb_ppants, 10))  

x_input = [i*10+10 for i in range(10)]
for i in range(nb_ppants):
    fitted_LR_std5[i,:]  = custom_sigmoid(x_input, a_fit_std5[i], b_fit_std5[i], d_fit_std5[i])
    fitted_LR_std10[i,:] = custom_sigmoid(x_input, a_fit_std10[i], b_fit_std10[i], d_fit_std10[i])
    fitted_LR_std25[i,:] = custom_sigmoid(x_input, a_fit_std25[i], b_fit_std25[i], d_fit_std25[i])

#### end regulation

#### start draw LR pic
LR_mean_std5 = np.mean(fitted_LR_std5, axis=0, keepdims=True)
LR_std_std5 = np.std(fitted_LR_std5, axis=0, keepdims=True)

LR_mean_std10 = np.mean(fitted_LR_std10, axis=0, keepdims=True)
LR_std_std10 = np.std(fitted_LR_std10, axis=0, keepdims=True)

LR_mean_std25 = np.mean(fitted_LR_std25, axis=0, keepdims=True)
LR_std_std25 = np.std(fitted_LR_std25, axis=0, keepdims=True)

xx = [i * 10 + 10 for i in range(0,10)]

lower_std5 = LR_mean_std5-LR_std_std5
higher_std5 = LR_mean_std5+LR_std_std5

lower_std10 = LR_mean_std10-LR_std_std10
higher_std10 = LR_mean_std10+LR_std_std10

lower_std25 = LR_mean_std25-LR_std_std25
higher_std25 = LR_mean_std25+LR_std_std25

plt.figure()
plt.plot(xx, LR_mean_std5.reshape(-1,), '--', linewidth = 3 , color = 'red', label = 'std  5' )
plt.fill_between(xx, lower_std5[0], higher_std5[0], alpha=0.5, facecolor = 'red')
plt.plot(xx, LR_mean_std10.reshape(-1,), '--', linewidth = 3 , color = 'yellow', label = 'std 10' )
plt.fill_between(xx, lower_std10[0], higher_std10[0], alpha=0.5, facecolor = 'yellow')
plt.plot(xx, LR_mean_std25.reshape(-1,), '--', linewidth = 3 , color = 'cyan', label = 'std 25' )
plt.fill_between(xx, lower_std25[0], higher_std25[0], alpha=0.5, facecolor = 'cyan')
plt.legend(loc = "lower right", fontsize = 20)
plt.tick_params('x',labelsize=20)
plt.tick_params('y',labelsize=20)
# plt.title("LEIA Model learning rate against PE")
# plt.xlabel("Prediction Error")
# plt.ylabel("Learning Rate")
# plt.savefig('LEIA_model_lr.svg', format='svg',dpi=500, bbox_inches='tight')
#### end drawing

