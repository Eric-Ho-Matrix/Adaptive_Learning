import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
import copy



def Load_vecterized_data(npz_file):
    load_data = np.load(npz_file, allow_pickle=True)

    model_data = load_data["arr_0"][0][0]

    # print(model_data.shape)    # (478,32,1)  

    model_response_ind = np.zeros((model_data.shape[1],model_data.shape[0]))

    for i in range(model_data.shape[1]):
        model_response_ind[i] = model_data[:,i,:].reshape(model_data.shape[0],)
    
    return model_response_ind      ### (32,478)


def slice_data(data):
    raw_data = np.zeros((100, 80))
    for i in range(10):
        raw_data[:,3*i:3*i+8] = copy.deepcopy(data[:,(i+2)*10+1 : (i+2)*10+9])

    return raw_data


MSE_NOISE     = np.zeros((11))
MSE_BASE      = np.zeros((11))
MSE_REVERSAL  = np.zeros((11))
SEM_noise     = np.zeros((11))
SEM_base      = np.zeros((11))
SEM_reversal  = np.zeros((11))

base_mse     = np.zeros((300, 11))
reversal_mse = np.zeros((300, 11))
noise_mse    = np.zeros((300, 11))

# n_batch = 0
for n_batch in range(3):
    for cpt_num in range(11):
        print("batch_number: {}, cpt_number; {}".format(n_batch+1, cpt_num))

        npz_file = "noise_driven_model/mean_val_{}/Mean_list_noise_driven_reversal_cpt_rev_cptnum_{}.npz".format(n_batch+1, cpt_num)
        load_data = np.load(npz_file, allow_pickle=True)
        mean = load_data["arr_0"]   ## (100, 12)
        mean = np.repeat(mean, 10, axis=-1)  ## (100, 120)
        calculate_mean = slice_data(mean)

        npz_file = "noise_driven_model/model_pred_{}/noise_driven_reversal_cpt_rev_cptnum_{}.npz".format(n_batch+1, cpt_num)
        noise_model = Load_vecterized_data(npz_file)   ## (100,120)
        calculate_noise_data = slice_data(noise_model)  ## (100,30)
        noise_mse[n_batch*100:(n_batch+1)*100, cpt_num] = np.sum(abs(calculate_noise_data - calculate_mean), axis=1) # (100,)


        npz_file = "base_model/model_pred_{}/base_cpt_rev_cptnum_{}.npz".format(n_batch+1, cpt_num)
        base_model = Load_vecterized_data(npz_file)   ## (100,120)
        calculate_base_data = slice_data(base_model)
        base_mse[n_batch*100:(n_batch+1)*100, cpt_num] = np.sum(abs(calculate_base_data - calculate_mean), axis=1)


### calculate mean error
for cpt_num in range(11):
    MSE_BASE[cpt_num] = (np.mean(base_mse[:,cpt_num])).item()
    SEM_base[cpt_num] = (np.array(np.std(base_mse[:,cpt_num]) / np.sqrt(300))).item()

    MSE_NOISE[cpt_num] = (np.mean(noise_mse[:,cpt_num])).item()
    SEM_noise[cpt_num] = (np.array(np.std(noise_mse[:,cpt_num]) / np.sqrt(300))).item()

    MSE_REVERSAL[cpt_num] = (np.mean(reversal_mse[:,cpt_num])).item()
    SEM_reversal[cpt_num] = (np.array(np.std(reversal_mse[:,cpt_num]) / np.sqrt(300))).item()


### start plotting
xx = [i/10 for i in range(11)]

plt.figure()

plt.scatter(xx, MSE_NOISE, color='red', s=50)
plt.plot(xx, MSE_NOISE, color='red', label="Flexible Model")
plt.fill_between(xx, MSE_NOISE - SEM_noise, MSE_NOISE + SEM_noise, alpha=0.5, facecolor='red')

plt.scatter(xx, MSE_BASE, color='blue', s=50)
plt.plot(xx, MSE_BASE, color='blue', label="Change-point Model")
plt.fill_between(xx, MSE_BASE - SEM_base, MSE_BASE + SEM_base, alpha=0.5, facecolor='blue')

plt.legend(fontsize=15)
# plt.set_yscale('log')  # Set Y-axis to log scale
plt.yscale('log')
plt.ylim(200, 500)
plt.tick_params(axis='both', labelsize=16)
plt.xlabel("Change-point / Reversal-point Ratio")
plt.ylabel("Absolute Error (logarithmic scale)")
# plt.savefig('abs_err_three_models_cpt_rev_comp_2.svg', format="svg" ,dpi=300, bbox_inches='tight')
plt.show()











