import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

 
BASE     = np.zeros((300, 11))
NOISE    = np.zeros((300, 11))
REVERSAL = np.zeros((300, 11))

# decoding data

for nb_batch in range(3):
    for cpt_num in range(11):

        path = "base_model/model_pred_{}/attr_cptnum_{}.npz".format(nb_batch+1, cpt_num)
        load_data = np.load(path, allow_pickle=True)
        base_model_attr = load_data["arr_0"]   # (100,)
        BASE[nb_batch*100:(nb_batch+1)*100, cpt_num] = base_model_attr


        path = "noise_driven_model/model_pred_{}/attr_cptnum_{}.npz".format(nb_batch+1, cpt_num)
        load_data = np.load(path, allow_pickle=True)
        noise_model_attr = load_data["arr_0"]
        NOISE[nb_batch*100:(nb_batch+1)*100, cpt_num] = noise_model_attr

### Calculate the standard error of mean of the number of latent states inferred

err_BASE = np.std(BASE, axis=0)          / np.sqrt(300)
err_NOISE = np.std(NOISE, axis=0)        / np.sqrt(300)
err_REVERSAL = np.std(REVERSAL, axis=0)  / np.sqrt(300)

xx = [i for i in range(11)]
plt.figure(figsize=(10,8))

# plt.plot(xx, np.mean(REVERSAL, axis=0), color='green', label='Reversal Model')
plt.plot(xx, np.mean(NOISE, axis=0), color='red', label='Flexible Model')
plt.plot(xx, np.mean(BASE, axis=0), color='blue', label='Change-point Model')

# plt.scatter(xx, np.mean(REVERSAL, axis=0), color='green', s=85)
plt.scatter(xx, np.mean(NOISE, axis=0), color='red', s=85)
plt.scatter(xx, np.mean(BASE, axis=0), color='blue', s=85)

plt.fill_between(xx, np.mean(BASE, axis=0)-err_BASE, np.mean(BASE, axis=0)+err_BASE, color='blue', alpha=0.3)
plt.fill_between(xx, np.mean(NOISE, axis=0)-err_NOISE, np.mean(NOISE, axis=0)+err_NOISE, color='red', alpha=0.3)


plt.xlabel("Change Point Number", fontsize=16)
plt.ylabel("The number of latent states inferred", fontsize=16)
plt.tick_params(axis='both', labelsize=20)
plt.legend(fontsize=15)
# plt.savefig('att_num_compara_2.svg', format="svg" ,dpi=300, bbox_inches='tight')
plt.show()
