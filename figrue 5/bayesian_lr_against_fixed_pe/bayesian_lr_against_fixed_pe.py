#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is to simulate the learning rate given fixed prediction error of the reduced Bayesian model
"""

### importing libraries ###
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from get_beliefs import *
from tqdm import tqdm
plt.rcParams['font.family'] = 'Times New Roman'


LR_ALL = []
for sigma in [5, 10, 25]:
    ## data generation
    u1 = 50
    sigma1 = sigma
    num1 = 10

    nb_participants = 1000

    LR = np.zeros((nb_participants, 10))
    for j in tqdm(range(nb_participants)):

        data1 = np.random.normal(u1, sigma1, (1, 10))
        data1 = np.maximum(data1, 0)

        second_state = np.zeros((1,2))

        data_all = list(np.concatenate([data1, second_state], axis=1).reshape(-1,))

        # trial_num = num1+2
                
        redc_bas_pred = np.zeros((10, len(data_all)))
        OUT = np.zeros((10, len(data_all)))

        ## start reduced bayesian model
        for i in range(10):
            beliefs = def_blf()
            get_beliefs(data_all, beliefs, (i+1)*10, sigma1)
            OUT[i,:] = data_all
            redc_bas_pred[i,:] = beliefs.pred[:-1].reshape(-1,)

        update = redc_bas_pred[:,11] - redc_bas_pred[:,10]
        PE     = OUT[:,10] - redc_bas_pred[:,10]
        lr = update / PE
        lr[lr<0] = 0
        LR[j,:] = lr

    LR_ALL.append(LR)

# print((end - start) * 1000)
cl = ['red', 'yellow', 'cyan']
std_list = [5, 10, 25]
plt.figure()
for i in range(len(LR_ALL)):
    lr_mean = np.mean(LR_ALL[i], axis = 0)
    lr_std = np.std(LR_ALL[i], axis = 0)
    xx = [i*10+10 for i in range(10)]

    plt.fill_between(xx, lr_mean - lr_std, lr_mean + lr_std, alpha=0.5, color = cl[i])
    plt.plot(xx, lr_mean, '--',color = cl[i],label = 'std {}'.format(std_list[i]))
plt.legend(loc = "lower right", fontsize = 20)
plt.tick_params('x',labelsize=20)
plt.tick_params('y',labelsize=20)
# plt.savefig('Reduced_Bayesian_model_lr.svg', format='svg',dpi=500, bbox_inches='tight')
plt.show()

