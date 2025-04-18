#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is to generate the fig 4 (against time) in the paper.
"""

## 4 8 12 17

### importing libraries ###
import copy
from scipy.stats import linregress
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from draw_plots_heat_map import Draw_plots
import matplotlib.patches as patches
plt.rcParams['font.family'] = 'Times New Roman'


# data generation
u1 = 50
u2 = 250
u3 = 150
u4 = 200
u5 = 200
u6 = 30
u7 = 30
u8 = 10
u9 = 10
u10 = 10

sigma1 = 0
sigma2 = 0
sigma3 = 0
sigma4 = 0.3
sigma5 = 0.3
sigma6 = 0.3
sigma7 = 0.3
sigma8 = 0.3
sigma9 = 0.3
sigma10 = 0.3

num1 = 8
num2 = 9
num3 = 8
num4 = 0
num5 = 0
num6 = 0
num7 = 0
num8 = 0
num9 = 0
num10 = 0

total_trials = 0

for i in range(1, 11):
    total_trials += locals()['num' + str(i)]

data1 = np.random.normal(u1, sigma1, (1, num1))
data2 = np.random.normal(u2, sigma2, (1, num2))
data3 = np.random.normal(u3, sigma3, (1, num3))
data4 = np.random.normal(u4, sigma4, (1, num4))
data5 = np.random.normal(u5, sigma5, (1, num5))
data6 = np.random.normal(u6, sigma6, (1, num6))
data7 = np.random.normal(u7, sigma7, (1, num7))
data8 = np.random.normal(u8, sigma8, (1, num8))
data9 = np.random.normal(u9, sigma9, (1, num9))
data10 = np.random.normal(u10, sigma10, (1, num10))


data12 = np.append(data1, data2)
data13 = np.append(data12, data3)
data14 = np.append(data13, data4)
data15 = np.append(data14, data5)
data16 = np.append(data15, data6)
data17 = np.append(data16, data7)
data18 = np.append(data17, data8)
data19 = np.append(data18, data9)
data110 = np.append(data19, data10)

num12 = num1 + num2
num13 = num12 + num3
num14 = num13 + num4
num15 = num14 + num5
num16 = num15 + num6
num17 = num16 + num7
num18 = num17 + num8
num19 = num18 + num9
num110 = num19 + num10


mean_generate = []
for i in range(num1):
    mean_generate.append(u1)
for i in range(num1, num12):
    mean_generate.append(u2)
for i in range(num12, num13):
    mean_generate.append(u3)
for i in range(num13, num14):
    mean_generate.append(u4)

for i in range(num14, num15):
    mean_generate.append(u5)
for i in range(num15, num16):
    mean_generate.append(u6)
for i in range(num16, num17):
    mean_generate.append(u7)

for i in range(num17, num18):
    mean_generate.append(u8)
for i in range(num18, num19):
    mean_generate.append(u9)
for i in range(num19, num110):
    mean_generate.append(u10)


class Spiking_Neuron:
    def __init__(self):
        # Define parameters
        self.threshold = 4.3  # Threshold for spiking
        self.resting_potential = -70  # Resting membrane potential in mV
        self.refractory_period = 600  # Refractory period in time steps
        self.membrane_potentials = []
        
        # Initialize variables
        self.membrane_potential = self.resting_potential
        self.in_refractory_period = 0  # Counter to track refractory period
        self.tau = 75
        
    def update_potential(self, input_sig):
        if self.in_refractory_period == 0:
            if self.membrane_potential < self.threshold:
                self.membrane_potential = np.minimum(0.9 * self.membrane_potential + input_sig, self.threshold)
                self.membrane_potentials.append(self.membrane_potential)
            else:
                self.membrane_potential = (1 - 1 / self.tau) * self.membrane_potential + (1 / self.tau) * self.resting_potential # Reset membrane potential
                # self.membrane_potential = self.resting_potential
                self.membrane_potentials.append(self.membrane_potential)
                self.in_refractory_period = self.refractory_period
        else:
            self.membrane_potential = (1 - 1 / self.tau) * self.membrane_potential + (1 / self.tau) * self.resting_potential
            # self.membrane_potential = self.resting_potential
            self.membrane_potentials.append(self.membrane_potential)
            self.in_refractory_period -= 1


### We start by defining all the functions needed for simulations ###

### Starting with connectivity matrices for all layers of the network ###

# output layer connectivity (ring-like, center-surround)
def output_connectivity(N_output, baseline, peak_height, peak_width):
    # open connectivity matrix
    conn = np.empty((N_output, N_output))
    conn[:] = np.nan

    # number of data points
    data_points = np.arange(0, N_output, 1)

    for i in range(N_output):
        bell1 = norm.pdf(data_points, i, peak_width)
        bell2 = norm.pdf(data_points, i+N_output, peak_width)
        bell3 = norm.pdf(data_points, i-N_output, peak_width)
        total_bell = bell1 + bell2 + bell3
        max_bell = max(total_bell)
        bell = baseline + ((peak_height-baseline)/max_bell)*total_bell
        conn[i, :] = bell

    return conn


def MSN_GP_TH_connectivity(N_output, weight_values, self_excitation):

    conn = np.ones((N_output, N_output)) * weight_values * (-1)
    np.fill_diagonal(conn, self_excitation)

    return conn

# RNN connectivity (encoding patterns/attractors)


def symmetric_conn(x, N_patterns, N_rnn):
    # if we want a rnn with orthogonal attractors, we need to make sure the parameter x is orthogonal
    # and x is encoded in function patts
    Force = 8
    weights = np.zeros((N_rnn, N_rnn))
    for u in range(N_patterns):
        weights += np.dot(x[u, :].reshape(1, N_rnn).T, x[u, :].reshape(1, N_rnn))
    weights -= 2.4 * np.ones((N_rnn, N_rnn))  # -2.9 -2.4
    
    return weights*(Force/N_rnn)

def asymmetric_conn_PFC_MSN():
    asy_conn = np.zeros((600, 600))
    
    for i in range(19):
        if i%2 == 0:
            asy_conn[0+i*30:30+i*30,30+i*30:60+i*30] = 1/30
            
        else:
            asy_conn[0+i*30:30+i*30,30+i*30:60+i*30] = (1/30)
            
    asy_conn[570:600,0:30] = (1/30)
    return (asy_conn/2.8) * 1


# RNN to MSN connectivity
def feed_RNN_to_MSN(N_rnn, N_output):

    g_u = 0.5
    g_sd = 0.1
    ffw = np.random.normal(g_u/N_rnn, g_sd/N_rnn, (N_rnn, N_output))

    return ffw

## change the inhibitory neuron

### we now write all network related functions ###

# threshold function for entropy induced inhibition in the memory integration layer
def thresh(x, limit):

    if x >= limit:
        j = x
    elif x < limit:
        j = 0
    return j

# function to build patterns

def patts(N_rnn, N_patterns):
    patt = np.zeros((N_patterns, N_rnn))
    for i in range(N_patterns):
        basic = np.zeros((1, N_rnn))
        basic[0, i*30:i*30+30] = 2.8   ## 3
        patt[i] = basic[0]
    return patt


# function to compute hopfield energy

def hopfield_energy(x, w):
    return -np.dot(x, np.dot(w, x.T))


# defining the thresholded Hebbian learning rule
def learning(x, y, teta_learn, weights, max_weight):

    lr = 0.2  # initial 0.01
    post = y-teta_learn
    pre = x-teta_learn
    post[post < 0] = 0
    pre[pre < 0] = 0
    w = (weights) - (0 * weights) + (lr * (np.matmul(pre.T, post)))  # weights decay
    w[w > max_weight] = max_weight
    #w = (w / np.max(w)) * max_weight
    w[w < 0] = 0
    return w

# decision function (check if decision is made through output layer stability)

def decision(y):

    convergence = 0.1
    x = np.arange(1, 6, 1)
    slope = linregress(x, y)[0]
    return np.abs(slope) < convergence


def update_window(H, H_window):

    # certainly a much smarter way to do this but I am tired :)
    window_size = 5
    new_H_window = np.zeros((1, window_size))
    new_H_window[0, -1] = H
    new_H_window[0, -2] = H_window[0, 4]
    new_H_window[0, -3] = H_window[0, 3]
    new_H_window[0, -4] = H_window[0, 2]
    new_H_window[0, -5] = H_window[0, 1]

    return new_H_window


def update_stability(S, S_window):

    # certainly a much smarter way to do this but I am tired :)
    window_size = 5
    new_S_window = np.zeros((1, window_size))
    new_S_window[0, -1] = S
    new_S_window[0, -2] = S_window[0, 4]
    new_S_window[0, -3] = S_window[0, 3]
    new_S_window[0, -4] = S_window[0, 2]
    new_S_window[0, -5] = S_window[0, 1]

    return new_S_window


def supervisory_signal(x, N_output, peak_width, peak_height):

    vec = np.arange(0, N_output, 1)
    bell1 = norm.pdf(vec, x, peak_width)
    bell2 = norm.pdf(vec, x+N_output, peak_width)
    bell3 = norm.pdf(vec, x-N_output, peak_width)
    total_bell = bell1 + bell2 + bell3
    max_bell = max(total_bell)
    bell = (total_bell/max_bell) * peak_height

    return bell.reshape(1, N_output)

def Entro_non_lin(Entropy, threshold):
    if Entropy >= threshold:
        return Entropy
    else:
        return 0


### Initializing model parameters ###

N_rnn = 600  # number of RNN neurons
N_patterns = 20  # number of encoded patterns
N_output = 360  # number of output neurons, which correspon to the potential location index of the bucket
tau_rnn = 1  # rnn neuron time constant
tau_output = 20  # output neuron time constant
tau_msn = 20  # msn neuron time constant
tau_gp = 20  # gp neuron time constant
tau_th = 20  # th neuron time constant
tau_mem = 1
tau_inh = 2  # time constant of hopfield energy 20
tau_w_inh = 200  # 2
tau_bar = 100
teta = 0.1
leak_H = 0
leak_I = 0.1
leak_mem = 0.25  # 0 for no leak or 1 for full leak or create
baseline = -0.1  # value of surround inhibition
peak_width = 20     # 10 15 width of self-excitation 10 THIS IS WHAT MATTERS!
peak_height = 0.20  # peak value of self-excitation
supervisory_height = 2  # strength of supervision
supervisory_width = 5   # 10spread of supervision 10
stable_max = 50  # nmber of stable iterations to make a decision
teta_hebb = 0.2  # threshold for hebb learning rule
learning_window = 500  # max time to switch attractor state for learning initial 500
a_noise_u = 0  # mean of output noise
a_noise_sd = 1  # standard deviation of output noise
rnn_noise_u = 0  # same for rnn
rnn_noise_sd = 2.4  # same for rnn
weight_values_msn = 1  # inhibitory weight values MSN layer
weight_values_gp = 1  # inhibitory weight values GP layer
weight_values_th = 1  # inhibitory weight values Th layer
self_excitation = 1
bias_inh = 2.8  # bias on the inhibitory neuron
max_inh = 0.2  #original 3  # set to 22 piece-wise linear transormation of the inhibitory neuron
inh_limit = 0.30  # 35 40 ca amrche
burnin = -1
Entro_gain = 10   # origin value 10
sup_gain = 50
bias_mem = 0.1
bias_msn = 0.35   # 7e-2
k=0.3   # 0.3
bias_th = 0.5  # 0.48 
bias_gp = 0.8  # 0.9 1.5

patterns = patts(N_rnn, N_patterns)
Ws = symmetric_conn(patterns, N_patterns, N_rnn)
aWs = asymmetric_conn_PFC_MSN()

# plot the connectivity
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
im = ax.imshow(Ws)
ax.set_title('connectivity matrix', fontsize=25)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
clb = fig.colorbar(im, cax=cax)
clb.set_label('weight values', labelpad=17,fontsize=15, rotation=270)
plt.tight_layout()
plt.show()
###  end plot

output_weights = output_connectivity(N_output, baseline, peak_height, peak_width)
weights_rnn_inh = np.zeros((1, N_rnn))
MSN_weights = output_connectivity(N_output, baseline, peak_height, peak_width)
GP_weights = MSN_GP_TH_connectivity( N_output, weight_values_gp, self_excitation)
TH_weights = MSN_GP_TH_connectivity(N_output, weight_values_th, self_excitation)


# further fixed weights
weight_Gp_Msn = 1
weight_Th_Gp = 1
weight_M_Th = 1
weight_Th_M = 20
weight_mem_Th = 0.11  ##original 0.11
weight_Th_mem = 1000  # 1
weight_mem_mem = 0.5
weight_msn_out = 50
weight_I_Th = 1   #original 1
weight_I_mem = 1000  # 1
W_inh_max = 100
W_inh_min = 0.2
beta = 1


# initialize rnn
rnn= copy.deepcopy(patterns[0].reshape(1,600))

# open record lists
H_window = np.zeros((1, 5))

# get the responses
model_response = []

# participants and blocks numbers
nb_participants = 1
nb_blocks = 1
nb_participants = 1
show_plots = 1

rec_sim = []

LEIA_Pred = []

p = 0
z = 0

#non-linear methods to allow for the expression of a single attractor at a time
#method 1: RNN weight matrix is 0 in the offdiagonal space
#method 2: non-linearity at zero in RNN unit activation 
    
method = 2

if method == 1:
    Ws[Ws<0] = 0
    print('Mehtod 1: Non linearity at zero on RNN weight values')
elif method == 2:
    print('Method 2: Non linearity at zero on RNN activation units')

# switcher = Switcher(0, patterns)

while(1):
    LEIA_Pred = []
    proper_inital_flag=0
    end_flag = 0
    # sim loop
    for p in range(1):  # nb_participants

        model_response = []

        for b in range(1):  # nb_blocks
            #b = 2
            # reset things
            mem_layer = np.zeros((1, N_output))
            outcomes = []

            # extract the number of trials for each block
            Fs = feed_RNN_to_MSN(N_rnn, N_output)
            weights_rnn_inh = np.zeros((1, N_rnn))
            max_weight = 0.1
            outcomes = data110
            # outcomes = np.ones(5)*50
            means = mean_generate

            First_stable_perd = []   # sample trial 4
            First_change_pts = []    # sample trial 8
            second_stable_perd = []  # sample trial 12
            second_change_pts = []   # sample trial 17

            for t in range(total_trials):  # nb_trials

                print('Participants: ' + str(p) + ' // block: ' +
                    str(b) + ' // trial: ' + str(t))

                # reset all but RNN
                output_layer = np.zeros((1, N_output))
                msn_layer = np.zeros((1, N_output))
                gp_layer = np.ones((1, N_output)) * 2
                th_layer = np.zeros((1, N_output))
                rbar = np.zeros((1, N_rnn))
                inh_neuron = 0
                H = 0  # hopfield for decision
                H_rnn = 0  # hopfield for RNN
                Entropy = 0  # multiplicative gain to noise in RNN
                count = 0
                supervision_input = np.zeros((1, N_output))
                stable_time = 0
                spike_neuron = Spiking_Neuron()

                rec_rnn = []
                rec_msn = []
                rec_mem = []
                rec_E = []
                rec_inh = []
                rec_th = []
                rec_output = []
                rec_GP = []

                # open record lists
                H_window = np.zeros((1, 5))
                S_window = np.zeros((1, 5))
                
                rnn = np.tanh(beta * rnn)
                

                # start of the decision dynamics
                while 1:
                    
                    # RNN dynamics
                    noise_rnn = np.random.normal(0, 0, N_rnn).reshape(1, N_rnn)
                    rnn += (-rnn + (np.matmul(rnn, Ws)) + spike_neuron.membrane_potential * 1 * (np.matmul(rnn, aWs))) / tau_rnn
                    rnn = np.tanh(beta * rnn)
                    if method == 2:
                        rnn[rnn < 0] = 0
                    
                    current_state = copy.deepcopy(rnn)

                    # low pass filter for inhibitory weights
                    rbar += (-rbar + rnn)/tau_bar

                    # MSN dynamics
                    noise_msn = np.random.normal(0, 0, N_output).reshape(1, N_output)
                    msn_layer += (-msn_layer + 12*np.matmul(rnn, Fs) + (output_layer * weight_msn_out) - bias_msn + noise_msn) / tau_msn
                    msn_layer = np.tanh(msn_layer)
                    msn_layer[msn_layer < 0] = 0

                    # GP dynamics
                    noise_gp = np.random.normal(0, 0, N_output).reshape(1, N_output)
                    gp_layer += (gp_layer - (msn_layer * weight_Gp_Msn) + bias_gp + noise_gp) / tau_gp
                    gp_layer = np.tanh(gp_layer)
                    gp_layer[gp_layer < 0] = 0
                    
                    # Th dynamics
                    noise_th = np.random.normal(0, 0.05, N_output).reshape(1, N_output)
                    th_layer += (-k*th_layer - (gp_layer * weight_Th_Gp) + (output_layer * weight_Th_M) + bias_th + noise_th) / tau_th
                    th_layer = np.tanh(th_layer)
                    th_layer[th_layer < 0] = 0
                    Entropy = (np.array([[0]]).astype(float)) * Entro_gain
                    spike_neuron.update_potential(0)

                    # output dynamics
                    noise_output = np.random.normal(0, 0, N_output).reshape(1, N_output)
                    output_layer += (-output_layer + np.matmul(output_layer, output_weights) + (th_layer * weight_M_Th) + supervision_input + noise_output) / tau_output
                    output_layer = np.tanh(output_layer)
                    output_layer[output_layer < 0] = 0


                    # compute hopfield energy and update the energy window
                    H = hopfield_energy(output_layer, output_weights)[0][0]
                    H_window = update_window(H, H_window)

                    # reset stability clock if unstable
                    if decision(H_window) == False:
                        stable_time = 0

                    # check if decision has been reached
                    if (count > 25) and decision(H_window) == True and np.mean(np.abs(H_window)) > 3:
                        stable_time += 1
                        if stable_time > stable_max:

                            test = np.argmax(output_layer)
                            model_response.append(np.argmax(output_layer))
                            # LEIA_Pred.append(np.argmax(output_layer))
                            print("LEIA prediction output: ")
                            LEIA_Pred.append(np.argmax(output_layer))
                            print(np.argmax(output_layer))
                            print(" ")

                            if t == 0: ## to make sure we can plot beautiful plots
                                if (u1-1) <= np.argmax(output_layer) <= (u1+1):
                                    pass
                                else: proper_inital_flag = 1
                            break
                    
                    rec_mem.append(copy.deepcopy(mem_layer))
                    rec_E.append(copy.deepcopy(Entropy))
                    rec_inh.append(copy.deepcopy(inh_neuron))
                    rec_rnn.append(copy.deepcopy(rnn))
                    rec_msn.append(copy.deepcopy(msn_layer))
                    rec_th.append(copy.deepcopy(th_layer))
                    rec_output.append(copy.deepcopy(output_layer))
                    rec_GP.append(copy.deepcopy(gp_layer))

                    count += 1  # count allows the network to build in some activity

                if proper_inital_flag==1:
                    break

                # get overlap values between rnn and stored patterns
                init_rnn = copy.deepcopy(rnn)

                # reset count (for learning window)
                count = 0

                # activation of output node
                supervision_input = supervisory_signal(outcomes[t], N_output, supervisory_width, supervisory_height)
                print("real output:")
                print(outcomes[t])
                    
                

                # start learning dynamics
                while 1:
                    

                    # hopfield energy and stability (window)
                    H = hopfield_energy(output_layer, output_weights)[0][0]
                    H_window = update_window(H, H_window)

                    # RNN dynamics
                    noise_rnn = np.random.normal(0, 0.0, N_rnn).reshape(1, N_rnn)
                    rnn += (-rnn + (np.matmul(rnn, Ws)) + spike_neuron.membrane_potential * 1 * (np.matmul(rnn, aWs))) / tau_rnn
                    rnn = np.tanh(beta * rnn)
                    if method == 2:
                        rnn[rnn < 0] = 0
                    
                    # rnn = sigmoid(rnn)
                    current_state = copy.deepcopy(rnn)

                    # low pass filter for inhibitory weights
                    rbar += (-rbar + rnn)/tau_bar

                    # MSN dynamics
                    noise_msn = np.random.normal(0, 0, N_output).reshape(1, N_output)
                    msn_layer += (-msn_layer + 12*np.matmul(rnn, Fs) + (output_layer * weight_msn_out) - bias_msn + noise_msn) / tau_msn
                    msn_layer = np.tanh(msn_layer)
                    msn_layer[msn_layer < 0] = 0

                    # GP dynamics
                    noise_gp = np.random.normal(0, 0, N_output).reshape(1, N_output)
                    gp_layer += (gp_layer - (msn_layer * weight_Gp_Msn) + bias_gp + noise_gp) / tau_gp
                    gp_layer = np.tanh(gp_layer)
                    gp_layer[gp_layer < 0] = 0

                    # Th dynamics
                    noise_th = np.random.normal(0, 0, N_output).reshape(1, N_output)
                    th_layer += (-k*th_layer - (gp_layer * weight_Th_Gp) + (output_layer * weight_Th_M) + bias_th + noise_th) / tau_th
                    th_layer = np.tanh(th_layer)
                    th_layer[th_layer < 0] = 0
                    Normalize = th_layer / np.max(th_layer)
                    Normalize = np.nan_to_num(Normalize, nan=0)
                    log_norm = -np.log2(Normalize)
                    log_norm[log_norm > 1000000] = 0
                    sum_act = np.sum(th_layer)
                    if sum_act < 1:
                        sum_act = 1
                    Entropy = ((np.dot(Normalize, log_norm.T)) / sum_act) * Entro_gain
                    Entropy = Entro_non_lin(Entropy, bias_inh)
                    spike_neuron.update_potential(copy.deepcopy(Entropy))

                    # output dynamics
                    noise_output = np.random.normal(0, 0, N_output).reshape(1, N_output)
                    output_layer += (-output_layer + np.matmul(output_layer, output_weights) + (th_layer * weight_M_Th) + (sup_gain*supervision_input) + noise_output) / tau_output
                    output_layer = np.tanh(output_layer)
                    output_layer[output_layer < 0] = 0

                    if count > learning_window:  # 500

                        if t > burnin:
    
                            Fs = learning(rnn, msn_layer, teta_hebb, Fs, max_weight)  # teta_hebb = 0.2

                        break

                    count += 1

                    rec_mem.append(copy.deepcopy(mem_layer))
                    rec_E.append(copy.deepcopy(Entropy))
                    rec_inh.append(copy.deepcopy(inh_neuron))
                    rec_rnn.append(copy.deepcopy(rnn))
                    rec_msn.append(copy.deepcopy(msn_layer))
                    rec_th.append(copy.deepcopy(th_layer))
                    rec_output.append(copy.deepcopy(output_layer))
                    rec_GP.append(copy.deepcopy(gp_layer))  

                if show_plots == 1:

                    if t == 4:
                        First_stable_perd.append(rec_rnn)
                        First_stable_perd.append(Fs)
                        First_stable_perd.append(rec_msn)
                        First_stable_perd.append(rec_GP)
                        First_stable_perd.append(rec_th)
                        First_stable_perd.append(rec_inh)
                        First_stable_perd.append(rec_E)
                        First_stable_perd.append(rec_output)
                        First_stable_perd.append(means)
                        First_stable_perd.append(test)
                        First_stable_perd.append(spike_neuron.membrane_potentials)


                    if t == 8:
                        First_change_pts.append(rec_rnn)
                        First_change_pts.append(Fs)
                        First_change_pts.append(rec_msn)
                        First_change_pts.append(rec_GP)
                        First_change_pts.append(rec_th)
                        First_change_pts.append(rec_inh)
                        First_change_pts.append(rec_E)
                        First_change_pts.append(rec_output)
                        First_change_pts.append(means)
                        First_change_pts.append(test)
                        First_change_pts.append(spike_neuron.membrane_potentials)
                    
                    if t == 12:
                        second_stable_perd.append(rec_rnn)
                        second_stable_perd.append(Fs)
                        second_stable_perd.append(rec_msn)
                        second_stable_perd.append(rec_GP)
                        second_stable_perd.append(rec_th)
                        second_stable_perd.append(rec_inh)
                        second_stable_perd.append(rec_E)
                        second_stable_perd.append(rec_output)
                        second_stable_perd.append(means)
                        second_stable_perd.append(test)
                        second_stable_perd.append(spike_neuron.membrane_potentials)
                        

                    if t == 17:
                        second_change_pts.append(rec_rnn)
                        second_change_pts.append(Fs)
                        second_change_pts.append(rec_msn)
                        second_change_pts.append(rec_GP)
                        second_change_pts.append(rec_th)
                        second_change_pts.append(rec_inh)
                        second_change_pts.append(rec_E)
                        second_change_pts.append(rec_output)
                        second_change_pts.append(means)
                        second_change_pts.append(test)
                        second_change_pts.append(spike_neuron.membrane_potentials)

                print("Entropy is :", Entropy)
                print(" ")
                
                if t==24: end_flag=1
    if end_flag==1:break



# plot the prediction outcome and the real out come
fig, ax = plt.subplots(figsize=(15,8))

# Create a rectangle patch
rectangle = patches.Rectangle((1.5, 49), 2, 105, linewidth=1, edgecolor='black', facecolor='yellow')
# Add the rectangle to the plot
ax.add_patch(rectangle)

# Create a rectangle patch
rectangle = patches.Rectangle((6.5, 49), 2, 105, linewidth=1, edgecolor='black', facecolor='cyan')
# Add the rectangle to the plot
ax.add_patch(rectangle)

# Create a rectangle patch
rectangle = patches.Rectangle((11.5, 149), 2, 105, linewidth=1, edgecolor='black', facecolor="orchid")
# Add the rectangle to the plot
ax.add_patch(rectangle)

# Create a rectangle patch
rectangle = patches.Rectangle((16.5, 149), 2, 105, linewidth=1, edgecolor='black', facecolor="springgreen")
# Add the rectangle to the plot
ax.add_patch(rectangle)

plt.plot(data110, 'o', color="grey", markersize=10, label='True hidden state')
plt.plot(LEIA_Pred, '-', linewidth=7, color='purple', label='LEIA model')
plt.yticks(fontsize=50)
plt.xticks(fontsize=50)
plt.legend(loc='upper left', fontsize=50)
# plt.savefig('fig_2_dataset_heat_map.svg', format="svg",dpi=300)
fig.tight_layout()
plt.show()

# # Create a 4x2 subplot grid using subplot2grid
fig = plt.figure(figsize=(50, 20))
for i, (record_list, trial_t) in enumerate(zip([First_stable_perd, First_change_pts, second_stable_perd, second_change_pts],[4,8,12,17])):
    Draw_plots(
            N_rnn   = N_rnn, 
            rec_rnn = record_list[0], 
            Fs      = record_list[1],
            rec_msn = record_list[2], 
            rec_GP  = record_list[3], 
            rec_th  = record_list[4], 
            rec_inh = record_list[5], 
            rec_E   = record_list[6], 
            rec_output = record_list[7], 
            means   = record_list[8], 
            trial = trial_t, 
            test = record_list[9],
            combine_flag=1,
            save_flag=1,
            f_name="figure_4_combine_heatmap",
            y_range=[0,6],
            fig=fig,
            ax1=plt.subplot2grid((5, 8), (0, 0+2*i), colspan=2),
            ax2=plt.subplot2grid((5, 8), (1, 0+2*i), colspan=2),
            ax3=plt.subplot2grid((5, 8), (2, 0+2*i), colspan=2),
            ax4=plt.subplot2grid((5, 8), (3, 0+2*i), colspan=2),
            ax5=plt.subplot2grid((5, 8), (4, 0+2*i), colspan=2)
            # ax6=plt.subplot2grid((6, 8), (5, 0+2*i)),
            # ax7=plt.subplot2grid((6, 8), (5, 1+2*i))
            )
plt.show()


# draw the weights and entropy
fig = plt.figure(figsize=(30, 15))
for i, (record_list, trial_t) in enumerate(zip([First_stable_perd, First_change_pts, second_stable_perd, second_change_pts],[4,8,12,17])):
    ax1=plt.subplot2grid((2, 4), (0, 0+i), colspan=1)
    ax2=plt.subplot2grid((2, 4), (1, 0+i), colspan=1)

    Fs      = record_list[1]
    im1 = ax1.imshow(Fs[0:100,:])
    if trial_t == 4:
        # ax1.set_title('RNN - MSN weights', fontsize=45, pad=10)
        # ax1.set_xlabel('Receiving Neurons', fontsize=45)
        ax1.tick_params(axis='y', labelsize=40)
        # ax1.set_ylabel('Sending Neurons', fontsize=45, labelpad=10)
        ax1.set_yticks([0, 30, 60, 90])
    else: ax1.set_yticks([])
    ax1.tick_params(axis='x', labelsize=40)
    ax1.set_xticks([0, 100, 200, 300])
    divider = make_axes_locatable(ax1)
    if trial_t == 17:
        cax = divider.append_axes("right", size="5%", pad=0.05)
        clb = fig.colorbar(im1, cax=cax)
        clb.ax.tick_params(labelsize=40)
        # clb.set_label('weight values', labelpad=55 ,fontsize=35, rotation=270)
        clb.set_ticks([0, 0.05, 0.1])

    # fig.savefig('RNN_MSN_Weight_{}.svg'.format(trial), format = 'svg',dpi=300)

    y_range=[0,6]
    rec_E   = record_list[6]
    plot_E = np.zeros((1, len(rec_E)))
    for i in range(len(rec_E)):
        plot_E[0, i] = rec_E[i]
    
    ax2.plot(plot_E[0, :], label='Entropy', linewidth=3, color="orange")
    ax2.axhline(y=bias_inh, linestyle = "--", label="Entropy Threshold", color='r')
    if trial_t == 4:
        # ax2.set_xlabel('Time', fontsize=45)
        # ax2.set_ylabel('Activity', fontsize=45, labelpad=10)
        ax2.set_yticks([1, 3, 5])
        ax2.tick_params(axis='y', labelsize=45)
        # ax2.set_title('Entropy', fontsize=45)
    else: ax2.set_yticks([])
    ax2.set_xticks([0, 200, 400, 600])

    ax2.tick_params(axis='x', labelsize=45)
    # ax2.set_ylim(y_range)
    
    duration = len(record_list[2])+2
    ax_twin = ax2.twinx()
    membrane_potentials = record_list[10]
    ax_twin.axhline(y = spike_neuron.threshold, linestyle = "--",label="Transition Threshold", color='b')
    ax_twin.plot(range(duration), membrane_potentials, linewidth=2, label="Transition Signal")
    if trial_t == 17:
        ax_twin.tick_params(axis='y', labelsize=45)
        ax_twin.set_yticks([0, -20, -50])

        # ax_twin.tick_params(axis='x', labelsize=25)
        # ax_twin.set_ylabel('Potential', labelpad=45 ,fontsize=45, rotation=270)
    else: ax_twin.set_yticks([])

    lines = ax2.get_lines() + ax_twin.get_lines()
    labels = [line.get_label() for line in lines]
    if trial_t == 17:
        ax2.legend(lines, labels, fontsize=35, loc="lower right")
    plt.subplots_adjust(hspace=0.1, wspace=0.2)
    # fig.savefig('Entropy_and_potential.svg', format = 'svg',dpi=300)
plt.show()