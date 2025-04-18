#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is to simulate the flexible model for the mix environment learning task.

"""

### importing libraries ###
from scipy.linalg import hadamard
import copy
from scipy.stats import linregress
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt 
from scipy.stats import norm
import numpy as np
import os



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

# MSN GP Thalamus connectivity (full inhibitory)
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
    weights -= 3.5 * np.ones((N_rnn, N_rnn))  # -2.9 # off-diag value is defined here -2.4  1.3
    
    return weights*(Force/N_rnn)

# vectorize version
def feed_RNN_to_MSN(N_rnn, N_output, n_sample):  # Fs = feed_RNN_to_MSN(N_rnn, N_output)

    g_u = 0.5
    g_sd = 0.1
    ffw = np.random.normal(g_u/N_rnn, g_sd/N_rnn, (n_sample, N_rnn, N_output))
    return ffw

# single version
def feed_RNN_to_MSN_single(N_rnn, N_output):  # Fs = feed_RNN_to_MSN(N_rnn, N_output)

    g_u = 0.5
    g_sd = 0.1
    ffw = np.random.normal(g_u/N_rnn, g_sd/N_rnn, (N_rnn, N_output))
    return ffw

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
        basic[0, i*30:i*30+30] = 3   ## 3 #diag value is defined here 3.15
        patt[i] = basic[0]
    return patt

# function to compute hopfield energy
def hopfield_energy(x, w):
    return -np.dot(x, np.dot(w, x.T))

# defining the thresholded Hebbian learning rule  Vectorize
# defining the thresholded Hebbian learning rule  Vectorize
def learning(x, y, teta_learn, weights, max_weight):  # Fs = learning(rnn, msn_layer, teta_hebb, Fs, max_weight)
    ## weights have shape (n_sample, 600, 360)
    ## max_weights have shape (10,1)
    lr = 0.01  # initial 0.01
    post = y-teta_learn   #shape(10,360)
    pre = x-teta_learn    #shape(10,600)
    post[post < 0] = 0
    pre[pre < 0] = 0
    
    pre_reshape = pre[:,:,np.newaxis]         # (10,600,1)
    post_reshape = post[:,np.newaxis,:]       # (10,1,360)
    
    mul_pre_post = np.einsum('ijk,ikl->ijl', pre_reshape, post_reshape)  #(nb_participants, 600,300)
    
    w = (weights) - (0 * weights) + (lr * mul_pre_post)  # weights decay  (n_sample, 600, 360)
    
    w[w > max_weight] = max_weight
    w[w < 0] = 0
    return w  # (10, 600,360)

def Mp_learning(x, y, teta_learn, weights, max_weight):  # Fs = learning(rnn, msn_layer, teta_hebb, Fs, max_weight)
    ## weights have shape (n_sample, 600, 360)
    ## max_weights have shape (10,1)
    lr = 0.05  # initial 0.01
    post = y-teta_learn   #shape(10,360)
    pre = x-teta_learn    #shape(10,600)
    post[post < 0] = 0
    # pre[pre < 0] = 0
    
    pre_reshape = pre[:,:,np.newaxis]         # (10,600,1)
    post_reshape = post[:,np.newaxis,:]       # (10,1,360)
    
    mul_pre_post = np.einsum('ijk,ikl->ijl', pre_reshape, post_reshape)  #(nb_participants, 600,300)
    
    w = (weights) - (0 * weights) + (lr * mul_pre_post)  # weights decay  (n_sample, 600, 360)
    
    w[w > max_weight] = max_weight
    w[w < -0.1] = -0.1
    return w  # (10, 600,360)


def decision(y, n_sample):     # if decision(H_window) == False:
                               #    stable_time = 0
    convergence = 0.1
    x = np.arange(1, 6, 1).reshape(1,5)
    x = np.repeat(x,n_sample,axis=0)
    slope = np.zeros((n_sample,1))
    for i in range(n_sample):
        slope[i,:] = linregress(x[i,:], y[i,:])[0]
    return np.abs(slope) < convergence


def update_window(H, H_window, n_sample):  # H_window = update_window(H, H_window) H(10,1) 
 
    # certainly a much smarter way to do this but I am tired :)
    window_size = 5
    new_H_window = np.zeros((n_sample, window_size))
    new_H_window[:, -1] = H[0]
    new_H_window[:, -2] = H_window[:, 4]
    new_H_window[:, -3] = H_window[:, 3]
    new_H_window[:, -4] = H_window[:, 2]
    new_H_window[:, -5] = H_window[:, 1]

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


def supervisory_signal(x, N_output, peak_width, peak_height):    # shape (32,360)

    vec = np.arange(0, N_output, 1).reshape(1,N_output)
    # VEC = np.repeat(vec, x.shape[0], axis=0)

    bell1 = norm.pdf(vec, x, peak_width)
    bell2 = norm.pdf(vec, x+N_output, peak_width)
    bell3 = norm.pdf(vec, x-N_output, peak_width)
    total_bell = bell1 + bell2 + bell3
    max_bell = np.max(total_bell, axis = 1, keepdims=True)
    bell = (total_bell/max_bell) * peak_height

    return bell.reshape(x.shape[0], N_output)
    
def Entro_non_lin(Entropy, threshold):
    Entropy_copy = copy.deepcopy(Entropy)
    Entropy_copy[Entropy_copy < threshold] = 0
    return Entropy_copy

def get_outcome(data_all, nb_participants, nb_trials):    
    # outcome shape (nb_participants, nb_trials)  
    # (32,119) when block is 1 or 3; (32,120) when block is 2 or 4

    outcome = np.zeros((nb_participants, nb_trials))   # nb_trials following the block no.
    
    for p in range(nb_participants):
        outcome[p] = data_all[p]

    return outcome
        
def get_nb_trials(current_block):
    if current_block+1 == 1 or current_block+1 == 3:
        return 119
    else:
        return 120
    
def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def compare_lists_and_return_indices(list1, list2):
    return [index for index, (item1, item2) in enumerate(zip(list1, list2)) if item1 > item2]


def compare_lists_and_return_indices(list1, number):
    return [index for index, item1 in enumerate(list1) if item1 >= number]

def generate_cov(chunk_size = 30, size = 600):
    beta = 1
    covariance = np.zeros((size, size))
    chunk_num = int(size/chunk_size)
    for i in range(chunk_num):
        for j in range(chunk_num):
            if i == j:
                covariance[i*chunk_size:(i+1)*chunk_size, j*chunk_size:(j+1)*chunk_size] = 1
            else:
                distance = abs(j - i)
                covariance[i*chunk_size:(i+1)*chunk_size, j*chunk_size:(j+1)*chunk_size] = np.exp(-beta*distance)

    return covariance

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
teta_hebb_Mp = 0.6  # threshold for hebb learning rule (motor to RNN)
learning_window = 500  # max time to switch attractor state for learning initial 500
a_noise_u = 0  # mean of output noise
a_noise_sd = 1  # standard deviation of output noise
rnn_noise_u = 0  # same for rnn
rnn_noise_sd = 2.4  # same for rnn
weight_values_msn = 1  # inhibitory weight values MSN layer
weight_values_gp = 1  # inhibitory weight values GP layer
weight_values_th = 1  # inhibitory weight values Th layer
self_excitation = 1
bias_inh = 3  # bias on the inhibitory neuron
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
cor_coeff = 0.05


# define cpt_rev ratio. 12 distributions in total. 1st dis is the base, 2nd dis has to be a change point. 
# we can define the following dis either be a cpt or rev start from 3rd dis
cpt_num = 0
rev_num = 10 - cpt_num

nb_participants = 100

rec_sim = []



npz_file = "real_outcome/real_outcome_noise_driven_reversal_cpt_rev_cptnum_{}.npz".format(cpt_num)
load_data = np.load(npz_file, allow_pickle=True)
data_all = load_data["arr_0"]   ## (nb_ppants, 100) (5, 120)

trial_num = 120

## end generation
# REC_RNN = []

each_dis_loop_time = 1
MODEL_RESPONSE = np.zeros((each_dis_loop_time, trial_num, nb_participants, 1))
for each_dis in range(each_dis_loop_time):
    patterns = patts(N_rnn, N_patterns)
    Ws = symmetric_conn(patterns, N_patterns, N_rnn)
    # aWs = asymmetric_conn_PFC_MSN()

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
    beta = 1.2

    # participants and blocks numbers
    # nb_participants = 5       ## number for get average value since this is a simulation test, NOT THE REAL NUMBER FOR HUMAN DATA
    nb_blocks = 1
    show_plots = 0
    verbose = 1

    # initialize rnn
    rnn = copy.deepcopy(patterns[0].reshape(1,600))
    rnn = np.repeat(rnn, nb_participants, axis=0)      # vectorize RNN

    # open record lists
    H_window = np.zeros((1, 5))

    # get the responses
    model_response = []

    #non-linear methods to allow for the expression of a single attractor at a time
    #method 1: RNN weight matrix is 0 in the offdiagonal space
    #method 2: non-linearity at zero in RNN unit activation 

    method = 2

    if method == 1:
        Ws[Ws<0] = 0
        print('Mehtod 1: Non linearity at zero on RNN weight values')
    elif method == 2:
        print('Method 2: Non linearity at zero on RNN activation units')

        
    # reset things
    outcomes  = []

    # extract the number of trials for each block
    nb_trials = trial_num
    Fs = feed_RNN_to_MSN(N_rnn, N_output, nb_participants)
    Mp = np.zeros((nb_participants, N_output, N_rnn))
    weights_rnn_inh = np.zeros((nb_participants, N_rnn))
    max_weight_Fs = 0.2*np.ones(((nb_participants, 1)))
    max_weight_Mp = 0.06*np.ones(((nb_participants, 1)))
    outcomes = get_outcome(data_all, nb_participants, nb_trials)

    # MAX_E = np.zeros((nb_participants, 1))

    chunk_size = 30
    mean = [0 for i in range(N_rnn)]
    cov_mtx = generate_cov(chunk_size, N_rnn)

    for t in range(nb_trials):  # nb_trials

        print(' // trial: ' + str(t))

        # reset all but RNN
        output_layer = np.zeros((nb_participants, N_output))
        msn_layer = np.zeros((nb_participants, N_output))
        gp_layer = np.ones((nb_participants, N_output)) * 2
        th_layer = np.zeros((nb_participants, N_output))
        rbar = np.zeros((nb_participants, N_rnn))
        inh_neuron = np.zeros((nb_participants, 1))
        H = np.zeros((nb_participants, 1))  # hopfield for decision
        H_rnn = 0  # hopfield for RNN
        Entropy = np.zeros((nb_participants, 1))  # multiplicative gain to noise in RNN
        count = np.zeros((nb_participants, 1))
        supervision_input = np.zeros((nb_participants, N_output))
        stable_time = np.zeros((nb_participants, 1))

        # open record lists
        H_window = np.zeros((1, 5))
        S_window = np.zeros((1, 5))
        
        rnn = np.tanh(beta * rnn)

        noise_ind = 0
        cor_noise = np.random.multivariate_normal(mean, cov_mtx, 2000 * nb_participants)  # sample 2000 
        

        # start of the decision dynamics
        while 1:
            # RNN dynamics
            # noise_rnn = np.random.normal(0, 0.05, (nb_participants, N_rnn))
            rnn += (-rnn + (np.matmul(rnn, Ws)) + Entropy * (np.einsum('ij,ijl->il', output_layer, Mp)) + cor_coeff * cor_noise[noise_ind : noise_ind+nb_participants,:]) / tau_rnn
            rnn = np.tanh(beta * rnn)
            if method == 2:
                rnn[rnn < 0] = 0
            
            # MSN dynamics
            noise_msn = np.random.normal(0, 0, N_output*nb_participants).reshape(nb_participants, N_output)
            msn_layer += (-msn_layer + 3*np.einsum('ij,ijl->il', rnn, Fs) + (output_layer * weight_msn_out) + noise_msn - bias_msn) / tau_msn
            msn_layer = np.tanh(msn_layer)
            msn_layer[msn_layer < 0] = 0

            # GP dynamics
            noise_gp = np.random.normal(0, 0, N_output*nb_participants).reshape(nb_participants, N_output)
            gp_layer += (gp_layer - (msn_layer *weight_Gp_Msn) + bias_gp + noise_gp) / tau_gp
            gp_layer = np.tanh(gp_layer)
            gp_layer[gp_layer < 0] = 0
            
            # Th dynamics
            noise_th = np.random.normal(0, 0.05, N_output*nb_participants).reshape(nb_participants, N_output)
            th_layer += (-k*th_layer - (gp_layer * weight_Th_Gp) + (output_layer * weight_Th_M) + bias_th + noise_th) / tau_th
            th_layer = np.tanh(th_layer)
            th_layer[th_layer < 0] = 0
            Entropy = (np.array([[0]]).astype(float)) * Entro_gain


            # output dynamics
            noise_output = np.random.normal(0, 0, N_output*nb_participants).reshape(nb_participants, N_output)
            output_layer += (-output_layer + np.matmul(output_layer, output_weights) + (th_layer * weight_M_Th) + supervision_input + noise_output) / tau_output
            output_layer = np.tanh(output_layer)
            output_layer[output_layer < 0] = 0

            # compute hopfield energy and update the energy window
            H = np.diagonal(hopfield_energy(output_layer, output_weights)).reshape(nb_participants,1)
            H_window = update_window(H, H_window, nb_participants)  # shape(n_sample,5)

            # reset stability clock if unstable
            # check if decision has been reached
            for i in range(nb_participants):
                if (count[i,:] > 25) and decision(H_window, nb_participants)[i,:] == True and np.mean(np.abs(H_window)[i,:]) > 3:
                    stable_time[i,:] += 1
                
            if np.all(stable_time > stable_max):

                test = np.argmax(output_layer, axis=1).reshape(nb_participants,1)
                model_response.append(np.argmax(output_layer, axis=1).reshape(nb_participants,1))
                if verbose:
                    print("LEIA prediction output: ")
                    print(np.argmax(output_layer, axis=1).reshape(nb_participants,1))
                    print(" ")
                break

            count += 1  # count allows the network to build in some activity
            noise_ind += nb_participants + 1


        # get overlap values between rnn and stored patterns
        init_rnn = copy.deepcopy(rnn)

        # reset count (for learning window)
        count = 0
            
        supervision_input = supervisory_signal(outcomes[:,t].reshape(nb_participants,1), N_output, supervisory_width, supervisory_height)
        print("real output:")
        print(outcomes[:,t].reshape(nb_participants,1))

        # start learning dynamics
        while 1:
            

            # hopfield energy and stability (window)
            H = np.diagonal(hopfield_energy(output_layer, output_weights)).reshape(nb_participants,1)
            H_window = update_window(H, H_window, nb_participants)

            # RNN dynamics
            # noise_rnn = np.random.normal(0, 0.05, (nb_participants, N_rnn))
            rnn += (-rnn + (np.matmul(rnn, Ws)) + Entropy * (np.einsum('ij,ijl->il', output_layer, Mp)) + cor_coeff * cor_noise[noise_ind : noise_ind+nb_participants,:]) / tau_rnn
            rnn = np.tanh(beta * rnn)
            if method == 2:
                rnn[rnn < 0] = 0
            

            # MSN dynamics
            noise_msn = np.random.normal(0, 0, N_output*nb_participants).reshape(nb_participants, N_output)
            msn_layer += (-msn_layer + 3*np.einsum('ij,ijl->il', rnn, Fs) + (output_layer * weight_msn_out) + noise_msn - bias_msn) / tau_msn
            msn_layer = np.tanh(msn_layer)
            msn_layer[msn_layer < 0] = 0

            # GP dynamics
            noise_gp = np.random.normal(0, 0, N_output*nb_participants).reshape(nb_participants, N_output)
            gp_layer += (gp_layer - (msn_layer * weight_Gp_Msn) + bias_gp +noise_gp) / tau_gp
            gp_layer = np.tanh(gp_layer)
            gp_layer[gp_layer < 0] = 0

            # Th dynamics
            noise_th = np.random.normal(0, 0, N_output*nb_participants).reshape(nb_participants, N_output)
            # th_layer += (-th_layer - (gp_layer * weight_Th_Gp) + (mem_layer * weight_Th_mem) + (output_layer * weight_Th_M) + noise_th) / tau_th
            th_layer += (-k*th_layer - (gp_layer * weight_Th_Gp) + (output_layer * weight_Th_M)+ bias_th + noise_th) / tau_th
            th_layer = np.tanh(th_layer)
            th_layer[th_layer < 0] = 0
            Normalize = th_layer / np.max(th_layer, axis=1, keepdims=True)
            Normalize = np.nan_to_num(Normalize, nan=0)
            log_norm = -np.log2(Normalize)
            log_norm[log_norm > 1000000] = 0
            sum_act = np.sum(th_layer, axis=1, keepdims=True)
            sum_act = np.maximum(sum_act, np.ones((sum_act.shape[0], sum_act.shape[1])))
            
            Entropy = np.diagonal(((np.dot(Normalize, log_norm.T)) / sum_act) * Entro_gain).reshape(nb_participants,1)
            Entropy = Entro_non_lin(Entropy, bias_inh)

            # output dynamics
            noise_output = np.random.normal(0, 0, N_output*nb_participants).reshape(nb_participants, N_output)
            output_layer += (-output_layer + np.matmul(output_layer, output_weights) + (th_layer * weight_M_Th) + (sup_gain*supervision_input) + noise_output) / tau_output
            output_layer = np.tanh(output_layer)
            output_layer[output_layer < 0] = 0


            if count > learning_window:  # 500

                if t > burnin:
    
                    Fs = learning(rnn, msn_layer, teta_hebb, Fs, 0.2)  # teta_hebb = 0.2
                    Mp = Mp_learning(output_layer, rnn, 0.6, Mp, 0.06)  

                    print("bias_inh: ", bias_inh)
                    print(" ")
                    
                if show_plots == 1:

                    # pass
                    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                    im = ax.imshow(Fs)
                    ax.set_title('RNN - MSN weights {}'.format(t), fontsize=25)
                    ax.set_xlabel('Receiving MSN Neurons', fontsize=20)
                    ax.set_ylabel('Sending RNN Neurons', fontsize=20)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    clb = fig.colorbar(im, cax=cax)
                    clb.set_label('weight values', labelpad=17,
                                fontsize=15, rotation=270)
                    plt.tight_layout()
                    plt.show()

                break

            count += 1
            noise_ind += nb_participants + 1
    MODEL_RESPONSE[each_dis] = np.array([model_response])


# rec_sim.append(np.mean(MODEL_RESPONSE, axis=0, keepdims=True))
rec_sim.append(MODEL_RESPONSE)
# DATA_REC.append(data_all)

# save the data
all_data_for_save = np.array(rec_sim)
np.savez("flxible_model_pred/noise_driven_reversal_cpt_rev_cptnum_{}".format(cpt_num), all_data_for_save)
