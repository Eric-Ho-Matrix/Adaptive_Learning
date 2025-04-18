#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Cris and Qin

This is the LEIA changepoint model with adaptive threshold, which we use asymetrical connectivity to push attractor state transition 
and perform rapid learning upon encountering a changepoint. 
"""

### importing libraries ###
from collections import Counter
from scipy.linalg import hadamard
import copy
from scipy import integrate
from scipy.sparse import random
from scipy.stats import linregress
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.io
import scipy as sc
import random as rd
import numpy as np
import os
from draw_plots_heat_map import Draw_plots
from matplotlib.gridspec import GridSpec
# plt.rcParams['font.family'] = 'Times New Roman'

### generate data

'''
This code illustrates a LEIA flexible model using simple data generation to demonstrate a typical helicopter task scenario.
In the following section, we define several changepoints and reversals, and for each context, 
we generate 10 trials with a specified noise level.
'''

u1 = 50
u2 = 150
u3 = 250
u4 = 150
u5 = 220
u6 = 50
u7 = 280
u8 = 50
u9 = 120
u10 = 270

sigma1  = 10
sigma2  = 10
sigma3  = 10
sigma4  = 10
sigma5  = 10
sigma6  = 10
sigma7  = 10
sigma8  = 10
sigma9  = 10
sigma10 = 10

num1  = 5
num2  = 5
num3  = 5
num4  = 5
num5  = 5
num6  = 5
num7  = 5
num8  = 5
num9  = 5
num10 = 5

total_trials = 0

for i in range(1, 11):
    total_trials += locals()['num' + str(i)]

data1 = np.random.normal(u1, sigma1, (1, num1))
data2 = np.random.normal(u2, sigma2, (1, num2))
data3 = np.random.normal(u3, sigma3, (1, num3))
# data3 = copy.deepcopy(data1)
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

num12  = num1  + num2
num13  = num12 + num3
num14  = num13 + num4
num15  = num14 + num5
num16  = num15 + num6
num17  = num16 + num7
num18  = num17 + num8
num19  = num18 + num9
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
### end generate data


### We start by defining all the functions needed for simulations ###q
### Starting with connectivity matrices for all layers of the network ###

# output layer connectivity (ring-like, center-surround)
def output_connectivity(N_output, baseline, peak_height, peak_width):
    """
    Constructs a bump attractor dynamics for the output layer. 
    This matrix defines the strength of connections between neurons based on 
    a Gaussian-like bell curve centered around each neuron, with periodic boundary conditions.

    Parameters:
    -----------
    N_output : int
        The number of neurons in the output layer.
    baseline : float
        The baseline connectivity value for all neurons.
    peak_height : float
        The maximum height of the Gaussian bell curve, representing the strongest connection.
    peak_width : float
        The standard deviation of the Gaussian bell curve, controlling the spread of connectivity.

    Returns:
    --------
    numpy.ndarray
        A 2D connectivity matrix of shape (N_output, N_output) encoding self-recurrent (bump attractor) dynamics in output layer.
    """
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
    """
    Constructs a identity topographical connectivity matrix for the striatum to GP and TH layers. 
    This matrix defines inhibitory connections between neurons , with self-excitation along the diagonal.
    This function define the double inhibition mechanism in the network.

    Parameters:
    -----------
    N_output : int
        The number of neurons in the output layer (e.g., GP or TH layer).
    weight_values : float
        The inhibitory weight value for the off-diagonal elements of the matrix.
    self_excitation : float
        The self-excitation value for the diagonal elements of the matrix.

    Returns:
    --------
    numpy.ndarray
        A 2D connectivity matrix of shape (N_output, N_output). The diagonal elements 
        represent self-excitation, while the off-diagonal elements represent inhibitory 
        connections between neurons.
    """

    conn = np.ones((N_output, N_output)) * weight_values * (-1)
    np.fill_diagonal(conn, self_excitation)

    return conn

# RNN connectivity (encoding patterns/attractors)
def symmetric_conn(x, N_patterns, N_rnn):
    """
    Constructs a symmetric connectivity matrix for an RNN (Recurrent Neural Network) 
    with orthogonal attractors. This function ensures that the attractors are encoded 
    in an orthogonal manner, which is useful for stable pattern storage and retrieval.

    Parameters:
    -----------
    x : numpy.ndarray
        A 2D array of shape (N_patterns, N_rnn) where each row represents a pattern 
        to be encoded in the RNN. Each pattern should ideally be orthogonal to others.
    N_patterns : int
        The number of patterns to be encoded in the RNN.
    N_rnn : int
        The number of neurons in the RNN.

    Returns:
    --------
    numpy.ndarray
        A symmetric connectivity matrix of shape (N_rnn, N_rnn). The diagonal 
        elements represent self-connections, and the off-diagonal elements represent 
        the connections between different neurons. The matrix is scaled by a force 
        factor to control the overall strength of the connections.
    """
    Force = 8
    weights = np.zeros((N_rnn, N_rnn))
    for u in range(N_patterns):
        weights += np.dot(x[u, :].reshape(1, N_rnn).T, x[u, :].reshape(1, N_rnn))
    weights -= 3.5 * np.ones((N_rnn, N_rnn))  # -2.9 # off-diag value is defined here  -1.03

    return weights*(Force/N_rnn)


# RNN to MSN connectivity
def feed_RNN_to_MSN(N_rnn, N_output):
    """
    Generates a feedforward weight matrix connecting RNN to striatum in the network. 
    The weights are initialized with random values drawn from a normal distribution.

    Parameters:
    -----------
    N_rnn : int
        The number of neurons in the RNN layer (600 in our case).
    N_output : int
        The number of neurons in the striatum layer (360 in our case).

    Returns:
    --------
    numpy.ndarray
        A 2D weight matrix of shape (N_rnn, N_output). Each element represents the 
        connection strength between an RNN neuron and an striatum neuron.
    """
    g_u = 0.5
    g_sd = 0.1
    ffw = np.random.normal(g_u/N_rnn, g_sd/N_rnn, (N_rnn, N_output))

    return ffw


### we now write all network related functions ###
# threshold function for entropy induced inhibition in the memory integration layer
def thresh(x, limit):
    """
    Threshold function for entropy-induced inhibition in the memory integration layer.

    Parameters:
    -----------
    x : float
        Input value (e.g., entropy).
    limit : float
        Threshold value.

    Returns:
    --------
    float
        Returns `x` if it is greater than or equal to `limit`, otherwise returns 0.
    """
    if x >= limit:
        j = x
    elif x < limit:
        j = 0
    return j

# function to build patterns
def patts(N_rnn, N_patterns):
    """
    Generates a set of patterns to be encoded in an RNN. Each pattern is represented as a row in a 2D array, 
    where specific groups of neurons (every 30 neurons) are activated with a fixed value (e.g., 2.8).
    These patterns are used to define attractor states in the RNN.

    Parameters:
    -----------
    N_rnn : int
        The total number of neurons in the RNN.
    N_patterns : int
        The number of distinct patterns to generate.

    Returns:
    --------
    numpy.ndarray
        A 2D array of shape (N_patterns, N_rnn), where each row represents a pattern.
    """
    patt = np.zeros((N_patterns, N_rnn))
    for i in range(N_patterns):
        basic = np.zeros((1, N_rnn))
        basic[0, i*30:i*30+30] = 3   ## 3 #diag value is defined here
        patt[i] = basic[0]
    return patt

# function to compute hopfield energy
def hopfield_energy(x, w):
    """
    Computes the Hopfield energy for a given state and weight matrix.

    Parameters:
    -----------
    x : numpy.ndarray
        The state vector of the network (e.g., neuron activations).
    w : numpy.ndarray
        The weight matrix representing connections between neurons.

    Returns:
    --------
    float
        The Hopfield energy, a scalar value representing the system's energy.
    """
    return -np.dot(x, np.dot(w, x.T))

# defining the thresholded Hebbian learning rule
def learning(x, y, teta_learn, weights, max_weight):
    """
    Updates the weights between PFC-striatum using a Hebbian-like learning rule.

    Parameters:
    -----------
    x : numpy.ndarray
        Pre-synaptic neuron activations.
    y : numpy.ndarray
        Post-synaptic neuron activations.
    teta_learn : float
        Learning threshold for neuron activations.
    weights : numpy.ndarray
        Current weight matrix to be updated.
    max_weight : float
        Maximum allowable weight value.

    Returns:
    --------
    numpy.ndarray
        Updated weight matrix after applying the learning rule.
    """

    lr = 0.01  # initial 0.01
    post = y-teta_learn
    pre = x-teta_learn
    post[post < 0] = 0
    pre[pre < 0] = 0
    w = (weights) - (0 * weights) + (lr * (np.matmul(pre.T, post)))  # weights decay
    w[w > max_weight] = max_weight
    w[w < 0] = 0
    return w

# defining the thresholded Hebbian learning rule
def learning_MP(x, y, teta_learn, weights, max_weight):  #x output_layer #y RNN
    """
    Updates the weights between motor cortex to PFC using a Hebbian-like learning rule.
    Notice we set threshold higher to enable 

    Parameters:
    -----------
    x : numpy.ndarray
        Pre-synaptic neuron activations.
    y : numpy.ndarray
        Post-synaptic neuron activations.
    teta_learn : float
        Learning threshold for neuron activations.
    weights : numpy.ndarray
        Current weight matrix to be updated.
    max_weight : float
        Maximum allowable weight value.

    Returns:
    --------
    numpy.ndarray
        Updated weight matrix after applying the learning rule.
    """
    lr = 0.01  # initial 0.01
    post = y-teta_learn
    pre = x-teta_learn
    # post[post < 0] = 0
    # pre[pre < 0] = 0
    w = (weights) - (0 * weights) + (lr * (np.matmul(pre.T, post)))  # weights decay
    w[w > max_weight] = max_weight
    # w[w < 0] = 0
    return w

# change point oddball, change the decay rate.
# max_weight
# divided
# keep weights update
# removing decay

# decision function (check if decision is made through output layer stability)
def decision(y):
    """
    Determines if the output layer has reached stability based on the slope of recent activity.

    Parameters:
    -----------
    y : numpy.ndarray
        A 1D array representing recent activity values of the output layer.

    Returns:
    --------
    bool
        True if the absolute slope of the activity is below the convergence threshold, indicating stability.
    """
    convergence = 0.1
    x = np.arange(1, 6, 1)
    slope = linregress(x, y)[0]
    return np.abs(slope) < convergence


def update_window(H, H_window):
    """
    Updates a sliding window with the latest Hopfield energy value.

    Parameters:
    -----------
    H : float
        The latest Hopfield energy value.
    H_window : numpy.ndarray
        A 1D array representing the current sliding window of energy values.

    Returns:
    --------
    numpy.ndarray
        A 1D array with the updated sliding window, where the newest value replaces the oldest.
    """

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
    """
    Updates a sliding window with the latest stability value.

    Parameters:
    -----------
    S : float
        The latest stability value.
    S_window : numpy.ndarray
        A 1D array representing the current sliding window of stability values.

    Returns:
    --------
    numpy.ndarray
        A 1D array with the updated sliding window, where the newest value replaces the oldest.
    """
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
    """
    Generates a Gaussian-like supervisory signal for the output layer.

    Parameters:
    -----------
    x : int
        The index of the target neuron (correspond to observation in the helicopter task) to center the signal.
    N_output : int
        The total number of neurons in the output layer.
    peak_width : float
        The standard deviation of the Gaussian curve.
    peak_height : float
        The maximum height of the Gaussian curve.

    Returns:
    --------
    numpy.ndarray
        A 1D array of shape (1, N_output) representing the supervisory signal.
    """

    vec = np.arange(0, N_output, 1)
    bell1 = norm.pdf(vec, x, peak_width)
    bell2 = norm.pdf(vec, x+N_output, peak_width)
    bell3 = norm.pdf(vec, x-N_output, peak_width)
    total_bell = bell1 + bell2 + bell3
    max_bell = max(total_bell)
    bell = (total_bell/max_bell) * peak_height

    return bell.reshape(1, N_output)

def Entro_non_lin(Entropy, threshold):
    """
    Computes the entropy of a system and checks if it exceeds a given threshold. 
    """

    marker = 0
    if Entropy >= threshold:
        marker = 1
        return Entropy, marker
    else:
        return 0, marker

def generate_cov(chunk_size = 30, size = 600):
    """
    Generates a covariance matrix that correlates to the attractor state. Use this 
    covariance to generate noise that drive the network to switch
    """
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
teta_hebb = 0.2  # threshold for hebb learning rule  0.2 original
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
bias_rnn = 1.5
alpha = 0.15
cor_coeff = 0.35
bias_rev = 0

patterns = patts(N_rnn, N_patterns)
Ws = symmetric_conn(patterns, N_patterns, N_rnn)

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


# initialize rnn
rnn= copy.deepcopy(patterns[0].reshape(1,600))

# open record lists
H_window = np.zeros((1, 5))

# get the responses
model_response = []

# participants and blocks numbers
nb_participants = 1
nb_blocks       = 1
show_plots = 1

rec_sim = []
LEIA_Pred = []


method = 2

if method == 1:
    Ws[Ws<0] = 0
    print('Mehtod 1: Non linearity at zero on RNN weight values')
elif method == 2:
    print('Method 2: Non linearity at zero on RNN activation units')

# rnn-msn weights refresh count
attar_cnt = 0 

# sim loop
for p in range(1):  # nb_participants

    model_response = []

    for b in range(nb_blocks):  # nb_blocks
        
        # reset things
        mem_layer = np.zeros((1, N_output))
        outcomes  = []

        # extract the number of trials for each block
        nb_trials = 360
        Fs = feed_RNN_to_MSN(N_rnn, N_output)
        Mp = np.zeros((N_output, N_rnn))
        weights_rnn_inh = np.zeros((1, N_rnn))
        max_weight = 0.2
        outcomes = np.maximum(data110,0)
        means    = mean_generate

        chunk_size = 30
        mean = [0 for i in range(N_rnn)]
        cov_mtx = generate_cov(chunk_size, N_rnn)

        USE_MARK = []

        for t in range(total_trials):  # nb_trials

            print('block: ' + str(b) + ' // trial: ' + str(t))

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

            rec_rnn          = []
            rec_msn          = []
            rec_mem          = []
            rec_E            = []
            rec_inh          = []
            rec_th           = []
            rec_output       = []
            rec_GP           = []
            rec_pseudo_state = []

            # open record lists
            H_window = np.zeros((1, 5))
            S_window = np.zeros((1, 5))
            
            rnn = np.tanh(beta * rnn)

            # for every trial, we need to sample noise for all time steps at once
            # it is extremely slow to sample noise for every time step
            noise_ind = 0
            cor_noise = np.random.multivariate_normal(mean, cov_mtx, 2000)  # sample 2000

            # start of the decision dynamics
            while 1:
                
                # RNN dynamics
                rnn += (-rnn + (np.matmul(rnn, Ws)) + Entropy * (np.matmul(output_layer, Mp)) + cor_coeff * cor_noise[noise_ind,:]) / tau_rnn
                rnn = np.tanh(beta * rnn)
                if method == 2: 
                    rnn[rnn < 0] = 0
                
                current_state = copy.deepcopy(rnn)

                # low pass filter for inhibitory weights
                rbar += (-rbar + rnn)/tau_bar

                # MSN dynamics
                noise_msn = np.random.normal(0, 0, N_output).reshape(1, N_output)
                msn_layer += (-msn_layer + 3*np.matmul(rnn, Fs) + (output_layer * weight_msn_out) + noise_msn - bias_msn) / tau_msn
                msn_layer = np.tanh(msn_layer)
                msn_layer[msn_layer < 0] = 0

                # GP dynamics
                noise_gp = np.random.normal(0, 0, N_output).reshape(1, N_output)
                gp_layer += (gp_layer - (msn_layer *weight_Gp_Msn) + bias_gp + noise_gp) / tau_gp
                gp_layer = np.tanh(gp_layer)
                gp_layer[gp_layer < 0] = 0
                
                # Th dynamics
                noise_th = np.random.normal(0, 0.05, N_output).reshape(1, N_output)
                th_layer += (-k*th_layer - (gp_layer * weight_Th_Gp) + (output_layer * weight_Th_M) + bias_th + noise_th) / tau_th
                th_layer = np.tanh(th_layer)
                th_layer[th_layer < 0] = 0
                Entropy = (np.array([[0]]).astype(float)) * Entro_gain

                # output dynamics
                noise_output = np.random.normal(0, 0, N_output).reshape(1, N_output)
                output_layer += (-output_layer + np.matmul(output_layer, output_weights) + (th_layer * weight_M_Th) + supervision_input + noise_output) / tau_output
                output_layer = np.tanh(output_layer)
                output_layer[output_layer < 0] = 0

                # Inhibitory neuron dynamics
                inh_neuron += ((-leak_I * inh_neuron) +(Entropy * weight_I_Th) - bias_inh) / tau_inh
                inh_neuron[inh_neuron < 0] = 0
                inh_neuron[inh_neuron > max_inh] = max_inh

                # update inhibtory weights
                weights_rnn_inh += ((W_inh_max - weights_rnn_inh) * ((rbar-teta) * inh_neuron)) / tau_w_inh
                weights_rnn_inh[weights_rnn_inh < 0] = 0

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

                        model_response.append(np.argmax(output_layer))
                        print("LEIA prediction output: ")
                        print(np.argmax(output_layer))
                        LEIA_Pred.append(np.argmax(output_layer))
                        print(" ")
                        break
                    

                
                rec_mem.append(copy.deepcopy(mem_layer))
                rec_E.append(copy.deepcopy(Entropy))
                rec_inh.append(copy.deepcopy(inh_neuron))
                rec_rnn.append(copy.deepcopy(rnn))
                rec_msn.append(copy.deepcopy(msn_layer))
                rec_th.append(copy.deepcopy(th_layer))
                rec_output.append(copy.deepcopy(output_layer))
                rec_GP.append(copy.deepcopy(gp_layer))
                rec_pseudo_state.append(copy.deepcopy(np.matmul(output_layer, Mp)))

                count += 1  # count allows the network to build in some activity
                noise_ind += 1

            # get overlap values between rnn and stored patterns
            init_rnn = copy.deepcopy(rnn)

            # reset count (for learning window)
            count = 0

            # activation of output node
            supervision_input = supervisory_signal(outcomes[t], N_output, supervisory_width, supervisory_height)
            print("real output:")
            print(outcomes[t])
            
            cache_prev_rnn = copy.deepcopy(rnn)

            # start learning dynamics
            while 1:

                # hopfield energy and stability (window)
                H = hopfield_energy(output_layer, output_weights)[0][0]
                H_window = update_window(H, H_window)

                rnn += (-rnn + (np.matmul(rnn, Ws)) + Entropy * (np.matmul(output_layer, Mp)) + cor_coeff * cor_noise[noise_ind,:]) / tau_rnn

                rnn = np.tanh(beta * rnn)
                if method == 2:
                    rnn[rnn < 0] = 0

                # MSN dynamics
                noise_msn = np.random.normal(0, 0, N_output).reshape(1, N_output)
                # msn_layer += (-msn_layer + np.matmul(rnn, Fs) + np.matmul(msn_layer, MSN_weights) + (output_layer * weight_msn_out) + noise_msn) / tau_msn
                msn_layer += (-msn_layer + 3*np.matmul(rnn, Fs) + (output_layer * weight_msn_out) + noise_msn - bias_msn) / tau_msn
                msn_layer = np.tanh(msn_layer)
                msn_layer[msn_layer < 0] = 0

                # GP dynamics
                noise_gp = np.random.normal(0, 0, N_output).reshape(1, N_output)
                gp_layer += (gp_layer - (msn_layer * weight_Gp_Msn) + bias_gp +noise_gp) / tau_gp
                gp_layer = np.tanh(gp_layer)
                gp_layer[gp_layer < 0] = 0

                # Th dynamics
                noise_th = np.random.normal(0, 0, N_output).reshape(1, N_output)
                # th_layer += (-th_layer - (gp_layer * weight_Th_Gp) + (mem_layer * weight_Th_mem) + (output_layer * weight_Th_M) + noise_th) / tau_th
                th_layer += (-k*th_layer - (gp_layer * weight_Th_Gp) + (output_layer * weight_Th_M)+ bias_th + noise_th) / tau_th
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

                Entropy, use_mark = Entro_non_lin(Entropy, bias_inh)
                USE_MARK.append(use_mark)

                # output dynamics
                noise_output = np.random.normal(0, 0, N_output).reshape(1, N_output)
                output_layer += (-output_layer + np.matmul(output_layer, output_weights) + (th_layer * weight_M_Th) + (sup_gain*supervision_input) + noise_output) / tau_output
                output_layer = np.tanh(output_layer)
                output_layer[output_layer < 0] = 0

                # Inhibitory neuron dynamics
                inh_neuron += ((-leak_I * inh_neuron) + (Entropy * weight_I_Th) - bias_inh) / tau_inh
                inh_neuron[inh_neuron < 0] = 0
                inh_neuron[inh_neuron > max_inh] = max_inh

                # update inhibtory weights
                weights_rnn_inh += ((W_inh_max - weights_rnn_inh) * ((rbar-teta) * inh_neuron)) / tau_w_inh
                weights_rnn_inh[weights_rnn_inh < 0] = 0

                if count > learning_window:  # 500

                    if t > burnin:
                        if attar_cnt >= 19:
                            attar_cnt = 0 ## refresh counter
                            Fs = feed_RNN_to_MSN(N_rnn, N_output)
                        Fs = learning(rnn, msn_layer, teta_hebb, Fs, max_weight)  # teta_hebb = 0.2
                        Mp = learning_MP(output_layer, rnn, teta_hebb, Mp, 0.05) #0.05
                        

                    if show_plots == 1:

                        # pass
                        # fig, ax = plt.subplots(1, 1, figsize=(10, 2))
                        fig, ax = plt.subplots(1, 1)
                        im = ax.imshow(Fs)
                        ax.set_title('RNN - MSN weights {}'.format(t), fontsize=25)
                        ax.tick_params(axis='x', labelsize=20)
                        ax.tick_params(axis='y', labelsize=20)
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        clb = fig.colorbar(im, cax=cax)
                        clb.ax.tick_params(labelsize=20)
                        plt.tight_layout()
                        plt.show()

                    break

                count += 1
                noise_ind += 1

                rec_mem.append(copy.deepcopy(mem_layer))
                rec_E.append(copy.deepcopy(Entropy))
                rec_inh.append(copy.deepcopy(inh_neuron))
                rec_rnn.append(copy.deepcopy(rnn))
                rec_msn.append(copy.deepcopy(msn_layer))
                rec_th.append(copy.deepcopy(th_layer))
                rec_output.append(copy.deepcopy(output_layer))
                rec_GP.append(copy.deepcopy(gp_layer))  
                rec_pseudo_state.append(copy.deepcopy(np.matmul(output_layer, Mp)))

            if show_plots == 1:
                fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                im = ax.imshow(Mp)
                ax.set_title('M1 - PFC weights', fontsize=25)
                ax.set_xlabel('Receiving PFC Neurons', fontsize=20)
                ax.set_ylabel('Sending M1 Neurons', fontsize=20)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                clb = fig.colorbar(im, cax=cax)
                clb.set_label('weight values', labelpad=17,
                                fontsize=15, rotation=270)
                plt.tight_layout()
                plt.show()
                
                Draw_plots(N_rnn   = N_rnn, 
                        rec_rnn = rec_rnn, 
                        # rec_msn = rec_msn, 
                        # rec_GP  = rec_GP, 
                        # rec_th  = rec_th, 
                        # rec_inh = rec_inh , 
                        rec_E   = rec_E, 
                        # rec_mem = rec_mem, 
                        rec_output = rec_output, 
                        # means   = means, 
                        trial = t, 
                        # test = test,
                        # show_heat_map = 1,
                        combine_flag=None)
                
                plot_pseudo_rnn = np.zeros((len(rec_pseudo_state), N_rnn))
                for i in range(len(rec_pseudo_state)):
                    plot_pseudo_rnn[i,:] = rec_pseudo_state[i][0].reshape(1,N_rnn)
                x_time = np.arange(0,len(rec_pseudo_state),1)
                fig, ax = plt.subplots(1,1,figsize=(12,5))
                axins = inset_axes(ax,
                                    width="5%",  # width = 5% of parent_bbox width
                                    height="100%",  # height : 50%
                                    loc='lower left',
                                    bbox_to_anchor=(1.01, 0., 1, 1),
                                    bbox_transform=ax.transAxes,
                                    borderpad=0)
                ax.set_xlabel('time (a.u.)', fontsize=30)
                ax.set_ylabel('Units', fontsize=30)
                ax.tick_params(axis='y', labelsize=25)
                ax.tick_params(axis='x', labelsize=25)
                ax.set_title('pseudo_state dynamics', fontsize=30, pad=20)
                im = ax.imshow(plot_pseudo_rnn.T, cmap='plasma', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
                ax.autoscale(False)
                cbar = fig.colorbar(im, cax=axins)
                cbar.ax.tick_params(labelsize=20)
                fig.tight_layout()  # otherwise the right y-label is slightly clipped
                plt.show()
                

    rec_sim.append(model_response)

fig = plt.figure(figsize=(15,8))
plt.plot(outcomes, 'o', color="grey", markersize=10, label='True hidden state')
plt.plot(LEIA_Pred, '-', linewidth=7, color='purple', label='LEIA model')
plt.title('LEIA Model Reversal Learning Task', fontsize=20)
plt.ylabel('position', fontsize=20)
plt.xlabel('Trial number', fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.legend(loc='upper right', fontsize=10)
fig.tight_layout()
plt.show()
