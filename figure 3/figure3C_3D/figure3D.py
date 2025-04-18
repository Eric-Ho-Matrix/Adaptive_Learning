import numpy as np
import scipy as sc
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
# from get_beliefs_2 import *   
import copy
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
    # intercept = model.intercept_

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


### get the correlation plot

# load data from matlab code (need to run matlab code first)
PE = np.loadtxt('PE_coeff.csv', delimiter=',')
CPP = np.loadtxt('CPP_coeff.csv', delimiter=',')
RU = np.loadtxt('RU_coeff.csv', delimiter=',')

# Generate sample data
x = np.array(Best_threshold_each_ppant)

oorange = "#8856a7"
bblue = "#7fcdbb"
rred = "#f03b20"


# Create a scatter plot with regression line and confidence interval
sns.regplot(x=x, y=PE, ci=95, scatter_kws={"color": rred}, line_kws={"color": rred}, label=r'$\beta_1$')
sns.regplot(x=x, y=CPP, ci=95, scatter_kws={"color": bblue}, line_kws={"color": bblue}, label=r'$\beta_2$')
sns.regplot(x=x, y=RU, ci=95, scatter_kws={"color": oorange}, line_kws={"color": oorange}, label=r'$\beta_3$')

# Customize plot

plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.xticks([1, 2, 3, 4, 5, 6])
plt.tick_params(axis='both', labelsize=20)

# Add legend
plt.legend(loc = 'upper left')

# Show the plot
plt.savefig('linear_regression_human_pe.svg', dpi=500, format='svg',bbox_inches='tight')
plt.show()


