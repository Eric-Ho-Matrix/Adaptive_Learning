
import copy
import random
import numpy as np

def reset_reversal_dis(rever_num):
    ## pick up #rever_num reversal points from index num3-num12
    rever_index = random.sample(range(2, 12), rever_num)
    rever_index.sort()
    return rever_index

def generate_data(lower_bound, upper_bound):
    mean_list = []

    for _ in range(12):
        u = random.randint(lower_bound, upper_bound)
        mean_list.append(u)

    return mean_list



nb_participants = 100  ## 10 here means for each update error, we use batch_size as 10 to get mean value

for cpt_num in range(10, -1, -1):

    rev_num = 10 - cpt_num

    print("cpt_num: ", cpt_num)

    data_temp = []

    MEAN_LIST = np.zeros((1, 12))


    for i in range(nb_participants):
        rev_index = reset_reversal_dis(rev_num)

        ## data generation
        sigma = 10  # std

        # bound for gaussian mean values
        lower_bound = 30
        upper_bound = 300
        
        if cpt_num == 10:
            mean_list = generate_data(lower_bound, upper_bound)
        else:
            npz_file = "mean_list.npz"
            load_data = np.load(npz_file, allow_pickle=True)
            mean_list = load_data["arr_0"][i]   ## (nb_ppants, 100) (5, 120)

        reversal_cache = []

        if len(rev_index) > 0:
            for i in range(len(mean_list)):
                if i in rev_index:
                    # 1. if previous one is also a reversal point, then we need to pick up another one
                    if i-1 in rev_index:
                        rmv_value = mean_list[i-1]
                        templist = copy.deepcopy(reversal_cache)
                        templist = [item for item in templist if item != rmv_value] # remove the value that equals to mean_list[i-1]
                        mean_list[i] = random.choice(templist)
                    # 2. if previous one is not a reversal point, then we can pick up from the cache[:-1]
                    else:
                        mean_list[i] = random.choice(reversal_cache[:-1])
                else:
                    reversal_cache.append(mean_list[i])

        MEAN_LIST = np.concatenate([MEAN_LIST, np.array(mean_list).reshape(1, 12)], axis=0)

        num1  = 10
        num2  = 10
        num3  = 10
        num4  = 10
        num5  = 10
        num6  = 10
        num7  = 10
        num8  = 10
        num9  = 10
        num10 = 10
        num11 = 10
        num12 = 10

        data1  = np.random.normal(mean_list[0],  sigma,  (1, num1))
        data2  = np.random.normal(mean_list[1],  sigma,  (1, num2))
        data3  = np.random.normal(mean_list[2],  sigma,  (1, num3))
        data4  = np.random.normal(mean_list[3],  sigma,  (1, num4))
        data5  = np.random.normal(mean_list[4],  sigma,  (1, num5))
        data6  = np.random.normal(mean_list[5],  sigma,  (1, num6))
        data7  = np.random.normal(mean_list[6],  sigma,  (1, num7))
        data8  = np.random.normal(mean_list[7],  sigma,  (1, num8))
        data9  = np.random.normal(mean_list[8],  sigma,  (1, num9))
        data10 = np.random.normal(mean_list[9],  sigma,  (1, num10))
        data11 = np.random.normal(mean_list[10], sigma,  (1, num11))
        data12 = np.random.normal(mean_list[11], sigma,  (1, num12))


        data1  = np.maximum(data1, 0)
        data2  = np.maximum(data2, 0)
        data3  = np.maximum(data3, 0)
        data4  = np.maximum(data4, 0)
        data5  = np.maximum(data5, 0)
        data6  = np.maximum(data6, 0)
        data7  = np.maximum(data7, 0)
        data8  = np.maximum(data8, 0)
        data9  = np.maximum(data9, 0)
        data10 = np.maximum(data10, 0)
        data11 = np.maximum(data11, 0)
        data12 = np.maximum(data12, 0)

        data_temp.append(np.concatenate([ data1, 
                                data2, 
                                data3, 
                                data4, 
                                data5,  
                                data6, 
                                data7, 
                                data8, 
                                data9, 
                                data10,
                                data11,
                                data12], axis=1))
        
    data_all = np.concatenate(data_temp, axis=0)

    trial_num = num1+num2+num3+num4+num5+num6+num7+num8+num9+num10+num11+num12

    MEAN_LIST = np.delete(MEAN_LIST, 0, axis=0)

    if cpt_num == 10:
        np.savez("mean_list", MEAN_LIST)

    np.savez("real_outcome/real_outcome_noise_driven_reversal_cpt_rev_cptnum_{}".format(cpt_num), data_all)
    np.savez("Mean_list/Mean_list_noise_driven_reversal_cpt_rev_cptnum_{}".format(cpt_num), MEAN_LIST)