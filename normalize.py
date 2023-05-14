import numpy as np
import json

def normalize():
    #Find the max x cordinate and max y co-ordinate across all key value pairs.
    # Now,normalize it using min max normalization and subtract 1 from it 
    # to bring the values to scale of (-0.5 ,0.5)
    with open('/data2/home/harshg/Semester2_DLCV/Assignment3_DLCV/data/uptown_funk.json') as f:
        data = json.load(f)
    # print("loaded json keys: {}".format(data.keys()))
    # print("number of keys:{}".format(len(data.keys())))
    t_end = len(data['nose'])
    # print("Number of timesteps is :{}".format(t_end))
    xcurr_max = 0
    xcurr_min = 1e5
    ycurr_max = 0
    ycurr_min =1e5
    for key in data.keys():
        x = data[key]
        x = np.array(x)
        max_values = np.max(x, axis=0)
        min_values = np.min(x, axis=0)
        if xcurr_max < np.max(max_values[0]):
            xcurr_max = np.max(max_values[0])
        if xcurr_min > np.min(min_values[0]):
            xcurr_min = np.min(min_values[0])
        if ycurr_max < np.max(max_values[1]):
            ycurr_max = np.max(max_values[1])
        if ycurr_min > np.min(min_values[1]):
            ycurr_min = np.min(min_values[1])
    print(ycurr_max,ycurr_min)
    print(xcurr_max,xcurr_min)
    data_new = data
    for key in data_new.keys():
        val = data_new[key]
        for i in range(t_end):
            data_new[key][i] = [2*(val[i][0]-xcurr_min)/(xcurr_max-xcurr_min)-1,2*(val[i][1]-ycurr_min)/(ycurr_max-ycurr_min)-1]
    # print(data_new.keys())
    xcurr_max = 0
    xcurr_min = 1e5
    ycurr_max = 0
    ycurr_min =1e5
    for key in data_new.keys():
        x = data_new[key]
        x = np.array(x)
        max_values = np.max(x, axis=0)
        min_values = np.min(x, axis=0)
        if xcurr_max < np.max(max_values[0]):
            xcurr_max = np.max(max_values[0])
        if xcurr_min > np.min(min_values[0]):
            xcurr_min = np.min(min_values[0])
        if ycurr_max < np.max(max_values[1]):
            ycurr_max = np.max(max_values[1])
        if ycurr_min > np.min(min_values[1]):
            ycurr_min = np.min(min_values[1])
    print(ycurr_max,ycurr_min)
    print(xcurr_max,xcurr_min)
    with open('/data2/home/harshg/Semester2_DLCV/Assignment3_DLCV/data/normalized_uptown_funk.json', 'w') as f:
        json.dump(data_new, f)

    pass



if __name__=='__main__':
    normalize()