# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 16:56:47 2017

@author: jc9730; Jiada Chen; HW3_q2
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

def main():
    inp_path = sys.argv[1]
    out_path = sys.argv[2]
    data = []
    labels= []
    output = []    
    
    with open(inp_path,'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data.append(row[0:2])
            labels.append(row[2])
    
    data = np.array(data, dtype = float)
    labels = np.array(labels, dtype = float)
    n = len(data)
    
    # Scaling Features
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0,ddof=1)
    data = (data - mu)/std
    data = np.hstack((np.ones((n,1)),data))
    
    # GD algorithm with given alpha/iter
    alphas = [.001, .005, .01, .05, .1, .5, 1., 5., 10.]
    itr = 100
    
    for alpha in alphas:
        betas = np.zeros(data.shape[1])
        for i in range(itr):
            betas = betas - alpha * np.dot(data.T,np.dot(betas,data.T) - labels)/n
        output_row = [alpha]
        output_row.append(itr)
        output_row.extend(betas)
        output.append(output_row)
    
    # My own choice of alpha/iter    
    my_alpha = 0.75
    my_itr = 300
    betas = np.zeros(data.shape[1])
    for i in range(my_itr):
        betas = betas - my_alpha * np.dot(data.T,np.dot(betas,data.T) - labels)/n
    
    output_row = [my_alpha]
    output_row.append(my_itr)
    output_row.extend(betas)
    output.append(output_row)
    
    
    with open(out_path,'w') as f:
        writer = csv.writer(f,delimiter = ',')
        writer.writerows(output)
        
if __name__ == "__main__":
    main()