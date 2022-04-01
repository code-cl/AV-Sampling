# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 16:18:04 2021

@author: Tangmei
"""

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import random
import gpytorch
# import utils.Shapley_utils as F
import copy

matplotlib.rcParams['backend'] = 'SVG'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.rc('font', family='arial', size=16)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def predict(self, x):
        x = torch.tensor(x).to(torch.float)
        return  self(x).mean

    
    def fit(self):
        # Find optimal model hyperparameters
        self.train()
        self.likelihood.train()
        
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        
        training_iter = 100
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()
            # print(loss)
        self.eval()
        self.likelihood.eval()
   

def evaluate_results(model, number_initial_points= 10, num_plot_markers=20, num_interval_points=1,directory='.', task='Classification'): #
    '''
    Plots the performance for removing and adding
    Args:
        vals: A list of different valuations of data points each
             in the format of an array in the same length of the data.
        config: to remove or add, RH = remove high value, RL =  remove
            low value, AH = add high value, AL = add low value
    Result = sio.loadmat('./temp/shapley_result.mat')
    X = Result['X']
    y = Result['y']
    V = Result['Value']
            
    '''
    # sum the marginal performance to get the final shapley value
    Result = sio.loadmat(directory+'/shapley_result.mat')
    X = Result['X']
    mem_tmc = Result['Value']
    
    Value_shapley_list = ((np.cumsum(mem_tmc, 0)/np.reshape(np.arange(1, len(mem_tmc)+1), (-1,1))))
    val_shapley = Value_shapley_list[-1]
    
    low_val_index = np.argsort(val_shapley) 
    high_val_index = low_val_index[::-1] # max to min
    plot_points = np.arange(0, num_plot_markers, num_interval_points)
    
    plot_points = plot_points + number_initial_points
    
    #print('Creating Random index ...')
    print('Evaluate the performance of Random sampling ...')
    RD_sample_index_lists = [ random.sample(list(low_val_index), i)  for i in plot_points]
    RD_scores = [evaluate_performance(model, sample_index.copy(), directory) for sample_index in RD_sample_index_lists]
    RD_scores = np.array(RD_scores).reshape(-1)[np.newaxis,:]
    
    # Repeat 5 times
    for i in range(5):
        #print('Creating random index ...', i)
        RD_sample_index_lists = [ random.sample(list(low_val_index), i)  for i in plot_points]
        RD_scores_temp = [evaluate_performance(model, sample_index.copy(), directory) for sample_index in RD_sample_index_lists]
        RD_scores = np.concatenate([RD_scores, np.array(RD_scores_temp).reshape(-1)[np.newaxis,:]])
        
    # Sampling by maximising the designed aggregation value
    #print('Creating HighAV index ...')
    print('Evaluate the performance of the proposed HighAV sampling ...')
    HighAV_sample_index_lists = HighAV_Sample(X, val_shapley, plot_points, directory)
    HighAV_scores = [evaluate_performance(model, sample_index.copy(), directory) for sample_index in HighAV_sample_index_lists]
    
    # Sampling by minimising the designed aggregation value
    #print('Creating LowAV index ...')
    print('Evaluate the performance of the proposed LowAV sampling ...')
    LowAV_sample_index_lists = LowAV_Sample(X, val_shapley, plot_points)
    LowAV_scores = [evaluate_performance(model, sample_index.copy(), directory) for sample_index in LowAV_sample_index_lists]
     
    # Sampling by k-means cluster
    # print('Creating Cluster index ...')
    print('Evaluate the performance of Cluster sampling ...')
    Cluster_sample_index_lists = [ Cluster_Sample(X, i) for i in plot_points]
    Cluster_scores = [evaluate_performance(model, sample_index.copy(), directory) for sample_index in Cluster_sample_index_lists]
    
    # Sampling by maximising the sum of data Shapley value
    #print('Creating HighSV index ...')
    print('Evaluate the performance of HighSV sampling ...')
    HighSV_sample_index_lists = [ high_val_index[:i] for i in plot_points]
    HighSV_scores = [evaluate_performance(model, sample_index.copy(), directory) for sample_index in HighSV_sample_index_lists]
    
    sio.savemat(directory+'/plot_allmethod_results.mat', {'plot_points':plot_points,
                                                          'HighAV_scores':HighAV_scores,
                                                          'LowAV_scores':LowAV_scores,
                                                          'Random_scores':RD_scores,
                                                          'HighSV_scores': HighSV_scores,
                                                          'Cluster_scores':Cluster_scores} )
    
    return 

def Cluster_Sample(X,num):
    from sklearn.cluster import KMeans
    k_means = KMeans(init='k-means++', n_clusters=num, n_init=10)
    k_means.fit(X)
    mu = k_means.cluster_centers_
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
    _, indices = nbrs.kneighbors(mu)
    return indices.reshape(-1)


def HighAV_Sample(X, V, num_list, directory):

    V = V - np.min(V)
    
    M = [gaussian_membership(X, X[i]) for i in range(len(X))]
    exist_value = np.zeros(len(X))
    
    return_list = [] 
    selected_index_list = []
    all_index_list = [i for i in range(len(X))]
    
    
    for i in range(np.max(num_list)):   
        
        # Update the potential data pool
        index_list = [x for x in all_index_list if x not in selected_index_list]
        
        # Calcuate the increment to current aggregation value of each candidate
        margin_value_list = [margin_value(M[j], V, exist_value) for j in index_list]
        
        # Choosing the instance with maximal increment to current aggregation value
        index = index_list[np.argmax(margin_value_list)]
        
        # Update the aggregation value
        exist_value = [np.max((exist_value[i], (M[index]*V)[i])) for i in range(len(X))]
        
        selected_index_list.append(index)
        
        return_list.append(copy.deepcopy(selected_index_list))

    # Sampling the dataset with corresponding size required by 'num_list'
    output_list = []
    for i in num_list:
        output_list.append(return_list[i-1])
        
    return output_list

def LowAV_Sample(X, V, num_list):
    
    V = V - np.min(V)
    
    #Calcuate the sample's membership to others 
    M = [gaussian_membership(X, X[i]) for i in range(len(X))]
     
    exist_value = np.zeros(len(X))
    return_list = [] 
    selected_index_list = []
    all_index_list = [i for i in range(len(X))]
    for i in range(np.max(num_list)):      
        index_list = [x for x in all_index_list if x not in selected_index_list]
        index = index_list[np.argmin([margin_value(M[j], V, exist_value) for j in index_list])]
        exist_value = [np.max((exist_value[i], (M[index]*V)[i])) for i in range(len(X))]
        selected_index_list.append(index)
        return_list.append(copy.deepcopy(selected_index_list))

    # Sampling the dataset with corresponding size required by 'num_list'
    output_list = []
    for i in num_list:
        output_list.append(return_list[i-1])
        
    return output_list
 
def margin_value(Mj, V, exist_value):
    margin_values = Mj*V - exist_value
    margin_values = margin_values[np.where(margin_values>0)]
    margin_value = np.sum(margin_values)
    
    return margin_value
 
    
def gaussian_membership(X, mu):
    '''
    Parameters
    ----------
    X : n_number * n_features
    DESCRIPTION.
    mu : 1 * n_features
    DESCRIPTION.
       
    Returns
    -------
    TYPE n_features
    DESCRIPTION.
       
    '''
    Cov = np.diag(1e2 * np.ones(len(mu)))
    A = X - mu
    P = np.exp( - A @ np.linalg.inv(Cov) @ A.T)
    return np.diagonal(P)
     
def evaluate_performance(basemodel, sample_index, directory):
    """
    Given a set of indexes, evaluate the performance.
    """

    data = sio.loadmat(directory +'/shapley_result.mat')

    x_train = data['X']
    y_train = data['y']
    x_test  = data['X_test']
    y_test  = data['y_test']
    
    model = train_model(basemodel, x_train[sample_index], y_train[sample_index])

    score = value(basemodel, model, x_test, y_test)
    
    return score

def train_model(basemodel, X, y):
    
    if basemodel == 'LGR':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=40)
        model.fit(X,y)
        
    if basemodel == 'GP':
        model = ExactGPModel(torch.tensor(X).to(torch.float), 
                       torch.tensor(y[:,0]).to(torch.float), 
                       gpytorch.likelihoods.GaussianLikelihood())
        model.fit() 
        
                  
    return model

from sklearn.metrics import mean_absolute_error
def value(basemodel, model, X, y):
    
    if basemodel == 'GP':  
        value = mean_absolute_error(y, model.predict(torch.tensor(X)).clone().detach())
     
    if basemodel == 'LGR':
        value = model.score(X, y)

    if value < 0:
        value = 0
    return value
















