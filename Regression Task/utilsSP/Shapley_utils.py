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
import gpytorch

torch.set_default_tensor_type(torch.DoubleTensor)

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
        x = torch.tensor(x)
        return  self(x).mean

    
    def fit(self):
        # Find optimal model hyperparameters
        self.train()
        self.likelihood.train()
        
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.1 )  # Includes GaussianLikelihood parameters
        
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
     
def plots_results(directory):
    
    Result = sio.loadmat(directory+'/plot_shapley_results.mat')
    plot_points = Result['plot_points'][0]
    RH_scores = Result['RH_scores'][0]
    RL_scores = Result['RL_scores'][0]
    AH_scores = Result['AH_scores'][0]
    AL_scores = Result['AL_scores'][0]
    

    plt.figure(figsize=(12,5))  
    size = 16
    fontfamily = 'arial'
    font = {'family':fontfamily,
            'size':size,
            'weight':25}
    
    ax = plt.subplot(121)
    plt.plot(plot_points, RH_scores, 'purple', lw=2, zorder=9, label='Remove High') 
    plt.plot(plot_points, RL_scores, 'red', lw=2, zorder=9, label='Remove Low')
    ax.set_xlabel('Remove Samples', fontproperties = fontfamily, size = size)
    ax.set_ylabel('Scores',fontproperties = fontfamily, size = size)
    plt.yticks(fontproperties = fontfamily, size = size) 
    plt.xticks(fontproperties = fontfamily, size = size) 
    # ax.set_title(Title, fontproperties = fontfamily, size = 12)
    plt.legend(prop=font)
    plt.tight_layout()
        
    ax = plt.subplot(122)
    plt.plot(plot_points, AH_scores, 'purple', lw=2, zorder=9, label='Add High') 
    plt.plot(plot_points, AL_scores, 'red', lw=2, zorder=9, label='Add Low')
    ax.set_xlabel('Add Samples', fontproperties = fontfamily, size = size)
    ax.set_ylabel('Scores',fontproperties = fontfamily, size = size)
    plt.yticks(fontproperties = fontfamily, size = size) 
    plt.xticks(fontproperties = fontfamily, size = size) 
    # ax.set_title(Title, fontproperties = fontfamily, size = 12)
    plt.legend(prop=font)
    plt.show()
    
    plt.subplots_adjust(wspace = 0.2, hspace = 0.01)

    Title = 'Sin'
    plt.savefig(directory + '/figs/Fig-'+ str(Title) +'.svg',format='svg')
    plt.savefig(directory + '/figs/Fig-'+ str(Title) +'.pdf',format='pdf')

def evaluate_results(basemodel, number_initial_points, num_plot_markers=20, num_interval_points=1,directory='.', task='Classification'): #
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
    mem_tmc = sio.loadmat(directory+'/shapley_result.mat')['Value']
    
    val_shapley_list = ((np.cumsum(mem_tmc, 0)/np.reshape(np.arange(1, len(mem_tmc)+1), (-1,1))))
    val_shapley = val_shapley_list[-1]
    low_val_index = np.argsort(val_shapley)
    high_val_index = low_val_index[::-1] # max to min
    
    plot_points = np.arange(0, num_plot_markers, num_interval_points)
    
    
    # remove high value points
    RH_sample_index_lists = [ high_val_index[i:] for i in plot_points]
    RH_scores = [evaluate_performance(basemodel, sample_index.copy(), directory, task) for sample_index in RH_sample_index_lists]
    # remove low value points
    RL_sample_index_lists = [ low_val_index[i:] for i in plot_points]
    RL_scores = [evaluate_performance(basemodel, sample_index.copy(), directory, task) for sample_index in RL_sample_index_lists]
    # add high value points
    AH_sample_index_lists = [ high_val_index[:i + number_initial_points] for i in plot_points]
    AH_scores = [evaluate_performance(basemodel, sample_index.copy(), directory, task) for sample_index in AH_sample_index_lists]
    # add low value points
    AL_sample_index_lists = [ low_val_index[:i + number_initial_points] for i in plot_points]
    AL_scores = [evaluate_performance(basemodel, sample_index.copy(), directory, task) for sample_index in AL_sample_index_lists]
    
    sio.savemat(directory+'/plot_shapley_results.mat', {'plot_points':plot_points, 
                                                        'RH_scores':RH_scores,
                                                        'RL_scores':RL_scores,
                                                        'AH_scores':AH_scores,
                                                        'AL_scores':AL_scores} )
    return 

from sklearn.metrics import mean_absolute_error
def value(basemodel, model, X, y):
    
    if basemodel == 'GP':  
        
        value = mean_absolute_error(y, model.predict(torch.tensor(X)).clone().detach())
     
    if basemodel == 'LGR':
        value = model.score(X, y)
    
    if value < 0:
        value = 0
        
    return value

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
        model = ExactGPModel(torch.tensor(X), 
                       torch.tensor(y[:,0]), 
                       gpytorch.likelihoods.GaussianLikelihood())
        model.fit() 
        
    return model


















