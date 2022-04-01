# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 16:52:45 2021

@author: Tangmei
"""
import os
import _pickle as pkl
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import warnings
import scipy.io as sio
from pathlib import Path
import utilsSP.Shapley_utils as F
#import gpytorch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Shapley(object):   
   
    def __init__(self, basemodel, X, y, X_test, y_test,
                 directory=None, task='Regression', n_start = 10):
        
        self.basemodel = basemodel
        self.task = task
        self.directory = directory
        self.random_score = 1
        if self.task == 'Regression':
            self.random_score = 1000
        if self.task == 'Classification':
            self.random_score = 0
        self.X_test    = torch.tensor(X_test) #[-num_test:]
        self.y_test    = torch.tensor(y_test) #[-num_test:]
        self.X, self.y = torch.tensor(X), torch.tensor(y)
        self.n_start = n_start
        
        n_points = len(self.X)
        self.margins_tmc = torch.zeros((0, n_points))
        if self.directory != None:
            if not os.path.exists(directory):
                os.makedirs(directory)
                os.makedirs(os.path.join(directory, 'plots'))

    def run(self, err = 0.0001, tolerance = 0.01):
        """Calculates data sources(points) values.
        Args:
            save_every: save marginal contrivbutions every n iterations.
            err: stopping criteria.
            tolerance: Truncation tolerance. If None, it's computed.
        """
        tmc_run = True
        while tmc_run :
            self._tmc_shap(tolerance = tolerance)
            print('Iter = ', len(self.margins_tmc),' Delta_E = ', np.round(self.delta_error(), 5), ' Err  = ', err)
            if self.delta_error() < err:               
                tmc_run = False

        print('Finished Shapley')

    def delta_error(self):
        '''
        Determine whether the iteration ends.

        '''
        margins = sio.loadmat(self.directory +'/shapley_result.mat')['Value']
        self.margins_tmc = margins
        
        all_vals = (np.cumsum(margins, 0)/np.reshape(np.arange(1, len(margins)+1), (-1,1)))[-50:]
        
        error = np.max(np.mean(all_vals  - np.min(all_vals, 0), 0))

        return error

    def _tmc_shap(self, iterations=20, tolerance=None):
                  
        """Runs TMC-Shapley algorithm.      
        Args:
            iterations: Number of iterations to run.`
            tolerance: Truncation tolerance ratio.
        """
        self._tol_mean_score()
        if tolerance is None:
            tolerance = self.tolerance
        temp_marginals = []
        margins_tmc = np.zeros((0, len(self.X)))
        for iteration in range(iterations):
            temp_marginals,_ = self.one_iteration(tolerance)
            
            if Path(self.directory +'/shapley_result.mat').is_file():
                margins_tmc = sio.loadmat(self.directory +'/shapley_result.mat')['Value']

            if 10*(iteration+1)/iterations % 1 == 0:
                print(' {} Margins, {} out of {} TMC_Shapley iterations.'.format(
                    len(margins_tmc), iteration + 1, iterations))

            self.margins_tmc = np.concatenate([
                self.margins_tmc, 
                np.reshape(temp_marginals, (1,-1))])
           
            sio.savemat(self.directory+'/shapley_result.mat', {'X':self.X.detach().numpy(), 
                                                               'y':self.y.detach().numpy(), 
                                                               'X_test': self.X_test.detach().numpy(),
                                                               'y_test': self.y_test.detach().numpy(),
                                                               'Value':self.margins_tmc})
             
    def one_iteration(self, tolerance):
        
        start_n = self.n_start
        
        idxs = np.random.permutation(len(self.X))
        marginal_contribs = np.zeros(len(self.X))
        all_trainerror = np.zeros(len(self.X))
        
        
        X_batch = self.X[idxs][0:start_n]
        y_batch = self.y[idxs][0:start_n]
        truncation_counter = 0
        new_score = self.random_score
        
        for n, idx in enumerate(idxs): 
            
            if n >= start_n: 
                
                old_score = new_score
                
                X_batch = np.concatenate([X_batch, self.X[idx].reshape(1,-1)])
                y_batch = np.concatenate([y_batch, self.y[idx].reshape(1,-1)])   
                
                if ( self.task =='Regression'
                    or len(set(y_batch[:,0])) == len(set(self.y_test[:,0].detach().numpy()))):
                    model = F.train_model(self.basemodel, X_batch, y_batch)
                    new_score = F.value(self.basemodel, model, self.X_test, self.y_test)
                    
                    all_trainerror[n] = new_score
                
                #compute the marginal gain
                if old_score != self.random_score:
                    
                    if self.task =='Regression':
                        marginal_contribs[idx] = (old_score - new_score)
                        
                    if self.task == 'Classification':
                        marginal_contribs[idx] = (new_score - old_score)
                    
                    #The condition for truncation
                    distance_to_full_score = np.abs(new_score - self.mean_score)
                   
                    if distance_to_full_score <= tolerance * self.mean_score: 
    	                truncation_counter += 1
    	                if truncation_counter > 3:
    	                    break
                    else:
                        truncation_counter = 0
                        
        print('idx', n, ' new_score : ', new_score)
        
        return marginal_contribs, all_trainerror


    def _tol_mean_score(self):
        '''
        Evaluate the mean of results provided by repeating 10 times on the total samples
        '''
        scores = []
        model = F.train_model(self.basemodel, self.X, self.y)
        for _ in range(10):
            bag_idxs = np.random.choice(len(self.y_test), len(self.y_test))
            scores.append(F.value(self.basemodel, model,
                self.X_test[bag_idxs],
                self.y_test[bag_idxs]))
            
        self.tol = np.std(scores)
        
        self.mean_score = np.mean((scores))
        print('mean_score : ', self.mean_score)


        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


