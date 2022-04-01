# -*- coding: utf-8 -*-
"""
Created on April 1, 2022

This code is part of the supplement materials of the submmited manuscript:'Data sampling via aggregation value'.

"""

from utilsSP.Shapley import Shapley
import scipy.io as sio
import warnings
import utilsSP.Shapley_utils as F

warnings.simplefilter("ignore")

task = 'Regression'
basemodel = 'GP' # NN, GP, LGR
n_start = 5

data = sio.loadmat('./Composite/data.mat')
result_directory = './Result'

x_train = data['x_train']
y_train = data['y_train']
x_test  = data['x_test']
y_test  = data['y_test']
    
if 1:
    model_Shapley = Shapley(basemodel, x_train,  y_train, x_test, y_test,
                    directory = result_directory, task = task, n_start = n_start)
    
    model_Shapley.run(err = 3,
                      tolerance = 0.3
                     )
    
# if 0:                                                   
#     F.evaluate_results(basemodel, number_initial_points=5, num_plot_markers=580, num_interval_points = 5, 
#                         directory=result_directory, task = task)
    
#     F.plots_results(directory=result_directory)
