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

task = 'Classification'
basemodel = 'LGR' # NN, GP, LGR
n_start = 10

data = sio.loadmat('./CWRU/HP0/data.mat')
result_directory = './Result'

x_train = data['x_train']
y_train = data['y_train']
x_test  = data['x_test']
y_test  = data['y_test']
    
if 1:
    model_Shapley = Shapley(basemodel, x_train,  y_train, x_test, y_test,
                    directory = result_directory, task = task, n_start = n_start)
    
    model_Shapley.run(err = 0.002,
                      tolerance = 0.02
                     )
    
# if 0:                                                   
#     F.evaluate_results(basemodel, number_initial_points=30, num_plot_markers=400, num_interval_points = 10, 
#                         directory=result_directory, task = task)
    
#     F.plots_results(directory=result_directory)
