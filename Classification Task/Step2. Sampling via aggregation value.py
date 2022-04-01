# -*- coding: utf-8 -*-
"""
Created on April 1, 2022

This code is part of the supplement materials of the submmited manuscript:'Data sampling via aggregation value'.

"""

import utilsAV.AggregationValue as F
import os
import warnings 
from pathlib import Path

warnings.simplefilter("ignore")

result_directory = './Result'
task = 'Classification'
basemodel = 'LGR' 

if Path(result_directory +'/plot_allmethod_results.mat').is_file():
    os.remove(result_directory +'/plot_allmethod_results.mat')

if Path(result_directory +'/shapley_result.mat').is_file():
    F.evaluate_results(basemodel, 
                        number_initial_points = 10, 
                        num_plot_markers = 400, 
                        num_interval_points = 4, 
                        directory = result_directory, 
                        task=task)
else:
    print("===========================================================")
    print("The directory don't contains the file 'shpley_result.mat' ")
    print("                   Please Run step1! ")
    print("===========================================================")
    