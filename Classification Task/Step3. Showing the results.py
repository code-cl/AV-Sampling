# -*- coding: utf-8 -*-
"""
Created on April 1, 2022

This code is part of the supplement materials of the submmited manuscript:'Data sampling via aggregation value'.


"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import warnings 
from pathlib import Path

warnings.simplefilter("ignore")

result_directory = './Result'
Title = 'Results on the CWRU HP0 task'

if Path(result_directory +'/plot_allmethod_results.mat').is_file():
    
    Result = sio.loadmat(result_directory +'/plot_allmethod_results.mat')
    plot_points    = Result['plot_points'][0]
    HighSV_scores  = Result['HighSV_scores']
    Cluster_scores = Result['Cluster_scores']
    HighAV_scores  = Result['HighAV_scores']
    LowAV_scores   = Result['LowAV_scores']
    Random_scores  = Result['Random_scores']
    
    plt.figure(figsize=(6,4.3))  
    size = 18
    fontfamily = 'arial'
    font = {'family':fontfamily,
            'size': 14,
            'weight':25}
    
    ax = plt.subplot()
    plt.subplots_adjust(left=0.15, right=0.98, bottom=0.15,top=0.9,wspace = 0.2, hspace = 0.05)
    
    plt.plot(plot_points.reshape(-1), np.mean(Random_scores, 0).reshape(-1), 
              '#757575', lw=2, zorder=9, label='Random')
    plt.fill_between(plot_points.reshape(-1), 
                    np.min(Random_scores, 0).reshape(-1), 
                    np.max(Random_scores, 0).reshape(-1), 
                    color='gray', alpha=0.3) 

    plt.plot(plot_points.reshape(-1), np.mean(Cluster_scores, 0).reshape(-1), 
             '#81c784', lw=2, zorder=9, label='Cluster')
   
    plt.plot(plot_points.reshape(-1), np.mean(HighSV_scores, 0).reshape(-1), 
             '#039be5', lw=2, zorder=9, label='HighSV') 

    plt.plot(plot_points.reshape(-1), np.mean(LowAV_scores, 0).reshape(-1), 
              '#6b4e9b', lw=2, zorder=9, label='LowAV')
      
    plt.plot(plot_points.reshape(-1), np.mean(HighAV_scores, 0).reshape(-1), 
             '#f44336', lw=2, zorder=9, label='HighAV')
    
    ax.set_xlabel('Number of samples', fontproperties = fontfamily, size = size)
    ax.set_ylabel('Accuracy',fontproperties = fontfamily, size = size)
    plt.yticks(fontproperties = fontfamily, size = size) 
    plt.xticks(fontproperties = fontfamily, size = size) 
    # ax.set_title(Title, fontproperties = fontfamily, size = 12)
    # plt.xlim([0, 30])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.minorticks_on()
    plt.tick_params(which='major',length=7,width=2)
    #plt.axhspan(0.85, 1, facecolor='#e91e63', alpha=0.2)
    plt.axhspan(0.8, 1, facecolor='#b3e5fc', alpha=0.5)
    plt.axhspan(0, 0.8, facecolor='#f9f9f9', alpha=0.5)
    # plt.axvline(x=42,linestyle='--',color='#607d8b')
    # plt.title(Title)
    plt.legend(prop = font,framealpha=0.6,loc=4)
    plt.ylim(0,1)
    plt.xlim(10,406)
    plt.grid(linestyle='-.',axis="y")
    plt.show()
    plt.savefig(result_directory + '/figs/Fig-'+ str(Title) +'.png',dpi=600,bbox_inches='tight')
    plt.savefig(result_directory + '/figs/Fig-'+ str(Title) +'.svg',format='svg',bbox_inches='tight')
    plt.savefig(result_directory + '/figs/Fig-'+ str(Title) +'.pdf',format='pdf',bbox_inches='tight')

        









