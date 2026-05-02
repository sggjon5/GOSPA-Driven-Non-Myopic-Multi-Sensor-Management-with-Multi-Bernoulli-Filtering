# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:21:02 2023

@author: sggjone5

The extractor class contains a method which extracts the highest weighted componenets.
    
"""
import numpy as np 


class Extractor():
    
    
    def __init__(self, updater):
        
        self.extracted_state = []
        self.extracted_cov = []
        
        self.all_extracted_states = []
        self.all_extracted_covs = []
        
    def extract_highest_weight(self, prob_existence_update, mean_update, cov_update, weight_update):

        if prob_existence_update > 0.5:
           max_weight_position = np.nanargmax(weight_update)
           mean_estimate = mean_update[:,[max_weight_position]]
           cov_estimate = cov_update[[max_weight_position],:,:]
           
        else:
            max_weight_position = np.nan
            mean_estimate = np.array([[np.nan], [np.nan], [np.nan], [np.nan]])
            nanarray = np.zeros((1,4,4))
            nanarray[:] = np.nan
            cov_estimate = nanarray
            
        
        self.extracted_state = mean_estimate
        self.extracted_cov = cov_estimate
        
        self.all_extracted_states.append(mean_estimate)
        self.all_extracted_covs.append(cov_estimate)
