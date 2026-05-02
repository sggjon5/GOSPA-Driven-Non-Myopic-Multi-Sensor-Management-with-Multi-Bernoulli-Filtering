# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:03:30 2023

@author: sggjone5
"""

import numpy as np

import Parameters



class Predictor():
    
    def __init__(self):
        
        pass
        
        
    def predict_pmb(self, prob_existence, means_existing, covariances_existing, lambdau, means_new, covariances_new):
        # Get multi-Bernoulli prediction parameters from the model
        F = Parameters.F  # Transition matrix for the state
        Q = Parameters.Q  # Process noise covariance matrix
        prob_survival = Parameters.prob_survival  # Survival probability
    
        # Get birth parameters from the model
        lambda_birth = np.array(Parameters.prob_birth)  # Birth intensity
        
        # lambdau = np.array([0])  # Existing track intensity
        number_birth_components = len(lambda_birth) # Number of birth components
        birth_means = Parameters.birth_means
        birth_covariances = Parameters.birth_covariances
        
        lambdab_threshold = 1e-4  # A threshold for low birth intensity components
    
        # Determine the number of measurements and existing tracks
        number_of_existing_track = len(prob_existence)
        number_of_new_tracks = len(lambdau)  
    
        # Predict existing tracks
        for i in range(number_of_existing_track):
            prob_existence[i] = prob_survival * prob_existence[i]  # Update existence probability for each track
            means_existing[:, i] = np.dot(F, means_existing[:, i])  # Predict the state for each existing track
            covariances_existing[i, :, :] = np.dot(F, np.dot(covariances_existing[i, :, :], F.T)) + Q  # Predict the covariance for each track
    
    
        # Predict existing PPP (Poisson Point Process) intensity

        for k in range(number_of_new_tracks):
            lambdau[k] = prob_survival * lambdau[k]  # Update the intensity for each PPP component
            means_new[:, k] = F @ means_new[:,k]  # Predict the state for each PPP component
            covariances_new[k, :, :] = np.dot(F, np.dot(covariances_new[k, :, :], F.T)) + Q  # Predict the covariance for each PPP component

        # Incorporate birth intensity into the PPP

        # Allocate memory for new birth components
        lambdau = np.append(lambdau, np.zeros(number_birth_components))  # Append zeros for new birth components
        means_new = np.column_stack((means_new, np.zeros((len(means_new), number_birth_components))))  # Add columns for new birth components to state matrix
        covariances_new = np.concatenate((covariances_new, np.zeros((number_birth_components, len(means_new), len(means_new)))), axis=0)  # Add new birth component covariances
    
        for k in range(number_birth_components):
            lambdau[number_of_new_tracks + k] = lambda_birth[k]  # Set intensity for each new birth component
            means_new[:, number_of_new_tracks + k] = birth_means[:, k]  # Set the state for each new birth component
            covariances_new[number_of_new_tracks + k, :, :] = birth_covariances[k, :, :]  # Set the covariance for each new birth component
        
        
        
        
            # Not shown in the paper -- truncate low-weight components
        ss = lambdau > lambdab_threshold  # Identify birth components with sufficient intensity
        lambdau = lambdau[ss]  # Remove birth components with low intensity
        means_new = means_new[:, ss]  # Remove the corresponding state components
        covariances_new = covariances_new[ss, :, :]  # Remove the corresponding covariance components
        
        # # concatenate birth components with existing components
        # prob_existence = np.concatenate((prob_existence, lambda_birth), axis=0)
        # means_existing = np.concatenate((means_existing, birth_means), axis=1)
        # covariances_existing = np.concatenate((covariances_existing, birth_covariances), axis=0)
        
        # self.pred_prob_existence = prob_existence
        # self.predicted_means_existing = means_existing
        # self.predicted_covariances_existing = covariances_existing
        # self.lambdau = lambdau
        # self.predicted_means_new = means_new
        # self.predicted_covariances_new = covariances_new
    
        return prob_existence, means_existing, covariances_existing, lambdau, means_new, covariances_new
    
    
    def predict_pmb_empty_poisson(self, prob_existence, means_existing, covariances_existing):
        # Get multi-Bernoulli prediction parameters from the model
        F = Parameters.F  # Transition matrix for the state
        Q = Parameters.Q  # Process noise covariance matrix
        prob_survival = Parameters.prob_survival  # Survival probability
    
        # Get birth parameters from the model
        lambda_birth = np.array(Parameters.prob_birth)  # Birth intensity
        
        # lambdau = np.array([0])  # Existing track intensity
        number_birth_components = len(lambda_birth) # Number of birth components
        birth_means = Parameters.birth_means
        birth_covariances = Parameters.birth_covariances
        
        
    
        # Determine the number of measurements and existing tracks
        number_of_existing_track = len(prob_existence)
         
    
        # Predict existing tracks
        for i in range(number_of_existing_track):
            prob_existence[i] = prob_survival * prob_existence[i]  # Update existence probability for each track
            means_existing[:, i] = np.dot(F, means_existing[:, i])  # Predict the state for each existing track
            covariances_existing[i, :, :] = np.dot(F, np.dot(covariances_existing[i, :, :], F.T)) + Q  # Predict the covariance for each track

    
        
        
        # concatenate birth components with existing components
        prob_existence = np.concatenate((prob_existence, lambda_birth), axis=0)
        means_existing = np.concatenate((means_existing, birth_means), axis=1)
        covariances_existing = np.concatenate((covariances_existing, birth_covariances), axis=0)
        
        
        return prob_existence, means_existing, covariances_existing
        
        
