# -*- coding: utf-8 -*-
"""
Created on Thu Oct 5 16:20:34 2023
@author: sggjone5
"""

import numpy as np
import Parameters

class Groundtruth():
    
    def __init__(self):
        
        # initial state as defined in the parameters file
        # self.initial_state = Parameters.initial_state
        
        # empty arrray of the correct state dimension and number of timesteps in length
        # for target state storage, each entry in the list corresponds to a different target
        self.target_states = []
        
        # self.target_states = [np.tile(np.array([[np.nan],[np.nan],[np.nan], [np.nan]]) , (1, Parameters.number_of_timesteps))]
                                     
        
        # whether there is a target there or not 1=present, 0 = not 
        self.cardinality = np.zeros([Parameters.number_of_timesteps, 1])
        
        # method called upon initialisation as will never need to initialise without generating an initial state
        # self.generate_initial_state()
        
        
    def generate_multitarget_groundtruth(self,):
        # for every timestep, check if new targets should be born (from every birth density),
        # check if targets should die, calculate next target positon

        
        for k in range(Parameters.number_of_timesteps):
    
            #################################################################
            # generate when targets are born and also their initial states
            #################################################################
            
            # for each birth density
            for density in range(Parameters.number_of_birth_densities):
                
                # check if a target should be born by drawing from a Poisson
                targets_born = np.random.poisson(Parameters.prob_birth[density])
                
                # for every target that is born at this density
                for i in range(targets_born):
                    
                    # draw a random sample from this density
                    new_target = np.random.multivariate_normal(Parameters.birth_means[:,density], Parameters.birth_covariances[density, :,:])
                    new_target = new_target.reshape(new_target.shape[0], -1) # make it 2d for the concatenation below
                    
                    # and add a new target array  to the target states list 
                    self.target_states.append(np.tile(np.array([[np.nan],
                                                                [np.nan],
                                                                [np.nan],
                                                                [np.nan]]) , (1, Parameters.number_of_timesteps)))
                    
                    # and set the current time steps state = to the new target state
                    self.target_states[-1][:,[k]] = new_target
            
            # now the above for loop is done, we have the total amount of targets and when they were born
            
        ########################################################################
        # generate the trajectories of the born targets and check when they die
        ########################################################################           
            
        # for each target array (of all timesteps)
        for i, target in enumerate(self.target_states):
            
            # and for each state at every timestep
            for timestep_state in range(Parameters.number_of_timesteps):
                
                # check if there is a target there, if no, pass
                if np.isnan(target[:,timestep_state][0]) == True:
                    pass
                
                # if yes
                else:
                    # check if it should die
                    if self.check_death() == True:
                        self.target_states[i][:,[timestep_state]] = np.array([[np.nan],
                                                                              [np.nan],
                                                                              [np.nan],
                                                                              [np.nan]])
                    # if no, generate new state ( being careful about indexing errors on the timestep_state+1)
                    elif timestep_state < (Parameters.number_of_timesteps-1):
                        
                        self.target_states[i][:,[timestep_state+1]] = self.generate_new_state(Parameters.F, self.target_states[i][:,[timestep_state]], Parameters.Q, 'noise', k)
                        # check it is within the surveillance area
                        self.check_surveillance_region(i, timestep_state)
                        
                        # make sure its dead at the end
                    if timestep_state == Parameters.number_of_timesteps-1: 
                        self.target_states[i][:,[timestep_state]] = np.array([[np.nan],
                                                                              [np.nan],
                                                                              [np.nan],
                                                                              [np.nan]])
                        
                    else:
                        pass
            
                        
                                    
        if self.target_states == []:
            # temp = np.zeros((4,Parameters.number_of_timesteps)) 
            # temp[:] = np.array([[np.nan],
            #                     [np.nan],
            #                     [np.nan],
            #                     [np.nan]])
            # self.target_states = [temp]
            print("no targets were born")

                   
        return self.target_states
    
    
    def generate_initial_state(self):
        # initialising to see if the target is born at the start of the simulation
        # for each birth density
        for density in range(Parameters.number_of_birth_densities):
            
            # check if a target should be born by drawing from a Poisson
            targets_born = np.random.poisson(Parameters.prob_birth[density])
            
            # for every target that is born at this density
            for i in range(targets_born):
                
                # draw a random sample from this density
                new_target = np.random.multivariate_normal(Parameters.birth_means[:,density], Parameters.birth_covariances[density, :,:])
                new_target = new_target.reshape(new_target.shape[0], -1) # make it 2d for the concatenation below
                
                # and add a new target array  to the target states list 
                self.target_states.append(np.tile(np.array([[np.nan],
                                                            [np.nan],
                                                            [np.nan],
                                                            [np.nan]]) , (1, Parameters.number_of_timesteps)))
                
                # and set the initial time steps state = to the new target state
                self.target_states[-1][:,[0]] = new_target
        

        
    
    # checks to see if the object existed at the previous timestep
    # True for yes, False for no
    def check_previous_existence(self, current_timestep):
        
        # if there is no object at the previous timestep
        if self.cardinality[current_timestep-1] == 0: 
            existence = False
        
        else:
            existence = True
            
        return existence    
        
            
    # checks if the object should be born   
    # True for yes, False for No    
    def check_birth(self):
        
        if Parameters.prob_birth >= np.random.rand():
            birth = True
            
        else:
            birth = False
            
        return birth
    

    # checks if target should die
    # True is yes, False is no
    def check_death(self):
        
        if Parameters.prob_death >= np.random.rand():
            death = True
            
        else:
            death = False
            
        return death 
        
    
    # generates a placeholder state for when there is no target alive
    def generate_empty_state(self, timestep):
        
        new_state = np.array([[np.nan],
                              [np.nan],
                              [np.nan],
                              [np.nan]])
        
        self.target_states[:,[timestep]] = new_state
        self.cardinality[timestep] = 0
                
    
    # generates a new state for when there is a target alive
    def generate_new_state(self, F, target_state, Q, noise, timestep):
        
        
        if noise == 'noise':
            noise = (np.linalg.cholesky(Q) @ np.random.randn(F.shape[0], target_state.shape[1]))
        
        else:
            noise = np.zeros([F.shape[0], target_state.shape[1]]) 

        new_state = np.add((F @ target_state), noise)
 
        return new_state
            
        # self.target_states[:,[timestep]] = new_state
        # self.cardinality[timestep] = 1
    
    
    # checks whether target has left the surveillance volume
    def check_surveillance_region(self, target_index, timestep):
        
        if self.target_states[target_index][:,[timestep]][0][0]  >= Parameters.surveillance_area[0][1]:
            self.target_states[target_index][:,[timestep]] = np.array([[np.nan],[np.nan],[np.nan],[np.nan]])
            self.cardinality[timestep] = 0
            
                
        if self.target_states[target_index][:,[timestep]][2][0] >= Parameters.surveillance_area[0][1]:
            self.target_states[target_index][:,[timestep]] = np.array([[np.nan],[np.nan],[np.nan],[np.nan]])
            self.cardinality[timestep] = 0
           
            
        if self.target_states[target_index][:,[timestep]][0][0] <= Parameters.surveillance_area[0][0]:
            self.target_states[target_index][:,[timestep]] = np.array([[np.nan], [np.nan], [np.nan], [np.nan]])
            self.cardinality[timestep] = 0
            
            
        if self.target_states[target_index][:,[timestep]][2][0] <= Parameters.surveillance_area[0][0]:
            self.target_states[target_index][:,[timestep]] = np.array([[np.nan],[np.nan],[np.nan],[np.nan]])
            self.cardinality[timestep] = 0
            