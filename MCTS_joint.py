# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:12:12 2024

@author: 11geo
"""

import numpy as np
from itertools import product
from decimal import Decimal

import Parameters
import utils



class MonteCarloTreeSearchNode_joint():
    
    def __init__(self, parent=None, means_existing=None, covariances_existing=None, sensor_means=None, prob_existences=None, prob_detections = None, cost = 0, first_cost = 0, lambdau = None, means_new = None, covariances_new = None, cost_function = None):
        
        self.is_top_node = False
        self.is_bottom_node = False
        
        self.means_existing = means_existing # target locations
        self.covariances_existing = covariances_existing # target covariance
        
        self.sensor_means = sensor_means # two sensor locations in a tuple
        self.prob_existences = prob_existences # probability of targets existences
        self.prob_detections = prob_detections # probablity of targets detection

        self.lambdau = lambdau # probability of existence for new targets, PPP part
        self.means_new = means_new # means of new targets, PPP part
        self.covariances_new =  covariances_new # covariances of new targets, PPP part
        
        self.parent = parent # parent node
        self.children = []
        
        if self.parent is None: # if initial node, parent is none 
            self.is_top_node = True # set the top node flag to true     
            self.depth = 0 # depth of this node is 0, therefore initial
            
        else:
            self.depth = self.parent.depth + 1 # if it wasnt the first node, add one to the global depth
            
            
        if self.depth >= Parameters.simulation_search_depth: # if it has reached the max global depth in its search
            self.is_bottom_node = True # set bottom node flag to true
        
        # need to add a variable that is just the cost of calculating the cost when it gets added
        self.first_cost = cost
        self.cost = cost # average cost of going through node
        self.number_of_visits = 0 # how many times the node has been visited before
        self.untried_actions = None # list of all available actions
        self.untried_actions = self.get_untried_actions() #all of the children which have not been tried
        self.cost_function = cost_function
        
    def get_untried_actions(self):
        
        self.untried_actions = self.get_legal_actions(self.sensor_means)
        
        return self.untried_actions


    def update_cost(self, cost, discount_factor): 
        # calculates the average cost of the node by using the visit count and the accumulated cost
        # of visiting the node to date.
       
        self.cost = ((self.cost * self.number_of_visits) + cost) / (self.number_of_visits + 1)
    
        return self.cost
    
   
    def get_number_of_visits(self):
        
        return self.number_of_visits
    
    def predict_pmb_empty_poisson(self, prob_existence, means_existing, covariances_existing):
        # Get multi-Bernoulli prediction parameters from the model
        F = Parameters.F[::2,::2]  # Transition matrix for the state in 2D
        Q = Parameters.Q[::2,::2]  # Process noise covariance matrix in 2D
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
        # number_of_new_tracks = len(lambdau)  
    
        # Predict existing tracks
        for i in range(number_of_existing_track):
            prob_existence[i] = prob_survival * prob_existence[i]  # Update existence probability for each track
            means_existing[:, i] = np.dot(F, means_existing[:, i])  # Predict the state for each existing track
            covariances_existing[i, :, :] = np.dot(F, np.dot(covariances_existing[i, :, :], F.T)) + Q  # Predict the covariance for each track

        
        # concatenate birth components with existing components
        prob_existence = np.concatenate((prob_existence, lambda_birth), axis=0)
        means_existing = np.concatenate((means_existing, birth_means[::2]), axis=1)
        covariances_existing = np.concatenate((covariances_existing, birth_covariances[:, [0, 2], :][:, :, [0, 2]]), axis=0)
        
    
        return prob_existence, means_existing, covariances_existing
    
   
    def _calc_cost(self, means, covs, prob_existences, sensor_positions):

        # function to take in a combination and return the costs jointly calculated
        
        means, covariances, prob_existences, cost_of_this_combination = self.joint_combination_cost(sensor_positions, means, covs, prob_existences)

        return means, covariances, prob_existences, cost_of_this_combination
    
    
    def joint_combination_cost(self, combination, means, covariances, pred_prob_existence):
        
        action1, action2 = combination
        
        # there are four options when using two sensors
        
        # S1     S2
        # empty  empty
        # empty   z
        # z      empty
        # z      z
        
        # we need to calculate the parameters associated with both options each time and then calculate a cost based on these
        means_existing_holding = np.zeros(means.shape)
        covariances_existing_holding = np.zeros(covariances.shape)
        prob_existences_existing_holding = np.zeros(np.array(pred_prob_existence).shape)
        
        cost_of_each_target = []
        # so first calculate S1 empty parameters for each target
        for i, prediction in enumerate(means.T):
            # positional_mean = prediction[::2]
            # positional_cov = covariances[i][::2,::2]
            pred_prob_existence_target = pred_prob_existence[i]
            
            P = covariances[i] #* Parameters.inflation_value
            x = prediction.reshape(2,-1)
            
            # sensor position
            s = action1.reshape(2,-1)
            s2 = action2.reshape(2,-1)
            
            distance = np.linalg.norm(s - x)
            distance2 = np.linalg.norm(s2 - x)
            
            # Sensor FOV radius
            delta = Parameters.sensor_radius
            
             # Probability of detection: Gaussian-like tapering within the FOV and beyond
            # if distance <= delta:
            pd_of_existing_target = Parameters.prob_detection * np.exp(-0.5 * (distance / delta) ** 2)
            # else:
            #     # Outside FOV: Further tapering off
            #     tapering_factor = np.exp(-0.5 * (distance / delta) ** 2) * np.exp(-0.5 * ((distance - delta) / delta) ** 2)
            #     pd_of_existing_target = Parameters.prob_detection * tapering_factor
            
            
             # Probability of detection: Gaussian-like tapering within the FOV and beyond
            # if distance2 <= delta:
            pd_of_existing_target2 = Parameters.prob_detection * np.exp(-0.5 * (distance2 / delta) ** 2)
            # else:
            #     # Outside FOV: Further tapering off
            #     tapering_factor2 = np.exp(-0.5 * (distance2 / delta) ** 2) * np.exp(-0.5 * ((distance2 - delta) / delta) ** 2)
            #     pd_of_existing_target2 = Parameters.prob_detection * tapering_factor2
            
   
            prob_detection = pd_of_existing_target
            prob_detection2 = pd_of_existing_target2
                    
            # for Z = EMPTY
            # associated with there being no measurement received
            # the predictions are took forward
            no_measurement_mean = prediction # x_{k|k}^0
            no_measurement_covariance = covariances[i] # P_{k|k}^0
            no_measurement_prob_existence = ((1 - prob_detection) * pred_prob_existence_target) / ((1 - pred_prob_existence_target) + (( 1 - prob_detection) * pred_prob_existence_target)) # r_{k|k}^0
            
            # then the costs associciated with S2 having no measurement, GIVEN S1 being empty
            no_measurement_meanS2 = no_measurement_mean # x_{k|k}^0
            no_measurement_covarianceS2 = no_measurement_covariance # P_{k|k}^0
            no_measurement_prob_existenceS2 = ((1 - prob_detection2) * no_measurement_prob_existence) / ((1 - no_measurement_prob_existence) + (( 1 - prob_detection2) * no_measurement_prob_existence)) # r_{k|k}^0
            
            if self.cost_function == 'GOSPA':
                no_measurement_costS2 = self.individual_cost_GOSPA(no_measurement_prob_existenceS2, no_measurement_covarianceS2, Parameters.c) # C(r_{k|k}^0, P_{k|k}^0)
            
            elif self.cost_function == 'KLD':
                no_measurement_costS2 = self.individual_KL_divergence(prediction, covariances[i], pred_prob_existence, no_measurement_mean, no_measurement_covariance) # C(r_{k|k}^0, P_{k|k}^0)
            
                
            
            # then the costs associciated with S2 having a measurement, GIVEN S1 being empty
            z_hat = (Parameters.H[:,[0,2]] @ no_measurement_mean) + Parameters.bias # mean of the synthetic measurement
            S_ak = (Parameters.H[:,[0,2]] @ no_measurement_covariance @ Parameters.H[:,[0,2]].T) + Parameters.R # cov of the synthetic measurement
            yes_measurement_meanS2 = no_measurement_mean # + ((cov @ Parameters.H.T @ np.linalg.inv(S_ak)) @ (synthetic_measurement - z_hat)) # mu_{k|k}^1
            yes_measurement_covarianceS2 = no_measurement_covariance - (no_measurement_covariance @ Parameters.H[:,[0,2]].T @ np.linalg.inv(S_ak) @ Parameters.H[:,[0,2]] @ no_measurement_covariance) # Sigma_{k|k}^1
            yes_measurement_prob_existenceS2 = 1 #(no_measurement_prob_existence * prob_detection2 * ((2 * np.pi)**(z_hat.shape[0]/2) * np.linalg.det(S_ak)**(-0.5))) / ((no_measurement_prob_existence * prob_detection2 * ((2 * np.pi)**(z_hat.shape[0]/2) * np.linalg.det(S_ak)**(-0.5) + Parameters.clutter_density)))
            
            if self.cost_function == 'GOSPA':
                yes_measurement_costS2 = self.individual_cost_GOSPA(yes_measurement_prob_existenceS2, yes_measurement_covarianceS2, Parameters.c)
            
            elif self.cost_function == 'KLD':
                yes_measurement_costS2 = self.individual_KL_divergence(prediction, covariances[i], pred_prob_existence, yes_measurement_meanS2, yes_measurement_covarianceS2) # C(r_{k|k}^0, P_{k|k}^0)
            
            # then S1 parameters being recorded a measurement

            # synthetic measurement
            z_hat = (Parameters.H[:,[0,2]] @ prediction) + Parameters.bias # mean of the synthetic measurement
            S_ak = (Parameters.H[:,[0,2]] @ covariances[i] @ Parameters.H[:,[0,2]].T) + Parameters.R # cov of the synthetic measurement
            yes_measurement_mean = prediction # + ((cov @ Parameters.H.T @ np.linalg.inv(S_ak)) @ (synthetic_measurement - z_hat)) # mu_{k|k}^1
            yes_measurement_covariance = covariances[i] - (covariances[i] @ Parameters.H[:,[0,2]].T @ np.linalg.inv(S_ak) @ Parameters.H[:,[0,2]] @ covariances[i]) # Sigma_{k|k}^1
            yes_measurement_prob_existence = 1 #(pred_prob_existence_target * prob_detection * ((2 * np.pi)**(z_hat.shape[0]/2) * np.linalg.det(S_ak)**(-0.5))) / ((pred_prob_existence_target * prob_detection * ((2 * np.pi)**(z_hat.shape[0]/2) * np.linalg.det(S_ak)**(-0.5) + Parameters.clutter_density)))
            
            
            # then S2 not receiving a measurement, GIVEN S1 has
            no_measurement_meanS2m = yes_measurement_mean # x_{k|k}^0
            no_measurement_covarianceS2m = yes_measurement_covariance # P_{k|k}^0
            no_measurement_prob_existenceS2m = ((1 - prob_detection2) * yes_measurement_prob_existence) / ((1 - yes_measurement_prob_existence) + (( 1 - prob_detection2) * yes_measurement_prob_existence)) # r_{k|k}^0
            
            
            
            if self.cost_function == 'GOSPA':
                no_measurement_costS2m = self.individual_cost_GOSPA(no_measurement_prob_existenceS2m, no_measurement_covarianceS2m, Parameters.c) # C(r_{k|k}^0, P_{k|k}^0)
            
            elif self.cost_function == 'KLD':
                no_measurement_costS2m = self.individual_KL_divergence(prediction, covariances[i], pred_prob_existence, no_measurement_meanS2m, no_measurement_covarianceS2m) # C(r_{k|k}^0, P_{k|k}^0)
            
            # then S2 receiving a measurement, GIVEN S1 has also
            z_hatS2m = (Parameters.H[:,[0,2]] @ yes_measurement_mean) + Parameters.bias # mean of the synthetic measurement
            S_akS2m = (Parameters.H[:,[0,2]] @ yes_measurement_covariance @ Parameters.H[:,[0,2]].T) + Parameters.R # cov of the synthetic measurement
          
            yes_measurement_meanS2m = yes_measurement_mean # + ((cov @ Parameters.H.T @ np.linalg.inv(S_ak)) @ (synthetic_measurement - z_hatS2m)) # mu_{k|k}^1
            yes_measurement_covarianceS2m = yes_measurement_covariance - (yes_measurement_covariance @ Parameters.H[:,[0,2]].T @ np.linalg.inv(S_akS2m) @ Parameters.H[:,[0,2]] @ yes_measurement_covariance) # Sigma_{k|k}^1
            yes_measurement_prob_existenceS2m = 1 #(yes_measurement_prob_existence * prob_detection2 * ((2 * np.pi)**(z_hatS2m.shape[0]/2) * np.linalg.det(S_akS2m)**(-0.5))) / ((yes_measurement_prob_existence * prob_detection2 * ((2 * np.pi)**(z_hatS2m.shape[0]/2) * np.linalg.det(S_akS2m)**(-0.5) + Parameters.clutter_density)))
            
            
            if self.cost_function == 'GOSPA':
                yes_measurement_costS2m = self.individual_cost_GOSPA(yes_measurement_prob_existenceS2m, yes_measurement_covarianceS2m, Parameters.c)
            
            elif self.cost_function == 'KLD':
                yes_measurement_costS2m = self.individual_KL_divergence(prediction, covariances[i], pred_prob_existence, yes_measurement_meanS2m, yes_measurement_covarianceS2m)
                
                
            cost_term_1 = ((1 - (pred_prob_existence_target * prob_detection)) * (1 - (no_measurement_prob_existence * prob_detection2))) * no_measurement_costS2
            cost_term_2 = (1 - (pred_prob_existence_target * prob_detection)) * (no_measurement_prob_existence * prob_detection2) * yes_measurement_costS2
            cost_term_3 = (pred_prob_existence_target * prob_detection) * (1 - (yes_measurement_prob_existence * prob_detection2)) * no_measurement_costS2m
            cost_term_4 = (pred_prob_existence_target * prob_detection) * (yes_measurement_prob_existence * prob_detection2) * yes_measurement_costS2m 
                           
            total_cost_target = cost_term_1 + cost_term_2 + cost_term_3 + cost_term_4

            weight_term_1 = ((1 - (pred_prob_existence_target * prob_detection)) * (1 - (no_measurement_prob_existence * prob_detection2)))
            weight_term_2 = (1 - (pred_prob_existence_target * prob_detection)) * (no_measurement_prob_existence * prob_detection2)
            weight_term_3 = (pred_prob_existence_target * prob_detection) * (1 - (yes_measurement_prob_existence * prob_detection2))
            weight_term_4 = (pred_prob_existence_target * prob_detection) * (yes_measurement_prob_existence * prob_detection2) 
            
            
            prob_existence_temp = (weight_term_1 * no_measurement_prob_existenceS2) + (weight_term_2 * yes_measurement_prob_existenceS2) + (weight_term_3 * no_measurement_prob_existenceS2m) + (weight_term_4 * yes_measurement_prob_existenceS2m)
            
            
            
            # weight_term_1 = (((1 - (pred_prob_existence_target * prob_detection)) * no_measurement_prob_existenceS2) * ((1 - (no_measurement_prob_existence * prob_detection2)) * no_measurement_prob_existenceS2))
            # weight_term_2 = (((1 - (pred_prob_existence_target * prob_detection)) * no_measurement_prob_existenceS2) * (no_measurement_prob_existence * prob_detection2 * yes_measurement_prob_existenceS2))
            # weight_term_3 = (pred_prob_existence_target * prob_detection * yes_measurement_prob_existenceS2) * ((1 - (yes_measurement_prob_existence * prob_detection2)) * no_measurement_prob_existenceS2m)
            # weight_term_4 = (pred_prob_existence_target * prob_detection * yes_measurement_prob_existenceS2) * (yes_measurement_prob_existence * prob_detection2 * yes_measurement_prob_existenceS2m) 
            
            
             
            target_mean_temp = ((weight_term_1 * no_measurement_prob_existenceS2 * no_measurement_meanS2) + (weight_term_2 * yes_measurement_prob_existenceS2 * yes_measurement_meanS2) + (weight_term_3 * no_measurement_prob_existenceS2m * no_measurement_meanS2m) + (weight_term_4 * yes_measurement_prob_existenceS2m * yes_measurement_meanS2m)) / prob_existence_temp
            target_cov_temp = ((weight_term_1 * no_measurement_prob_existenceS2 * (no_measurement_covarianceS2 + ((no_measurement_meanS2-target_mean_temp)@(no_measurement_meanS2-target_mean_temp).T))) + (weight_term_2 * yes_measurement_prob_existenceS2 * (yes_measurement_covarianceS2 + ((yes_measurement_meanS2-target_mean_temp)@(yes_measurement_meanS2-target_mean_temp).T))) + (weight_term_3 * no_measurement_prob_existenceS2m * (no_measurement_covarianceS2m + ((no_measurement_meanS2m-target_mean_temp)@(no_measurement_meanS2m-target_mean_temp).T))) + (weight_term_4 * yes_measurement_prob_existenceS2m * (yes_measurement_covarianceS2m + ((yes_measurement_meanS2m-target_mean_temp)@(yes_measurement_meanS2m-target_mean_temp).T)))) / prob_existence_temp
            # prob_existence_temp = (weight_term_1 * no_measurement_prob_existenceS2) + (weight_term_2 * yes_measurement_prob_existenceS2) + (weight_term_3 * no_measurement_prob_existenceS2m) + (weight_term_4 * yes_measurement_prob_existenceS2m)


            # need to return a new weighted mean, covariance and prob existence here for existing targets          
            means_existing_holding[:,[i]] = target_mean_temp.reshape((2,-1))
            covariances_existing_holding[[i],:,:] = target_cov_temp
            prob_existences_existing_holding[i] = prob_existence_temp

            cost_of_each_target.append(total_cost_target)

        total_cost = np.sum(cost_of_each_target)

        return means_existing_holding, covariances_existing_holding, prob_existences_existing_holding, total_cost


    def individual_cost_GOSPA(self, prob_existence, measurement_covariance, c):

        # breaking down the detection threshod into sub parts
        trace_part = 2 * (np.trace(measurement_covariance)/c**2) 
        min_part = 2 - (np.nanmin((trace_part,1)))
       
        optimal_detection_threshold = 1 / min_part # \Gamma_d^*
                
        if prob_existence > optimal_detection_threshold: # if  exist
            cost = (((c**2) / 2) * (1 - prob_existence)) +  (prob_existence * (np.nanmin((np.trace(measurement_covariance), c**2))))
    
            
        else: # if not exist
            cost = ((c**2) / 2) * (prob_existence)
            
        return cost
    
    
    def individual_KL_divergence(self, prior_mean, prior_cov, prior_exist, posterior_mean, posterior_cov):        
        DIMENSION = prior_mean.shape[0]
        first_term = np.trace(np.linalg.inv(posterior_cov) @ prior_cov)
        second_term = np.log(np.linalg.det(prior_cov)/np.linalg.det(posterior_cov))
        third_term = DIMENSION
        fourth_term = (posterior_mean-prior_mean).T @ np.linalg.inv(posterior_cov) @ (posterior_mean - prior_mean)
        
        divergence = prior_exist/2 * (first_term - second_term - third_term + fourth_term)
        
        return -divergence
    

    def expand(self):

         
        action = self.untried_actions.pop() # get an untried action
        
        # prediction step
        prob_existences, means_existing, covariances_existing = self.predict_pmb_empty_poisson(self.prob_existences.copy(), self.means_existing.copy(), self.covariances_existing.copy()) # predict where the target will be next
        sensor_means = action # self._calc_sensor_pos(self.sensor_mean, action)
        prob_detections = self.calculating_the_expected_pd(means_existing, covariances_existing, sensor_means)

        means_existing, covariances_existing, prob_existences, total_cost = self._calc_cost(means_existing, covariances_existing, prob_existences, sensor_means)
        
        child_node = MonteCarloTreeSearchNode_joint(parent = self,
                                                        means_existing = means_existing,
                                                        covariances_existing = covariances_existing,
                                                        sensor_means = sensor_means,
                                                        prob_existences = prob_existences,
                                                        prob_detections = prob_detections,
                                                        cost = total_cost,
                                                        first_cost = total_cost,
                                                        cost_function = self.cost_function
                                                        )

        
        
        self.children.append(child_node)
        
        return child_node
    
    def _calc_sensor_pos(self, current_state, theta):
        
        # returns the next state (x and y)(2D) in the sequence based on the current (2D state)
        # and also the fixed distance actions dynamics
        
        radius = Parameters.sensor_speed * Parameters.t
        x = current_state[0][0] + (radius * np.cos(theta))
        y = current_state[1][0] + (radius * np.sin(theta))
        next_state = np.array([[x],
                               [y]])

        return next_state
    
    
    
    def calculating_the_expected_pd(self, means_existing, covariances_existing, sensor_means):
        # this takes in all targets and a single sensor position and returns the expected Pd of both as a list
        pds_of_existing_target = []
        
        for sensor_mean in sensor_means:
            for i, prediction in enumerate(means_existing.T):
                if means_existing.shape[0] == 4:
                    positional_mean = prediction[::2]
                    positional_cov = covariances_existing[i][::2, ::2]
                else:
                    positional_mean = prediction
                    positional_cov = covariances_existing[i]
                
                # Sensor position
                s = sensor_mean.reshape(2, -1)
                
                # Distance between the sensor and the target
                distance = np.linalg.norm(s - positional_mean.reshape(2, -1))
                
                # Sensor FOV radius
                delta = Parameters.sensor_radius
                
                # Probability of detection: Gaussian-like tapering within the FOV and beyond
                # if distance <= delta:
                pd_of_existing_target = Parameters.prob_detection * np.exp(-0.5 * (distance / delta) ** 2)
                # else:
                #     # Outside FOV: Further tapering off
                #     tapering_factor = np.exp(-0.5 * (distance / delta) ** 2) * np.exp(-0.5 * ((distance - delta) / delta) ** 2)
                #     pd_of_existing_target = Parameters.prob_detection * tapering_factor
                
                pds_of_existing_target.append(pd_of_existing_target)
        
        return pds_of_existing_target  # SHOULD NOW BE 2 OF THEM
    
    
    def is_terminal_node(self):
        
        if self.depth >= Parameters.simulation_search_depth:
            return True
        
        else:
            return False
    
        
    def get_costs_from_up_tree(self, current_node):
        all_parent_costs = []
        top_node_reached = False
        
        while not top_node_reached:
            all_parent_costs = [current_node.parent.first_cost] + all_parent_costs
            current_node = current_node.parent
            top_node_reached = current_node.is_top_node
            
        return all_parent_costs
    
    def rollout(self):
        all_mcts_predictions = []
        # setting local variables equal to the current self ones for the recursive call of the rollout
        current_node = self
        current_rollout_depth = self.depth
        sensor_means = self.sensor_means
        means_existing = self.means_existing
        covariances_existing = self.covariances_existing
        prob_existences = self.prob_existences
        prob_detections = self.prob_detections
        cost = self.cost
        first_cost = self.first_cost
        cost_function = self.cost_function

        # PPP parts (not used in MB)
        lambdau = self.lambdau
        means_new = self.means_new
        covariances_new = self.covariances_new
        
        costs = self.get_costs_from_up_tree(current_node) # returns all parent costs until the top of the tree exluding the current nodes
        costs.append(cost) # adds the current nodes cost

        # while not current_node.is_terminal_node():
        while not (Parameters.simulation_search_depth - current_rollout_depth) <= 0:  
            available_actions = self.get_legal_actions(sensor_means) # based on where the sensor is now, what actions can it do
            action = self.rollout_policy(available_actions) # select one of these accoridng to the rollout policy
            pred_prob_existences, pred_means_existing, pred_covariances_existing = self.predict_pmb_empty_poisson(prob_existences.copy(), means_existing.copy(), covariances_existing.copy()) # calculate target density
            sensor_means = action #self._calc_sensor_pos(sensor_mean, action) # find where the sensor will be based on that action
            
            means_existing, covariances_existing, prob_existences, total_cost = self._calc_cost(pred_means_existing, pred_covariances_existing, pred_prob_existences, sensor_means)

            costs.append(total_cost)

            current_rollout_depth += 1
            all_mcts_predictions.append(means_existing)
            
     
        # create a weight for future reward for starting at root node
        # [0, 1, lambda**i, lambda**(i+1) ... lambda**N] where N is the depth of the tree
        weights = [0, 1] + [Parameters.discount_factor**i for i in range(1, len(costs)-1)]
        
        
        cost = np.sum(np.array(costs) * np.array(weights))
    
        return cost
    
    def backpropagate(self, cost):
        
        # update the visit count 
        self.cost = self.update_cost(cost, Parameters.discount_factor)
        self.number_of_visits += 1
        
        if not self.is_top_node: # if you are at the root node, dont update anything
        
            # if there is a parent, backpropagate again
            self.parent.backpropagate(cost)
            
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    
    def best_child(self, c_param = Parameters.best_child_c_param):

        
        choices_weights = [-(c.cost) + c_param * np.sqrt((np.log(self.number_of_visits) / c.number_of_visits)) for c in self.children]
        
        return self.children[np.argmax(choices_weights)]
    
    
    
    def rollout_policy(self, available_actions):
        
        selection = np.random.randint(0,len(available_actions))
        
        
        
        return available_actions[selection]
    
    
    def tree_policy(self):
        current_node = self
        
        while not current_node.is_terminal_node():
            
            if not current_node.is_fully_expanded():
                return current_node.expand()
            
            else:
                current_node = current_node.best_child()
                
        return current_node
    
    
    def best_action(self):
        
        simulation_number = Parameters.MCTS_simulation_number
        
        for i in range(simulation_number):
            
            # v is the best child returned or the expanded children
            v = self.tree_policy()
            cost = v.rollout()
            v.backpropagate(cost)
        
        
        return self.best_child(c_param = 0) # the original one
        
    
        

    def get_legal_actions(self, current_states):
        all_sensors_available_positions_list = []
        
        for current_state in current_states:
            all_available_states = []
            all_available_actions = []
            
            theta = (2 * np.pi) / Parameters.number_of_actions
            turn_angles = np.arange(0, 2 * np.pi, theta)
    
            for angle in turn_angles:   
                all_available_actions.append(angle)    
                
            for angle in all_available_actions:
                
                next_state = utils.fixed_distance_actions_transition(current_state, Parameters.sensor_speed, Parameters.t, angle)
                all_available_states.append(next_state)
        
            all_available_states.append(current_state)
            
            filtered_actions = [action for action in all_available_states 
                        if Parameters.surveillance_area[0][0] <= action[0] <= Parameters.surveillance_area[0][1] and
                           Parameters.surveillance_area[0][0] <= action[1] <= Parameters.surveillance_area[0][1]]
            
            
            # Initialize temp_actions_list with all actions
            temp_actions_list = filtered_actions
            
            for obstacle in Parameters.obstacles_list:
                # Get actions that are not in conflict with the current obstacle
                
                obstacle_omitted_actions = utils.check_for_obstacles_list(obstacle, temp_actions_list)
                
                if len(Parameters.obstacles_list) == 1:
                    temp_actions_list = obstacle_omitted_actions
                
                if len(Parameters.obstacles_list) > 1:
                    # Update temp_actions_list with the actions that are valid across all processed obstacles
                    temp_actions_list = utils.unique_arrays(obstacle_omitted_actions, temp_actions_list)
           
            if len(Parameters.obstacles_list) >= 1:
                available_actions = temp_actions_list
            else:
                available_actions = filtered_actions
            
            
            
            all_sensors_available_positions_list.append(available_actions)
        
        # work out all of the possible combinations of sensor positions, along with their corresponding index
        combinations = list(product(*all_sensors_available_positions_list)) # * operator unpacks the list and gives each element as an individual argument to the function
        
        return combinations       