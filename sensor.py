# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 18:25:52 2023

@author: sggjone5
"""
import numpy as np
import scipy
from itertools import product


from predict import Predictor
from update import Updater
from extractor import Extractor
import Parameters
import utils
from MCTS import MonteCarloTreeSearchNode_individual
from MCTS_joint import MonteCarloTreeSearchNode_joint
from decimal import Decimal



class Sensor_base_class():
    
    def __init__(self):
        
        self.movement_model = Movement_model()
        self.predictor = Predictor()
        self.measurements = Measurements()
        self.updater = Updater(self.predictor, self.measurements)
        self.extractor = Extractor(self.updater)
        

        
class Movement_model():
    """
    class to define the movement capabilities of the agile sensors.
    
    state represented as a 2d coordinate in [x,y]
    
    sensor can take actions that sit on the circuference of a circle with fixed radius,
    centred around the sensors current position.
    
    """
    
    def __init__(self):
        
        # representing x and y position in the 2d cartesian grid
        self.state = np.array([[0],[0]])
        
     
    
    def fixed_distance_actions_transition(self, current_state, speed, t, theta):
        
        # returns the next state (x and y)(2D) in the sequence based on the current (2D state)
        # and also the fixed distance actions dynamics
        
        radius = speed * t
    
        x = current_state[0][0] + (radius * np.cos(theta))
        y = current_state[1][0] + (radius * np.sin(theta))
        next_state = np.array([[x],
                               [y]])
        
        
        return next_state


class Measurements():
    
    # a class to generate the measurements based off of the groundtruth target
    
    def __init__(self):
        
        self.all_measurements = [np.array([[np.nan],
                                          [np.nan]])] * Parameters.number_of_timesteps
        
    def generate_measurements_multiple_targets(self, target_state, selected_sensor, k):
        
        # generates detections based on whether the target is alive or not 
        # and if so, where it is
        # generates clutter based on the clutter rate in Parameters.py and
        # the sensors current location
        # stores a list of all measurements (detecions and clutter combined) 
        # as an attribute of the measurements class 
   
        detections = np.array([[np.nan], [np.nan]])
        # transpose to iterate over columns    
        for j, state in enumerate(target_state.T):
            
            state = state[:, np.newaxis]
            is_all_nan = np.all(np.isnan(state))

            # Map the result to 0 or 1
            cardinality = 0 if is_all_nan else 1
            
            # cardinality = np.where(np.any(np.isnan(state[0]), axis=0), 0, 1)
            # if there is an object there
            if cardinality > 0:
                
                # determine whether it shoud be detected using the probability of detection
                detected_or_not = np.random.rand()
                
                if detected_or_not <= Parameters.prob_detection: #TODO change to exponent version?
                    
                    # if detected then generate a detection
                    detection = self.generate_observation(Parameters.H, Parameters.R, state, 'noise')
                    detections = np.hstack((detections, detection))
                    x = 1
                    
                else:
                    detection = np.array([[np.nan], [np.nan]])
                    detections = np.hstack((detections, detection))
            
            # if there is no object there
            else:
                # then append an empty
                detection = np.array([[np.nan], [np.nan]])
                detections = np.hstack((detections, detection))
                
            
            
            # clutter generation
        number_of_clutter_points = np.random.poisson(Parameters.clutter_rate)
        clutter = np.zeros((2,number_of_clutter_points))
        
        for i in range(number_of_clutter_points):
            clutter_point = selected_sensor.reshape(2,1) + (np.random.uniform(-1,1,(2,1)) * Parameters.sensor_radius)
            clutter[:,[i]] = clutter_point
        
        
        # combing the clutter and detections
        measurements = np.hstack((detections, clutter))
        
        for i in range(measurements.shape[1]):
            
            distance_from_sensor = np.sqrt(((selected_sensor[0] - measurements[:,i][0]) **2) + ((selected_sensor[1] - measurements[:,i][1]) **2))
                
            # if the measurement is outside of the FOV of the selected sensor
            if distance_from_sensor > Parameters.sensor_radius:
                
    
                # remove it (set = to nan)
                measurements[:,[i]] = np.array([[np.nan], [np.nan]])
                
                
        concatenated_measurements = np.concatenate((self.all_measurements[k], measurements), axis=1)
        
        # Find columns that contain NaN values
        nan_columns = np.any(np.isnan(concatenated_measurements), axis=0)
        
        # Remove columns with NaN values
        measurements_without_nans = concatenated_measurements[:, ~nan_columns]
        
        self.all_measurements[k] = measurements_without_nans
            
    
    def generate_observation(self, H, R, target_state, noise):
        
        # generates a detection of the target depending on where the 
        # target is located. Is used in generate_measurements()
        # returns a detection based on the target_state
        
        if noise == 'noise':
            noise = (np.linalg.cholesky(R) @ np.random.randn(H.shape[0], target_state.shape[1]))
            
        else:
            noise = np.zeros((R.shape[1], target_state.shape[1]))
          
        # checking if it is zeros i.e. empty
        obj = 0
        for i in target_state:
            
            if i != 0:
               observation = np.add((H @ target_state), noise)
               obj = 1
               break
        
        if obj != 1:
            observation = np.array([[0], [0]])
            
        return observation

    
    
class GOSPA_Sensor_updated_maths(Sensor_base_class):
    
    def __init__(self):
        super(GOSPA_Sensor_updated_maths, self).__init__()
        
        self.all_selected_sensors = []
        
        self.available_sensor_positions = []

        self.current_sensor_position = np.array([[0],[0]])
       
        self.synthetic_measurements = []
        
        self.selected_means = []
        
        self.all_available_actions_every_timestep = []
         

    # finds available positions to move to, based on a cicle, centred at the sensors current position
    def find_available_actions_fixed_distance(self, current_state, number_of_actions, speed, t, turn_limits = False, max_left_turn_deg = None, max_right_turn_deg = None,):
        
        all_available_states = []
        all_available_actions=[]
        
        # if there are no turning bounds imposed then take the full circle and divide by 
        # how many actions defined
        if turn_limits == False:
            
            theta = (2 * np.pi) / number_of_actions
            turn_angles = np.arange(0, 2 * np.pi, theta)
            
        
        # if there are turn limits, take that segment of the circle and divide it by how
        # many actions are defined, going up in turnlimitspread/numberofactions increments
        # going from max left to right
        else:
            
            max_left_turn_rad = -(np.deg2rad(max_left_turn_deg))
            max_right_turn_rad = np.deg2rad(max_right_turn_deg)
            
            theta = abs(max_left_turn_rad - max_right_turn_rad)/number_of_actions
            
            turn_angles = np.arange(max_left_turn_rad, max_right_turn_rad, theta)
        
        for angle in turn_angles:
            
            all_available_actions.append(angle)    
        
        # for obstacle in Parameters.obstacles_list:
        #     obstacle_omitted_actions = utils.check_for_obstacles_list(obstacle, all_available_actions, current_state)
        #     all_available_actions = obstacle_omitted_actions
            
        # for every action (angle) available, work out what the new state would be [[x],[y]]
        # if that angle was to be selected and append it to a list
        
        # if len(Parameters.obstacles_list) == 0:
        obstacle_omitted_actions = all_available_actions
            
        for angle in obstacle_omitted_actions:
            
            next_state = self.movement_model.fixed_distance_actions_transition(current_state, speed, t, angle)
            all_available_states.append(next_state)
    
        all_available_states.append(current_state)
 
        self.available_sensor_positions = all_available_states
        
        self.all_available_actions_every_timestep.append(all_available_states)
        return all_available_states
    

 
    def manage_multiple_sensors_mb_individually_MCTS(self, sensors, prob_existences, means_existing, covariances_existing, cost_function = None):
        cost_of_each_combination = []
        
        # extractiong positional information from the selected mean and cov
        positional_means = means_existing[::2]
        positional_covs = covariances_existing[:, [0, 2], :][:, :, [0, 2]]
        
        both_roots = []
        for sensor in sensors:
            
            root = MonteCarloTreeSearchNode_individual(parent=None, means_existing=positional_means, covariances_existing=positional_covs, sensor_mean=sensor.movement_model.state, prob_existences = prob_existences, cost = 0, cost_function = cost_function)
            selected_sensor_node = root.best_action()
            
            both_roots.append(root)
            sensor.movement_model.state = selected_sensor_node.sensor_mean
            sensor.all_selected_sensors.append(selected_sensor_node.sensor_mean)
            sensor.current_sensor_position = selected_sensor_node.sensor_mean

            self.movement_model.state = selected_sensor_node.sensor_mean
            
            self.all_selected_sensors.append(selected_sensor_node.sensor_mean)
            
            self.current_sensor_position = selected_sensor_node.sensor_mean

            cost_of_each_combination.append(selected_sensor_node.cost)
        
        return cost_of_each_combination, both_roots

        

    def manage_multiple_sensors_pmb_individually_MCTS(self, sensors, prob_existences, means_existing, covariances_existing, lambdau, means_new, covariances_new):
        cost_of_each_combination = []
        
        # Extractiong positional information from the selected mean and cov
        positional_means = means_existing[::2]
        positional_covs = covariances_existing[:, [0, 2], :][:, :, [0, 2]]
        
        positional_means_new = means_new[::2]
        positional_covariances_new = covariances_new[:, [0, 2], :][:, :, [0, 2]]
        
        both_roots = []
        for sensor in sensors:
            
            root = MonteCarloTreeSearchNode_individual(parent=None, means_existing=positional_means, covariances_existing=positional_covs, sensor_mean=sensor.movement_model.state, prob_existences = prob_existences, cost = 0, lambdau = lambdau, means_new = positional_means_new, covariances_new = positional_covariances_new)
            selected_sensor_node = root.best_action()
            
            both_roots.append(root)
            sensor.movement_model.state = selected_sensor_node.sensor_mean
            sensor.all_selected_sensors.append(selected_sensor_node.sensor_mean)
            sensor.current_sensor_position = selected_sensor_node.sensor_mean

            self.movement_model.state = selected_sensor_node.sensor_mean
            self.all_selected_sensors.append(selected_sensor_node.sensor_mean)
            self.current_sensor_position = selected_sensor_node.sensor_mean  
            
            cost_of_each_combination.append(selected_sensor_node.cost)
    
        return cost_of_each_combination, both_roots


    def manage_multiple_sensors_mb_full_jointly_calculated_MCTS(self, sensors, prob_existences, means_existing, covariances_existing, cost_function = None):
        # receives all of the active sensors as a list and all of the information calculated in the predict stage
        positional_means = means_existing[::2]
        positional_covs = covariances_existing[:, [0, 2], :][:, :, [0, 2]]
        
        cost_of_each_combination = []
        sensor_means = []
        
        for sensor in sensors:
            sensor_means.append(sensor.movement_model.state)
            
        root = MonteCarloTreeSearchNode_joint(parent=None, means_existing=positional_means, covariances_existing=positional_covs, sensor_means=sensor_means, prob_existences = prob_existences, cost = 0, cost_function = cost_function)
        selected_sensor_node = root.best_action()
        
        cost_of_each_combination.append(selected_sensor_node.cost)
  
        for sensor_number, sensor in enumerate(sensors):
            sensor.current_sensor_position = selected_sensor_node.sensor_means[sensor_number]
            sensor.movement_model.state = selected_sensor_node.sensor_means[sensor_number]
            sensor.all_selected_sensors.append(selected_sensor_node.sensor_means[sensor_number])

            
        self.all_selected_sensors.append(selected_sensor_node.sensor_means[sensor_number])
    
        return cost_of_each_combination, root

    def manage_multiple_sensors_pmb_full_jointly_calculated_MCTS(self, sensors, prob_existences, means_existing, covariances_existing,  lambdau, means_new, covariances_new,):
        # receives all of the active sensors as a list and all of the information calculated in the predict stage
        positional_means = means_existing[::2]
        positional_covs = covariances_existing[:, [0, 2], :][:, :, [0, 2]]
        
        positional_means_new = means_new[::2]
        positional_covariances_new = covariances_new[:, [0, 2], :][:, :, [0, 2]]
        
        cost_of_each_combination = []
        sensor_means = []
        
        for sensor in sensors:
            sensor_means.append(sensor.movement_model.state)
            
        root = MonteCarloTreeSearchNode_joint(parent=None, means_existing=positional_means, covariances_existing=positional_covs, sensor_means=sensor_means, prob_existences = prob_existences, cost = 0, means_new = positional_means_new, covariances_new = positional_covariances_new, lambdau = lambdau,)
        selected_sensor_node = root.best_action()
        
        cost_of_each_combination.append(selected_sensor_node.cost)
    
 
        for sensor_number, sensor in enumerate(sensors):
            sensor.current_sensor_position = selected_sensor_node.sensor_means[sensor_number]
            sensor.movement_model.state = selected_sensor_node.sensor_means[sensor_number]
            sensor.all_selected_sensors.append(selected_sensor_node.sensor_means[sensor_number])
       
        self.all_selected_sensors.append(selected_sensor_node.sensor_means[sensor_number])
        
        return cost_of_each_combination, root


    def find_actions_for_a_sensor(self):
              
        # find the available actions
        available_sensor_positions = self.find_available_actions_fixed_distance(self.movement_model.state, Parameters.number_of_actions, Parameters.sensor_speed, Parameters.t, turn_limits = False, max_left_turn_deg = None, max_right_turn_deg = None)

        available_actions = available_sensor_positions
        
        filtered_actions = [action for action in available_actions 
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
        
        return available_actions


    def joint_combination_cost(self, combination, means, covariances, pred_prob_existence):
        
        action1, action2 = combination
        
        # there are four hypotheses when using two sensors
        
        # S1     S2
        # empty  empty
        # empty   z
        # z      empty
        # z      z
        
        # We need to calculate the parameters associated with both hypotheses each time and then calculate a cost based on these
        
        cost_of_each_target = []
        
        # First calculate S1 empty parameters for each target
        for i, prediction in enumerate(means.T):
            
            pred_prob_existence_target = pred_prob_existence[i]
            
            P = covariances[i] #* Parameters.inflation_value
            x = prediction.reshape(2,-1)
            
            # Sensor position
            s = action1.reshape(2,-1)
            s2 = action2.reshape(2,-1)
            
            distance = np.linalg.norm(s - x)
            distance2 = np.linalg.norm(s2 - x)
            
            # Sensor FOV radius
            delta = Parameters.sensor_radius
            
             # Probability of detection: Gaussian-like tapering within the FOV and beyond
            if distance <= delta:
                pd_of_existing_target = Parameters.prob_detection * np.exp(-0.5 * (distance / delta) ** 2)
            else:
                # Outside FOV: Further tapering off
                tapering_factor = np.exp(-0.5 * (distance / delta) ** 2) * np.exp(-0.5 * ((distance - delta) / delta) ** 2)
                pd_of_existing_target = Parameters.prob_detection * tapering_factor
            
            
             # Probability of detection: Gaussian-like tapering within the FOV and beyond
            if distance2 <= delta:
                pd_of_existing_target2 = Parameters.prob_detection * np.exp(-0.5 * (distance2 / delta) ** 2)
            else:
                # Outside FOV: Further tapering off
                tapering_factor2 = np.exp(-0.5 * (distance2 / delta) ** 2) * np.exp(-0.5 * ((distance2 - delta) / delta) ** 2)
                pd_of_existing_target2 = Parameters.prob_detection * tapering_factor2
   
            prob_detection = pd_of_existing_target # of sensor one
            prob_detection2 = pd_of_existing_target2 # of sensor two
            
            
            # for Z = EMPTY
            # associated with there being no measurement received
            # The predictions are took forward
            no_measurement_mean = prediction # x_{k|k}^0
            no_measurement_covariance = covariances[i] # P_{k|k}^0
            no_measurement_prob_existence = ((1 - prob_detection) * pred_prob_existence_target) / ((1 - pred_prob_existence_target) + (( 1 - prob_detection) * pred_prob_existence_target)) # r_{k|k}^0
            
            # then the costs associciated with S2 having no measurement, GIVEN S1 being empty
            no_measurement_meanS2 = no_measurement_mean # x_{k|k}^0
            no_measurement_covarianceS2 = no_measurement_covariance # P_{k|k}^0
            no_measurement_prob_existenceS2 = ((1 - prob_detection2) * no_measurement_prob_existence) / ((1 - no_measurement_prob_existence) + (( 1 - prob_detection2) * no_measurement_prob_existence)) # r_{k|k}^0
            no_measurement_costS2 = self.individual_cost(no_measurement_prob_existenceS2, no_measurement_covarianceS2, Parameters.c) # C(r_{k|k}^0, P_{k|k}^0)
            
            # then the costs associciated with S2 having a measurement, GIVEN S1 being empty
            z_hat = (Parameters.H @ no_measurement_mean) + Parameters.bias # mean of the synthetic measurement
            S_ak = (Parameters.H @ no_measurement_covariance @ Parameters.H.T) + Parameters.R # cov of the synthetic measurement
            yes_measurement_meanS2 = no_measurement_mean # + ((cov @ Parameters.H.T @ np.linalg.inv(S_ak)) @ (synthetic_measurement - z_hat)) # mu_{k|k}^1
            yes_measurement_covarianceS2 = no_measurement_covariance - (no_measurement_covariance @ Parameters.H.T @ np.linalg.inv(S_ak) @ Parameters.H @ no_measurement_covariance) # Sigma_{k|k}^1
            yes_measurement_prob_existenceS2 = (no_measurement_prob_existence * prob_detection2 * Decimal((2 * np.pi)**(z_hat.shape[0]/2) * np.linalg.det(S_ak)**(-0.5))) / ((no_measurement_prob_existence * prob_detection2 * Decimal((2 * np.pi)**(z_hat.shape[0]/2) * np.linalg.det(S_ak)**(-0.5) + Parameters.clutter_density)))
            yes_measurement_costS2 = self.individual_cost(yes_measurement_prob_existenceS2, yes_measurement_covarianceS2, Parameters.c)
            
            # then S1 parameters being recorded a measurement

            # synthetic measurement
            z_hat = (Parameters.H @ prediction) + Parameters.bias # mean of the synthetic measurement
            S_ak = (Parameters.H @ covariances[i] @ Parameters.H.T) + Parameters.R # cov of the synthetic measurement
            yes_measurement_mean = prediction # + ((cov @ Parameters.H.T @ np.linalg.inv(S_ak)) @ (synthetic_measurement - z_hat)) # mu_{k|k}^1
            yes_measurement_covariance = covariances[i] - (covariances[i] @ Parameters.H.T @ np.linalg.inv(S_ak) @ Parameters.H @ covariances[i]) # Sigma_{k|k}^1
            yes_measurement_prob_existence = (pred_prob_existence_target * prob_detection * Decimal((2 * np.pi)**(z_hat.shape[0]/2) * np.linalg.det(S_ak)**(-0.5))) / ((pred_prob_existence_target * prob_detection * Decimal((2 * np.pi)**(z_hat.shape[0]/2) * np.linalg.det(S_ak)**(-0.5) + Parameters.clutter_density)))
            
            
            # then S2 not receiving a measurement, GIVEN S1 has
            no_measurement_meanS2m = yes_measurement_mean # x_{k|k}^0
            no_measurement_covarianceS2m = yes_measurement_covariance # P_{k|k}^0
            no_measurement_prob_existenceS2m = ((1 - prob_detection2) * yes_measurement_prob_existence) / ((1 - yes_measurement_prob_existence) + (( 1 - prob_detection2) * yes_measurement_prob_existence)) # r_{k|k}^0
            no_measurement_costS2m = self.individual_cost(no_measurement_prob_existenceS2m, no_measurement_covarianceS2m, Parameters.c) # C(r_{k|k}^0, P_{k|k}^0)
            
            
            
            # then S2 receiving a measurement, GIVEN S1 has also
            z_hatS2m = (Parameters.H @ yes_measurement_mean) + Parameters.bias # mean of the synthetic measurement
            S_akS2m = (Parameters.H @ yes_measurement_covariance @ Parameters.H.T) + Parameters.R # cov of the synthetic measurement
            yes_measurement_meanS2m = yes_measurement_mean # + ((cov @ Parameters.H.T @ np.linalg.inv(S_ak)) @ (synthetic_measurement - z_hatS2m)) # mu_{k|k}^1
            yes_measurement_covarianceS2m = yes_measurement_covariance - (yes_measurement_covariance @ Parameters.H.T @ np.linalg.inv(S_akS2m) @ Parameters.H @ yes_measurement_covariance) # Sigma_{k|k}^1
            yes_measurement_prob_existenceS2m = (yes_measurement_prob_existence * prob_detection2 * Decimal((2 * np.pi)**(z_hatS2m.shape[0]/2) * np.linalg.det(S_akS2m)**(-0.5))) / ((yes_measurement_prob_existence * prob_detection2 * Decimal((2 * np.pi)**(z_hatS2m.shape[0]/2) * np.linalg.det(S_akS2m)**(-0.5) + Parameters.clutter_density)))
            yes_measurement_costS2m = self.individual_cost(yes_measurement_prob_existenceS2m, yes_measurement_covarianceS2m, Parameters.c)
            
            
            
            cost_term_1 = ((1 - (pred_prob_existence_target * prob_detection)) * (1 - (no_measurement_prob_existence * prob_detection2))) * no_measurement_costS2
            cost_term_2 = (1 - (pred_prob_existence_target * prob_detection)) * (no_measurement_prob_existence * prob_detection2) * yes_measurement_costS2
            cost_term_3 = (pred_prob_existence_target * prob_detection) * (1 - (yes_measurement_prob_existence * prob_detection2)) * no_measurement_costS2m
            cost_term_4 = (pred_prob_existence_target * prob_detection) * (yes_measurement_prob_existence * prob_detection2) * yes_measurement_costS2m 
                           
            total_cost_target = cost_term_1 + cost_term_2 + cost_term_3 + cost_term_4

    
            cost_of_each_target.append(total_cost_target)

        total_cost = Decimal(np.sum(cost_of_each_target))

        return Decimal(total_cost)
        

    def GOSPA_cost(self, mean, cov, pred_prob_existence, prob_detection, c, bias):
        
        prob_detection = Decimal(prob_detection)
        pred_prob_existence = Decimal(pred_prob_existence)
        
        # synthetic measurement
        z_hat = (Parameters.H @ mean) + bias # mean of the synthetic measurement
        S_ak = (Parameters.H @ cov @ Parameters.H.T) + Parameters.R # cov of the synthetic measurement

        # generate the synthetic measurement and calculate whether or not the sensor would pick it up based on
        # its location and FOV
        synthetic_measurement = z_hat
        
        self.synthetic_measurements.append(synthetic_measurement)
        
        # for Z = EMPTY
        # associated with there being no measurement received
        # the predictions are took forward
        # no_measurement_mean = mean # x_{k|k}^0
        no_measurement_covariance = cov # P_{k|k}^0
        no_measurement_prob_existence = ((1 - prob_detection) * pred_prob_existence) / ((1 - pred_prob_existence) + (( 1 - prob_detection) * pred_prob_existence)) # r_{k|k}^0
        no_measurement_cost = self.individual_cost(no_measurement_prob_existence, no_measurement_covariance, Parameters.c) # C(r_{k|k}^0, P_{k|k}^0)
    
        # for  Z = {z_hat_{a_k}}
        # associated with there being a measurement received
        # the second term in yes_measurement_mean will dissappear as z_hat = synthetic_measurement
        # yes_measurement_mean = mean # + ((cov @ Parameters.H.T @ np.linalg.inv(S_ak)) @ (synthetic_measurement - z_hat)) # mu_{k|k}^1
        yes_measurement_covariance = cov - (cov @ Parameters.H.T @ np.linalg.inv(S_ak) @ Parameters.H @ cov) # Sigma_{k|k}^1
        
        yes_measurement_prob_existence = (pred_prob_existence * prob_detection * Decimal((2 * np.pi)**(z_hat.shape[0]/2) * np.linalg.det(S_ak)**(-0.5))) / ((pred_prob_existence * prob_detection * Decimal((2 * np.pi)**(z_hat.shape[0]/2) * np.linalg.det(S_ak)**(-0.5) + Parameters.clutter_density)))
        
        yes_measurement_cost = self.individual_cost(yes_measurement_prob_existence, yes_measurement_covariance, Parameters.c)
        
        # caluclating the total cost
        term1 = Decimal(1 - (pred_prob_existence * prob_detection)) * no_measurement_cost
        term2 = Decimal((prob_detection * pred_prob_existence)) * yes_measurement_cost
        
        total_cost_of_action = term1 + term2
        
        return total_cost_of_action, yes_measurement_cost, no_measurement_cost
    
    
    # for PPP targets
    def GOSPA_cost_new_targets(self, mean, cov, lambdau, prob_detection, c, bias):
   
        # synthetic measurement
        z_hat = (Parameters.H @ mean) + bias # mean of the synthetic measurement
        S_ak = (Parameters.H @ cov @ Parameters.H.T) + Parameters.R # cov of the synthetic measurement


        # generate the synthetic measurement and calculate whether or not the sensor would pick it up based on
        # its location and FOV
        synthetic_measurement = z_hat
        
        self.synthetic_measurements.append(synthetic_measurement)
        
        
        # no_measurement_mean = mean
        no_measurement_covariance = cov
        no_measurement_prob_existence = (1 - prob_detection) * lambdau
        no_measurement_cost = self.individual_cost(no_measurement_prob_existence, no_measurement_covariance, Parameters.c)
        
        # for  Z = {z_hat_{a_k}}
        # associated with there being a measurement received
        # the second term in yes_measurement_mean will dissappear as z_hat = synthetic_measurement
        # yes_measurement_mean = mean # + ((cov @ Parameters.H.T @ np.linalg.inv(S_ak)) @ (synthetic_measurement - z_hat)) # mu_{k|k}^1
        yes_measurement_covariance = cov - (cov @ Parameters.H.T @ np.linalg.inv(S_ak) @ Parameters.H @ cov) # Sigma_{k|k}^1
        
        # N(z;z_hat,S_ak)
        pdf_synthetic_measurement = scipy.stats.multivariate_normal(z_hat.squeeze(), S_ak).pdf(z_hat.squeeze())
        
        yes_measurement_prob_existence_numerator = lambdau * prob_detection * pdf_synthetic_measurement
        yes_measurement_prob_existence_denominator = Parameters.clutter_density + (lambdau * prob_detection * pdf_synthetic_measurement)
        
        # there are two expressions for this term in the note, this is the first one
        yes_measurement_prob_existence = yes_measurement_prob_existence_numerator / yes_measurement_prob_existence_denominator
        
        yes_measurement_cost = self.individual_cost(yes_measurement_prob_existence, yes_measurement_covariance, Parameters.c)
        
        # yes_measurement_cost = - yes_measurement_cost # inversing just this part
        
        # caluclating the total cost
        total_cost_of_action = no_measurement_cost + yes_measurement_cost
        
        return total_cost_of_action
     
    def individual_cost(self, prob_existence, measurement_covariance, c):
        
        prob_existence = str(prob_existence)
        prob_existence = Decimal(prob_existence)
        
        # breaking down the detection threshod into sub parts
        trace_part = Decimal(2) * (Decimal(np.trace(measurement_covariance))/Decimal(c**2)) 
        min_part = Decimal(2) - (np.nanmin((trace_part,Decimal(1))))
       
        optimal_detection_threshold = Decimal(1) / Decimal(min_part) # \Gamma_d^*
        # print(optimal_detection_threshold)
                
        if prob_existence > float(optimal_detection_threshold): # if  exist
            cost = (Decimal((c**2) / 2) * Decimal(1 - prob_existence)) +  (Decimal(prob_existence) * (np.nanmin((Decimal(np.trace(measurement_covariance)), Decimal(c**2)))))
  
        else: # if not exist
            cost = (Decimal(c**2) / 2) * (Decimal(prob_existence))
            
        return Decimal(cost)
    

    def merge_Gaussians(self, weights, means, covariances):
    
        merged_weight = np.nansum(weights)
        
        weighted_sum_of_means = []
        weighted_sum_of_covs = []
        
        for i in range(len(weights)):
            weighted_sum_of_means.append(weights[i] * means[:,i])
                
        weighted_sum_of_means = np.asarray(weighted_sum_of_means)
        weighted_sum_of_means = np.nansum(weighted_sum_of_means, axis = 0)
        merged_mean = (1/merged_weight) * weighted_sum_of_means
        merged_mean = merged_mean.reshape((4,1))
        
        for i in range(len(weights)):
            difference = means[:,[i]] - merged_mean
            weighted_sum_of_covs.append((weights[i]/merged_weight) * (covariances[i,:,:] + difference @ difference.T) )
            
        np.asarray(weighted_sum_of_covs)
        merged_covariance = np.nansum(weighted_sum_of_covs, axis = 0)
        
        return merged_weight, merged_mean, merged_covariance


