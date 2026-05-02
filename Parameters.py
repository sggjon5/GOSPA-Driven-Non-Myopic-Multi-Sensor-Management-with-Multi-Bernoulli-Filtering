# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:12:31 2023

@author: sggjone5
"""
import numpy as np

#################################################
#   S I M U L A T I O N   P A R A M E T E R S    #
#################################################
number_of_timesteps = 200

surveillance_area = np.array([[-250, 250],
                              [-250, 250]])

number_of_mc_runs = 50

#########################################
#   S E N S O R   P A R A M E T E R S   #
#########################################
sensor_radius = 40
number_of_actions = 6
sensor_speed = 15

proximity_distance = sensor_radius * 3 # joint optimisation threshold


#######################################
#   B I R T H   P A R A M E T E R S   #
#######################################
number_of_birth_densities = 1
prob_birth = [0.05]  * number_of_birth_densities # sets all birth densities with equal weighting
prob_existence_birth = np.array([[1]]) # targets born with existence probability of 1

initial_mean = np.array([[0],
                         [0],
                         [0],
                         [0]])

initial_covariance = np.zeros((1,4,4))
initial_covariance[[0],:,:] = np.diag([10, 10, 10, 10])

birth_means = np.zeros((initial_mean.shape[0], number_of_birth_densities))
birth_covariances = np.zeros((number_of_birth_densities,4,4))

birth_means[:,[0]] = np.array([[0],[0.1],[0],[0.1]]) # changing the birth location of birth density 0
birth_covariances[[0],:,:] = 0.2 * np.diag([30, 30, 30, 30])

# birth_means[:,[1]] = np.array([[-50],[0.1],[-120],[0.1]]) # changing the birth location of birth density 0
# birth_covariances[[1],:,:] = 0.2 * np.diag([30, 30, 30, 30])


#########################################
#   T A R G E T   P A R A M E T E R S   #
#########################################
prob_survival = 0.99 
prob_death = 1 - prob_survival

prob_detection = 0.999 # probability of detection when target is within sensor FOV

clutter_rate = 2 # defined as a poisson intensity

clutter_density = clutter_rate / (np.pi * (sensor_radius **2))


###############################################
#   F I L T E R I N G   P A R A M E T E R S   #
###############################################
existence_threshold = 0.5 # track existence probability threshold
pruning_threshold = 1e-4   # for existing targets

# PPP elements
lambdau = np.array([0]) # intiailised to how many targets you expect at timestep 0
lambdab = 0
lambda_false_alarm = clutter_density
lambda_birth_threshold = 1e-4 # threshold for low birth intensity

state_dimensions = birth_means.shape[0]

bias = 0

t = 1 # time step

# Transition matrix
F = np.array([[1, t, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, t],
              [0, 0, 0, 1]]) 

# Random variance in velocity
sigma_v = 0.8

# Process noise covariance matrix
Q = np.array([[(t**3)/3, (t**2)/2, 0, 0],
              [(t**2)/2, t, 0, 0],
              [0, 0, (t**3)/3, (t**2)/2],
              [0, 0, (t**2)/2, t]]) * (sigma_v) 

# Observation matrix
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

# Observation noise covariance
R = np.array([[1, 0],
              [0, 1]]) * 3


############################################################
#   P E R F O R M A N C E   Q U A N T I F I C A T I O N    #
############################################################
c = sensor_radius * 2 # maximum localistion error for GOSPA
p = 2 # outlier penalisaton term for GOSPA


######################################
#   M C T S   P A R A M E T E R S    #
######################################
best_child_c_param = 0.05 # epsilon for exploration versus exploitation weighting
simulation_search_depth = 3 # default non-myopic rollout length
MCTS_simulation_number = 150 # default budget for growing tree to this number of nodes
discount_factor = 0.9 # decay factor, often denoted as lambda in the literature


################################################
#   O B S T A C L E S   P A R A M E T E R S    #
################################################
# for flatblock 1
coordB3 = np.array([[30],[90]]) # top right
coordD3 = np.array([[-30],[60]]) # bottom left

# for flatblock 2
coordC3 = np.array([[40],[-75]]) # top right
coordA3 = np.array([[-25],[-95]]) # bottom left