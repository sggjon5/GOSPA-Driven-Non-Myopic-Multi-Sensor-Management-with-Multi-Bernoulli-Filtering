# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:40:39 2023 

This code base is for the centralised approach to multi-sensor sensor management
for Multi-Bernoulli (MB) filtering.

Each sensor (S1, S2) has the ability to conduct its own decision making, independent of the other and
the ability to share information to a centralised decision making module which jointly optimises the actions
taken by the sensors.

The filtering is centralised, and all sensors operate on shared information from a centralised, multi-Bernoulli
filtering loop, with a sequential update step, to avoid unnecessary information loss about the origin of
the measurements recieved by either sensor.

    
@author: sggjone5
"""
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


obstacle_seed = 1777 
non_obstacle_seed = 1001
new_seed = 123546 # 126 is good for 1 birth nonmyopic, 

selected_seed = new_seed
np.random.seed(selected_seed) 

from utils import update_anim_plot, merge_measurements, merge_gaussians, calculate_average_algorithm_GOSPA_error, GOSPA_breakdown_plots, save_frame_as_image, sample_sensor_birth_positions, save_error_results
import Parameters
from sensor import GOSPA_Sensor_updated_maths
import obstacles
from groundtruth import Groundtruth
from lbp import lbp
from tomb import tomb
from gospa import calculate_gospa

#%% Placeholder Variables & Obstacles
######################################################
# Placeholder variables
######################################################
# Obstacles
oFlatBlock = obstacles.obstacle_three(Parameters.coordD3, Parameters.coordB3)
obstacles_list = [oFlatBlock] # [] # empty list for no obstacles 
Parameters.obstacles_list = obstacles_list

# Filter parameters
all_means_for_plotting = []
all_covs_for_plotting = []
all_prob_existences = []
all_predictions_for_plotting = []
all_predictions_prob_exist_for_plotting = []
all_prediction_covs_for_plotting = []


# Plotting lists
information_shared = []
planning_nonmyopically = []
costs_list = []
all_the_tree_roots = []

# Errors
all_gospa = []
all_loc = [] # localisation
all_miss = [] # missed
all_fal = [] # false

# Flags
non_myopic = True
myopic = True

timestep_for_saving = 152

KLD = True

MCTS1 = True
MCTS1_budget_joint = 49
MCTS1_budget_individual = 7
MCTS1_lookahead = 5

MCTS2 = True
MCTS2_budget_joint = 49
MCTS2_budget_individual = 7
MCTS2_lookahead = 10

MCTS3 = True
MCTS3_budget_joint = 200
MCTS3_budget_individual = 40
MCTS3_lookahead = 5

MCTS4 = True
MCTS4_budget_joint = 200
MCTS4_budget_individual = 40
MCTS4_lookahead = 10

MCTS5 = False
MCTS5_budget_joint = 600
MCTS5_budget_individual = 100
MCTS5_lookahead = 5

#%% Groundtruth generation
######################################################
# Initialising the groundtruth
######################################################
oGroundtruth = Groundtruth()
oGroundtruth.generate_multitarget_groundtruth()


# %% MCTS - Myopic KLD
#################################
#   M C T S   -   M Y O P I C   K L D   #
#################################
np.random.seed(selected_seed) 

all_myopicKLD_gospa_error = []
all_myopicKLD_loc_gospa_error = []
all_myopicKLD_miss_gospa_error = []
all_myopicKLD_fal_gospa_error = []

start_time_myopicKLD = time.time()
myopic = True
non_myopic = False



for mc_run in range(Parameters.number_of_mc_runs):
    
    # Filter parameters
    all_means_for_plotting = []
    all_covs_for_plotting = []
    all_prob_existences = []
    all_predictions_for_plotting = []
    all_predictions_prob_exist_for_plotting = []
    all_prediction_covs_for_plotting = []


    # Plotting lists
    information_shared = []
    planning_nonmyopically = []
    costs_list = []
    all_the_tree_roots = []

    # Errors
    all_gospa = []
    all_loc = [] # localisation
    all_miss = [] # missed
    all_fal = [] # false
    
    oSensor1 =  GOSPA_Sensor_updated_maths() 
    oSensor2 = GOSPA_Sensor_updated_maths()
    oMajorSensor = GOSPA_Sensor_updated_maths() # centralised high level decision making module (does not have a physical presence in the simulation)
    sensor_list = [oSensor1, oSensor2]
    # Initialising birth locations of sensors (randomised x, fixed y)
    sensor_birth_positions = sample_sensor_birth_positions(
        obstacles_list=Parameters.obstacles_list,
        y_values=Parameters.sensor_birth_y_values,
        margin=Parameters.sensor_birth_x_margin,
        rng=np.random,
    )
    oSensor1.current_sensor_position = sensor_birth_positions[0]  # [x,y]
    oSensor2.current_sensor_position = sensor_birth_positions[1]  # [x,y]
    
    # Setting the starting position of each sensor internally
    oSensor1.movement_model.state = oSensor1.current_sensor_position.copy()
    oSensor2.movement_model.state = oSensor2.current_sensor_position.copy()
    
    ######################################################
    # Initialising the filter parameters
    ######################################################
    # Existing targets
    state_dimensions = Parameters.birth_means.shape[0]
    estimated_number_of_targets = Parameters.prob_birth[0] * Parameters.number_of_birth_densities
    prob_existence = [0] 
    means_existing = np.zeros(Parameters.birth_means[:,[0]].shape)
    covariances_existing = np.zeros(Parameters.birth_covariances[[0],:,:].shape)
    
    # PPP components (empty in MB case)
    lambdau = np.zeros(Parameters.lambdau.shape)
    means_new = np.zeros(Parameters.birth_means.shape)
    covariances_new = np.zeros(Parameters.birth_covariances.shape)
    
    # Filtering loop - MB (empty poisson)
    
    start_time = time.time()
    for k in range(Parameters.number_of_timesteps):
        
        #####################
        #   P R E D I C T   #
        #####################
        
        prob_existence, means_existing, covariances_existing = oMajorSensor.predictor.predict_pmb_empty_poisson(prob_existence.copy(), means_existing.copy(), covariances_existing.copy())
        
        all_predictions_prob_exist_for_plotting.append(prob_existence)
        all_predictions_for_plotting.append(means_existing)
        all_prediction_covs_for_plotting.append(covariances_existing)
        
        
        #########################################
        #   S E N S O R   M A N A G E M E N T   #
        #########################################
        # If the sensors are in close proximty to eachother
        if np.linalg.norm(oSensor1.current_sensor_position - oSensor2.current_sensor_position) < Parameters.proximity_distance:
            
            # Jointly optimise the sensors
            print('jointly optimised...')
                        
            if myopic == True:
                Parameters.MCTS_simulation_number = 49
                Parameters.simulation_search_depth = 1
                
                planning_nonmyopically.append(False)
                
            cost_of_each_combination, root = oMajorSensor.manage_multiple_sensors_mb_full_jointly_calculated_MCTS([oSensor1, oSensor2],  prob_existence, means_existing, covariances_existing, cost_function='KLD')
            all_the_tree_roots.append(root) 
            
            information_shared.append(True)
            
    
        # otherwise, complete independent sensor management   
        else:
            print('individually optimised...')
      
            if myopic == True:
                Parameters.MCTS_simulation_number = 7
                Parameters.simulation_search_depth = 1
                
                planning_nonmyopically.append(False)
                
            cost_of_each_combination, root = oMajorSensor.manage_multiple_sensors_mb_individually_MCTS([oSensor1, oSensor2], prob_existence, means_existing, covariances_existing, cost_function = 'KLD')
            all_the_tree_roots.append(root)
            
            information_shared.append(False)
       
        costs_list.append(cost_of_each_combination)
    
        #################################################
        #   G E N E R A T E   M E A S U R E M E N T S   #
        #################################################
        # Sensor 1
        oSensor1.measurements.generate_measurements_multiple_targets(np.array(oGroundtruth.target_states)[:,:,k].T, oSensor1.current_sensor_position, k)
        
        # Sensor 2
        oSensor2.measurements.generate_measurements_multiple_targets(np.array(oGroundtruth.target_states)[:,:,k].T, oSensor2.current_sensor_position, k)
        
        # Combining sensor measurements
        arrays = [oSensor1.measurements.all_measurements[k], oSensor2.measurements.all_measurements[k]] 
        combined_measurements = merge_measurements(arrays)
        oMajorSensor.measurements.all_measurements[k] = combined_measurements # pass this information to the central filtering loop
    
        ###################
        #   U P D A T E   #
        ###################
        # If multiple sensors, each do their own update here in a sequential manner
        
        # S1
        # Update
        lambdau_temp, means_new_temp, covariances_new_temp, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = oMajorSensor.updater.update_pmb_empty_poisson(lambdau, means_new, covariances_new, prob_existence, means_existing, covariances_existing, oSensor1.measurements.all_measurements[k], [oSensor1])
        # Loopy Beleif Propagation
        pupd, pnew = lbp(wupd, wnew)
        # Track oriented multi-Bernoulli
        prob_existence, means_existing, covariances_existing = tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
        
        prob_existence = prob_existence.reshape((-1,1)) # reshaping to allow for sequential update
        
        # S2
        # Update
        lambdau, means_new, covariances_new, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = oMajorSensor.updater.update_pmb_empty_poisson(lambdau_temp, means_new_temp, covariances_new_temp, prob_existence, means_existing, covariances_existing, oSensor2.measurements.all_measurements[k], [oSensor2])
        # Loopy Beleif Propagation
        pupd, pnew = lbp(wupd, wnew)
        # Track oriented multi-Bernoulli
        prob_existence, means_existing, covariances_existing = tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
    
        #####################
        #   M E R G I N G   #
        #####################
               
        if len(prob_existence) > 0:
          prob_existence, means_existing, covariances_existing = merge_gaussians(prob_existence, means_existing, covariances_existing)
        
        prob_existence[prob_existence >= 1] = 1
         
        # if len(lambdau) > 0:
        #   lambdau, means_new, covariances_new = merge_gaussians(lambdau, means_new, covariances_new)
        
        
        #####################
        #   P R U N I N G   #
        #####################
        # Existing targets
        filter_indices = prob_existence > Parameters.pruning_threshold
        prob_existence = prob_existence[filter_indices]
        means_existing = means_existing[:, filter_indices] 
        covariances_existing = covariances_existing[filter_indices,:,:]
        
        # PPP elements
        # filter_indices = lambdau > Parameters.pruning_threshold
        # lambdau = lambdau[filter_indices]
        # means_new = means_new[:, filter_indices] 
        # covariances_new = covariances_new[filter_indices,:,:]

        # Eliminate filter entries if they are outside of the surveillance region
        # Create the conditions for x and y coordinates
        x_condition = (means_existing[0, :] >= Parameters.surveillance_area[0, 0]) & (means_existing[0, :] <= Parameters.surveillance_area[0, 1])
        y_condition = (means_existing[2, :] >= Parameters.surveillance_area[1, 0]) & (means_existing[2, :] <= Parameters.surveillance_area[1, 1])
        
        # Combine the conditions
        combined_condition = x_condition & y_condition # central
    
        # Filter the array based on the combined condition 
        filtered_prob_existence = prob_existence[combined_condition]
        filtered_means_existing = means_existing[:, combined_condition] 
        filtered_covaiances_existing = covariances_existing[combined_condition,:,:]
        prob_existence = filtered_prob_existence.copy()
        means_existing = filtered_means_existing.copy()
        covariances_existing = filtered_covaiances_existing.copy()
               
        # Existence Thresholding
        ss = np.array(prob_existence) > Parameters.existence_threshold
        n = np.sum(ss)
      
        prob_existence_extracted = prob_existence[ss]
        means_existing_extracted = means_existing[:,ss]
        covariances_existing_extracted = covariances_existing[ss,:,:]
        
        
        # Store for plotting
        means_existing_for_plot = means_existing_extracted.copy()
        if means_existing_extracted.shape[1] == 0:
            means_existing_for_plot = np.array([[np.nan],
                                                [np.nan],
                                                [np.nan],
                                                [np.nan]])
            
        all_means_for_plotting.append(means_existing_extracted)
        all_prob_existences.append(prob_existence_extracted)
        all_covs_for_plotting.append(covariances_existing_extracted)
        
    
        #################################################
        #   C A L C U L A T E   G O S P A   E R R O R   #
        #################################################
        #Removing placeholder nans
        # Check which columns contain np.nan values
        groundtruths = np.array(oGroundtruth.target_states)[:,:,k].T
        nan_columns_gt = np.any(np.isnan(groundtruths), axis=0)
        nan_columns_filter_estimates = np.any(np.isnan(means_existing_extracted), axis=0)
        
        # Remove columns with np.nan values
        no_nan_groundtruth = groundtruths[:, ~nan_columns_gt]
        no_nan_filter_estimates = means_existing_extracted[:, ~nan_columns_filter_estimates]
        
        gospa, tracks_to_targets, gospa_loc, gospa_miss, gospa_false = calculate_gospa(no_nan_groundtruth.T, no_nan_filter_estimates.T, Parameters.c, Parameters.p)
        
        # gospa has already been **(1/p) in the function defenition but the localisation, missed and false have not
        gospa = gospa**2
        
        all_gospa.append(gospa)
        all_loc.append(gospa_loc)
        all_miss.append(gospa_miss)
        all_fal.append(gospa_false)
        
        #################################################
        print("errors calculated ", k , " myopic KLD \n")       

    all_myopicKLD_gospa_error.append(all_gospa)
    all_myopicKLD_loc_gospa_error.append(all_loc)
    all_myopicKLD_miss_gospa_error.append(all_miss)
    all_myopicKLD_fal_gospa_error.append(all_fal)
    
average_myopicKLD_error, localisation_myopicKLD_error, missed_myopicKLD_error, false_errorKLD_error = calculate_average_algorithm_GOSPA_error(all_myopicKLD_gospa_error, all_myopicKLD_loc_gospa_error, all_myopicKLD_miss_gospa_error, all_myopicKLD_fal_gospa_error)
fig_myopicKLD = GOSPA_breakdown_plots(all_myopicKLD_gospa_error, all_myopicKLD_loc_gospa_error, all_myopicKLD_miss_gospa_error, all_myopicKLD_fal_gospa_error, Parameters.number_of_timesteps, Parameters.c, fig=None, sensor_alg = "KLD - myopic")

# Save raw MC errors + per-timestep mean/std to ./results
myopicKLD_stats = save_error_results("myopicKLD", all_myopicKLD_gospa_error, all_myopicKLD_loc_gospa_error, all_myopicKLD_miss_gospa_error, all_myopicKLD_fal_gospa_error)
print('Myopic KLD: mean-over-time std(GOSPA) =', myopicKLD_stats['overall_std_gospa_mean_over_time'])


end_time_myopicKLD = time.time()
elapsed_time_myopicKLD = end_time_myopicKLD - start_time_myopicKLD

# Creating tuple of birth density information
birth_densities = (Parameters.birth_means.copy(), Parameters.birth_covariances.copy())

# Reformatting target track information for plots
target_tracks = oGroundtruth.target_states
target_tracks_matrix = np.array(target_tracks).reshape(-1, Parameters.number_of_timesteps)[::4]


#######################################
#   2 D   A N I M A T E D   P L O T   #
#######################################
# Creating figure for animated 2D plotting
fig, ax = plt.subplots()
plt.tight_layout()

ani = FuncAnimation(fig, update_anim_plot, frames=range(Parameters.number_of_timesteps), repeat=True, interval=100, fargs=(ax, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, [all_means_for_plotting], sensor_list, information_shared, planning_nonmyopically))

ani.save('animated_plot_myopicKLD.gif', writer='pillow', fps=10) # saves to root directory


# Snapshot of scenario
save_frame_as_image(timestep_for_saving, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, [all_means_for_plotting], sensor_list, information_shared, planning_nonmyopically, 'Myopic_scenario_snapshotKLD')

#######################################
#   3 D   A N I M A T E D   P L O T   #
#######################################
# Creating figure for 3D plotting
# fig3d = plt.figure()
# ax = fig3d.add_subplot(111, projection='3d')
# ani3d = FuncAnimation(fig3d, update_anim_plot_3d, frames=range(Parameters.number_of_timesteps), repeat=True, interval=100, fargs=(ax, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, all_means_for_plotting, sensor_list))
# ani3d.save('animated_plot_3d.gif', writer='pillow', fps=10) # saves to root directory


#########################################
#   G O S P A   E R R O R   P L O T S   #
#########################################
# fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1)
# ax1.plot(range(len(all_gospa)), all_gospa)
# ax2.plot(range(len(all_loc)), all_loc)
# ax3.plot(range(len(all_miss)), all_miss)
# ax4.plot(range(len(all_fal)), all_fal)

# ax1.set_title('GOSPA')
# ax2.set_title('Localisation error')
# ax3.set_title('Missed error')
# ax4.set_title('False error')
# ax1.set_ylim(-5, max(all_gospa)+20)
# ax2.set_ylim(-5, max(all_loc) + 20)
# ax3.set_ylim(-5, max(all_miss) + 20)
# ax4.set_ylim(-5, max(all_fal) + 20)
# plt.tight_layout()
# plt.savefig('GOSPA_error_plots.png')

# plt.show()


# %% MCTC - Myopic
#################################
#   M C T S   -   M Y O P I C   #
#################################
np.random.seed(selected_seed) 

all_myopic_gospa_error = []
all_myopic_loc_gospa_error = []
all_myopic_miss_gospa_error = []
all_myopic_fal_gospa_error = []

start_time_myopic = time.time()
myopic = True
non_myopic = False



for mc_run in range(Parameters.number_of_mc_runs):
    
    # Filter parameters
    all_means_for_plotting = []
    all_covs_for_plotting = []
    all_prob_existences = []
    all_predictions_for_plotting = []
    all_predictions_prob_exist_for_plotting = []
    all_prediction_covs_for_plotting = []


    # Plotting lists
    information_shared = []
    planning_nonmyopically = []
    costs_list = []
    all_the_tree_roots = []

    # Errors
    all_gospa = []
    all_loc = [] # localisation
    all_miss = [] # missed
    all_fal = [] # false
    
    oSensor1 =  GOSPA_Sensor_updated_maths() 
    oSensor2 = GOSPA_Sensor_updated_maths()
    oMajorSensor = GOSPA_Sensor_updated_maths() # centralised high level decision making module (does not have a physical presence in the simulation)
    sensor_list = [oSensor1, oSensor2]
    # Initialising birth locations of sensors (randomised x, fixed y)
    sensor_birth_positions = sample_sensor_birth_positions(
        obstacles_list=Parameters.obstacles_list,
        y_values=Parameters.sensor_birth_y_values,
        margin=Parameters.sensor_birth_x_margin,
        rng=np.random,
    )
    oSensor1.current_sensor_position = sensor_birth_positions[0]  # [x,y]
    oSensor2.current_sensor_position = sensor_birth_positions[1]  # [x,y]
    
    # Setting the starting position of each sensor internally
    oSensor1.movement_model.state = oSensor1.current_sensor_position.copy()
    oSensor2.movement_model.state = oSensor2.current_sensor_position.copy()
    
    ######################################################
    # Initialising the filter parameters
    ######################################################
    # Existing targets
    state_dimensions = Parameters.birth_means.shape[0]
    estimated_number_of_targets = Parameters.prob_birth[0] * Parameters.number_of_birth_densities
    prob_existence = [0] 
    means_existing = np.zeros(Parameters.birth_means[:,[0]].shape)
    covariances_existing = np.zeros(Parameters.birth_covariances[[0],:,:].shape)
    
    # PPP components (empty in MB case)
    lambdau = np.zeros(Parameters.lambdau.shape)
    means_new = np.zeros(Parameters.birth_means.shape)
    covariances_new = np.zeros(Parameters.birth_covariances.shape)
    
    # Filtering loop - MB (empty poisson)
    
    start_time = time.time()
    for k in range(Parameters.number_of_timesteps):
        
        #####################
        #   P R E D I C T   #
        #####################
        
        prob_existence, means_existing, covariances_existing = oMajorSensor.predictor.predict_pmb_empty_poisson(prob_existence.copy(), means_existing.copy(), covariances_existing.copy())
        
        all_predictions_prob_exist_for_plotting.append(prob_existence)
        all_predictions_for_plotting.append(means_existing)
        all_prediction_covs_for_plotting.append(covariances_existing)
        
        
        #########################################
        #   S E N S O R   M A N A G E M E N T   #
        #########################################
        # If the sensors are in close proximty to eachother
        if np.linalg.norm(oSensor1.current_sensor_position - oSensor2.current_sensor_position) < Parameters.proximity_distance:
            
            # Jointly optimise the sensors
            print('jointly optimised...')
                        
            if myopic == True:
                Parameters.MCTS_simulation_number = 49
                Parameters.simulation_search_depth = 1
                
                planning_nonmyopically.append(False)
                
            cost_of_each_combination, root = oMajorSensor.manage_multiple_sensors_mb_full_jointly_calculated_MCTS([oSensor1, oSensor2],  prob_existence, means_existing, covariances_existing, cost_function = 'GOSPA')
            all_the_tree_roots.append(root)
            
            information_shared.append(True)
            
    
        # otherwise, complete independent sensor management   
        else:
            print('individually optimised...')
      
            if myopic == True:
                Parameters.MCTS_simulation_number = 7
                Parameters.simulation_search_depth = 1
                
                planning_nonmyopically.append(False)
                
            cost_of_each_combination, root = oMajorSensor.manage_multiple_sensors_mb_individually_MCTS([oSensor1, oSensor2], prob_existence, means_existing, covariances_existing, cost_function = 'GOSPA')
            all_the_tree_roots.append(root)
            
            information_shared.append(False)
       
        costs_list.append(cost_of_each_combination)
    
        #################################################
        #   G E N E R A T E   M E A S U R E M E N T S   #
        #################################################
        # Sensor 1
        oSensor1.measurements.generate_measurements_multiple_targets(np.array(oGroundtruth.target_states)[:,:,k].T, oSensor1.current_sensor_position, k)
        
        # Sensor 2
        oSensor2.measurements.generate_measurements_multiple_targets(np.array(oGroundtruth.target_states)[:,:,k].T, oSensor2.current_sensor_position, k)
        
        # Combining sensor measurements
        arrays = [oSensor1.measurements.all_measurements[k], oSensor2.measurements.all_measurements[k]] 
        combined_measurements = merge_measurements(arrays)
        oMajorSensor.measurements.all_measurements[k] = combined_measurements # pass this information to the central filtering loop
    
        ###################
        #   U P D A T E   #
        ###################
        # If multiple sensors, each do their own update here in a sequential manner
        
        # S1
        # Update
        lambdau_temp, means_new_temp, covariances_new_temp, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = oMajorSensor.updater.update_pmb_empty_poisson(lambdau, means_new, covariances_new, prob_existence, means_existing, covariances_existing, oSensor1.measurements.all_measurements[k], [oSensor1])
        # Loopy Beleif Propagation
        pupd, pnew = lbp(wupd, wnew)
        # Track oriented multi-Bernoulli
        prob_existence, means_existing, covariances_existing = tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
        
        prob_existence = prob_existence.reshape((-1,1)) # reshaping to allow for sequential update
        
        # S2
        # Update
        lambdau, means_new, covariances_new, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = oMajorSensor.updater.update_pmb_empty_poisson(lambdau_temp, means_new_temp, covariances_new_temp, prob_existence, means_existing, covariances_existing, oSensor2.measurements.all_measurements[k], [oSensor2])
        # Loopy Beleif Propagation
        pupd, pnew = lbp(wupd, wnew)
        # Track oriented multi-Bernoulli
        prob_existence, means_existing, covariances_existing = tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
    
        #####################
        #   M E R G I N G   #
        #####################
               
        if len(prob_existence) > 0:
          prob_existence, means_existing, covariances_existing = merge_gaussians(prob_existence, means_existing, covariances_existing)
        
        prob_existence[prob_existence >= 1] = 1
         
        # if len(lambdau) > 0:
        #   lambdau, means_new, covariances_new = merge_gaussians(lambdau, means_new, covariances_new)
        
        
        #####################
        #   P R U N I N G   #
        #####################
        # Existing targets
        filter_indices = prob_existence > Parameters.pruning_threshold
        prob_existence = prob_existence[filter_indices]
        means_existing = means_existing[:, filter_indices] 
        covariances_existing = covariances_existing[filter_indices,:,:]
        
        # PPP elements
        # filter_indices = lambdau > Parameters.pruning_threshold
        # lambdau = lambdau[filter_indices]
        # means_new = means_new[:, filter_indices] 
        # covariances_new = covariances_new[filter_indices,:,:]

        # Eliminate filter entries if they are outside of the surveillance region
        # Create the conditions for x and y coordinates
        x_condition = (means_existing[0, :] >= Parameters.surveillance_area[0, 0]) & (means_existing[0, :] <= Parameters.surveillance_area[0, 1])
        y_condition = (means_existing[2, :] >= Parameters.surveillance_area[1, 0]) & (means_existing[2, :] <= Parameters.surveillance_area[1, 1])
        
        # Combine the conditions
        combined_condition = x_condition & y_condition # central
    
        # Filter the array based on the combined condition 
        filtered_prob_existence = prob_existence[combined_condition]
        filtered_means_existing = means_existing[:, combined_condition] 
        filtered_covaiances_existing = covariances_existing[combined_condition,:,:]
        prob_existence = filtered_prob_existence.copy()
        means_existing = filtered_means_existing.copy()
        covariances_existing = filtered_covaiances_existing.copy()
               
        # Existence Thresholding
        ss = np.array(prob_existence) > Parameters.existence_threshold
        n = np.sum(ss)
      
        prob_existence_extracted = prob_existence[ss]
        means_existing_extracted = means_existing[:,ss]
        covariances_existing_extracted = covariances_existing[ss,:,:]
        
        
        # Store for plotting
        means_existing_for_plot = means_existing_extracted.copy()
        if means_existing_extracted.shape[1] == 0:
            means_existing_for_plot = np.array([[np.nan],
                                                [np.nan],
                                                [np.nan],
                                                [np.nan]])
            
        all_means_for_plotting.append(means_existing_extracted)
        all_prob_existences.append(prob_existence_extracted)
        all_covs_for_plotting.append(covariances_existing_extracted)
        
    
        #################################################
        #   C A L C U L A T E   G O S P A   E R R O R   #
        #################################################
        #Removing placeholder nans
        # Check which columns contain np.nan values
        groundtruths = np.array(oGroundtruth.target_states)[:,:,k].T
        nan_columns_gt = np.any(np.isnan(groundtruths), axis=0)
        nan_columns_filter_estimates = np.any(np.isnan(means_existing_extracted), axis=0)
        
        # Remove columns with np.nan values
        no_nan_groundtruth = groundtruths[:, ~nan_columns_gt]
        no_nan_filter_estimates = means_existing_extracted[:, ~nan_columns_filter_estimates]
        
        gospa, tracks_to_targets, gospa_loc, gospa_miss, gospa_false = calculate_gospa(no_nan_groundtruth.T, no_nan_filter_estimates.T, Parameters.c, Parameters.p)
        
        # gospa has already been **(1/p) in the function defenition but the localisation, missed and false have not
        gospa = gospa**2
        
        all_gospa.append(gospa)
        all_loc.append(gospa_loc)
        all_miss.append(gospa_miss)
        all_fal.append(gospa_false)
        
        #################################################
        print("errors calculated ", k , " myopic \n")       

    all_myopic_gospa_error.append(all_gospa)
    all_myopic_loc_gospa_error.append(all_loc)
    all_myopic_miss_gospa_error.append(all_miss)
    all_myopic_fal_gospa_error.append(all_fal)
    
average_myopic_error, localisation_myopic_error, missed_myopic_error, false_error_error = calculate_average_algorithm_GOSPA_error(all_myopic_gospa_error, all_myopic_loc_gospa_error, all_myopic_miss_gospa_error, all_myopic_fal_gospa_error)
fig_myopic = GOSPA_breakdown_plots(all_myopic_gospa_error, all_myopic_loc_gospa_error, all_myopic_miss_gospa_error, all_myopic_fal_gospa_error, Parameters.number_of_timesteps, Parameters.c, fig=fig_myopicKLD, sensor_alg = "GD - myopic")

# Save raw MC errors + per-timestep mean/std to ./results
myopic_stats = save_error_results("myopic", all_myopic_gospa_error, all_myopic_loc_gospa_error, all_myopic_miss_gospa_error, all_myopic_fal_gospa_error)
print('Myopic: mean-over-time std(GOSPA) =', myopic_stats['overall_std_gospa_mean_over_time'])


end_time_myopic = time.time()
elapsed_time_myopic = end_time_myopic - start_time_myopic

# Creating tuple of birth density information
birth_densities = (Parameters.birth_means.copy(), Parameters.birth_covariances.copy())

# Reformatting target track information for plots
target_tracks = oGroundtruth.target_states
target_tracks_matrix = np.array(target_tracks).reshape(-1, Parameters.number_of_timesteps)[::4]


#######################################
#   2 D   A N I M A T E D   P L O T   #
#######################################
# Creating figure for animated 2D plotting
fig, ax = plt.subplots()
plt.tight_layout()

ani = FuncAnimation(fig, update_anim_plot, frames=range(Parameters.number_of_timesteps), repeat=True, interval=100, fargs=(ax, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, [all_means_for_plotting], sensor_list, information_shared, planning_nonmyopically))

ani.save('animated_plot_myopic.gif', writer='pillow', fps=10) # saves to root directory


# Snapshot of scenario
save_frame_as_image(timestep_for_saving, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, [all_means_for_plotting], sensor_list, information_shared, planning_nonmyopically, 'Myopic_scenario_snapshot')

#######################################
#   3 D   A N I M A T E D   P L O T   #
#######################################
# Creating figure for 3D plotting
# fig3d = plt.figure()
# ax = fig3d.add_subplot(111, projection='3d')
# ani3d = FuncAnimation(fig3d, update_anim_plot_3d, frames=range(Parameters.number_of_timesteps), repeat=True, interval=100, fargs=(ax, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, all_means_for_plotting, sensor_list))
# ani3d.save('animated_plot_3d.gif', writer='pillow', fps=10) # saves to root directory


#########################################
#   G O S P A   E R R O R   P L O T S   #
#########################################
# fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1)
# ax1.plot(range(len(all_gospa)), all_gospa)
# ax2.plot(range(len(all_loc)), all_loc)
# ax3.plot(range(len(all_miss)), all_miss)
# ax4.plot(range(len(all_fal)), all_fal)

# ax1.set_title('GOSPA')
# ax2.set_title('Localisation error')
# ax3.set_title('Missed error')
# ax4.set_title('False error')
# ax1.set_ylim(-5, max(all_gospa)+20)
# ax2.set_ylim(-5, max(all_loc) + 20)
# ax3.set_ylim(-5, max(all_miss) + 20)
# ax4.set_ylim(-5, max(all_fal) + 20)
# plt.tight_layout()
# plt.savefig('GOSPA_error_plots.png')

# plt.show()


# %% MCTS - Set 1
#################################
#   M C T S   -   S E T   1   #
#################################
if MCTS1 == True:
    np.random.seed(selected_seed) 
    
    all_MCTS1_gospa_error = []
    all_MCTS1_loc_gospa_error = []
    all_MCTS1_miss_gospa_error = []
    all_MCTS1_fal_gospa_error = []
    
    start_time_MCTS1 = time.time()
    myopic = False
    non_myopic = True
    
    # Filter parameters
    all_means_for_plotting = []
    all_covs_for_plotting = []
    all_prob_existences = []
    all_predictions_for_plotting = []
    all_predictions_prob_exist_for_plotting = []
    all_prediction_covs_for_plotting = []
    
    
    # Plotting lists
    information_shared = []
    planning_nonmyopically = []
    costs_list = []
    all_the_tree_roots = []
    
    # Errors
    all_gospa = []
    all_loc = [] # localisation
    all_miss = [] # missed
    all_fal = [] # false
    
    for mc_run in range(Parameters.number_of_mc_runs):
        
        # Filter parameters
        all_means_for_plotting = []
        all_covs_for_plotting = []
        all_prob_existences = []
        all_predictions_for_plotting = []
        all_predictions_prob_exist_for_plotting = []
        all_prediction_covs_for_plotting = []
    
    
        # Plotting lists
        information_shared = []
        planning_nonmyopically = []
        costs_list = []
        all_the_tree_roots = []
    
        # Errors
        all_gospa = []
        all_loc = [] # localisation
        all_miss = [] # missed
        all_fal = [] # false
        
        oSensor1 =  GOSPA_Sensor_updated_maths() 
        oSensor2 = GOSPA_Sensor_updated_maths()
        oMajorSensor = GOSPA_Sensor_updated_maths() # centralised high level decision making module (does not have a physical presence in the simulation)
        sensor_list = [oSensor1, oSensor2]
        # Initialising birth locations of sensors (randomised x, fixed y)
        sensor_birth_positions = sample_sensor_birth_positions(
            obstacles_list=Parameters.obstacles_list,
            y_values=Parameters.sensor_birth_y_values,
            margin=Parameters.sensor_birth_x_margin,
            rng=np.random,
        )
        oSensor1.current_sensor_position = sensor_birth_positions[0]  # [x,y]
        oSensor2.current_sensor_position = sensor_birth_positions[1]  # [x,y]
            
        # Setting the starting position of each sensor internally
        oSensor1.movement_model.state = oSensor1.current_sensor_position.copy()
        oSensor2.movement_model.state = oSensor2.current_sensor_position.copy()
        
        ######################################################
        # Initialising the filter parameters
        ######################################################
        # Existing targets
        state_dimensions = Parameters.birth_means.shape[0]
        estimated_number_of_targets = Parameters.prob_birth[0] * Parameters.number_of_birth_densities
        prob_existence = [0] 
        means_existing = np.zeros(Parameters.birth_means[:,[0]].shape)
        covariances_existing = np.zeros(Parameters.birth_covariances[[0],:,:].shape)
        
        # PPP components (empty in MB case)
        lambdau = np.zeros(Parameters.lambdau.shape)
        means_new = np.zeros(Parameters.birth_means.shape)
        covariances_new = np.zeros(Parameters.birth_covariances.shape)
        
        # Filtering loop - MB (empty poisson)
        
        start_time = time.time()
        for k in range(Parameters.number_of_timesteps):
            
            #####################
            #   P R E D I C T   #
            #####################
            
            prob_existence, means_existing, covariances_existing = oMajorSensor.predictor.predict_pmb_empty_poisson(prob_existence.copy(), means_existing.copy(), covariances_existing.copy())
            
            all_predictions_prob_exist_for_plotting.append(prob_existence)
            all_predictions_for_plotting.append(means_existing)
            all_prediction_covs_for_plotting.append(covariances_existing)
            
            
            #########################################
            #   S E N S O R   M A N A G E M E N T   #
            #########################################
            # If the sensors are in close proximty to eachother
            if np.linalg.norm(oSensor1.current_sensor_position - oSensor2.current_sensor_position) < Parameters.proximity_distance:
                
                # Jointly optimise the sensors
                print('jointly optimised...')
                
                if non_myopic == True:
                    Parameters.MCTS_simulation_number = MCTS1_budget_joint
                    Parameters.simulation_search_depth = MCTS1_lookahead
                    
                    planning_nonmyopically.append(True)
                    
                cost_of_each_combination, root = oMajorSensor.manage_multiple_sensors_mb_full_jointly_calculated_MCTS([oSensor1, oSensor2],  prob_existence, means_existing, covariances_existing, cost_function = 'GOSPA')
                all_the_tree_roots.append(root)
                
                information_shared.append(True)
                
        
            # otherwise, complete independent sensor management   
            else:
                print('individually optimised...')
                
                if non_myopic == True:
                    Parameters.MCTS_simulation_number = MCTS1_budget_individual
                    Parameters.simulation_search_depth = MCTS1_lookahead
                    
                    planning_nonmyopically.append(True)
              
                    
                cost_of_each_combination, root = oMajorSensor.manage_multiple_sensors_mb_individually_MCTS([oSensor1, oSensor2], prob_existence, means_existing, covariances_existing, cost_function = 'GOSPA')
                all_the_tree_roots.append(root)
                
                information_shared.append(False)
           
            costs_list.append(cost_of_each_combination)
        
            #################################################
            #   G E N E R A T E   M E A S U R E M E N T S   #
            #################################################
            # Sensor 1
            oSensor1.measurements.generate_measurements_multiple_targets(np.array(oGroundtruth.target_states)[:,:,k].T, oSensor1.current_sensor_position, k)
            
            # Sensor 2
            oSensor2.measurements.generate_measurements_multiple_targets(np.array(oGroundtruth.target_states)[:,:,k].T, oSensor2.current_sensor_position, k)
            
            # Combining sensor measurements
            arrays = [oSensor1.measurements.all_measurements[k], oSensor2.measurements.all_measurements[k]] 
            combined_measurements = merge_measurements(arrays)
            oMajorSensor.measurements.all_measurements[k] = combined_measurements # pass this information to the central filtering loop
        
            ###################
            #   U P D A T E   #
            ###################
            # If multiple sensors, each do their own update here in a sequential manner
            
            # S1
            # Update
            lambdau_temp, means_new_temp, covariances_new_temp, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = oMajorSensor.updater.update_pmb_empty_poisson(lambdau, means_new, covariances_new, prob_existence, means_existing, covariances_existing, oSensor1.measurements.all_measurements[k], [oSensor1])
            # Loopy Beleif Propagation
            pupd, pnew = lbp(wupd, wnew)
            # Track oriented multi-Bernoulli
            prob_existence, means_existing, covariances_existing = tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
            
            prob_existence = prob_existence.reshape((-1,1)) # reshaping to allow for sequential update
            
            # S2
            # Update
            lambdau, means_new, covariances_new, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = oMajorSensor.updater.update_pmb_empty_poisson(lambdau_temp, means_new_temp, covariances_new_temp, prob_existence, means_existing, covariances_existing, oSensor2.measurements.all_measurements[k], [oSensor2])
            # Loopy Beleif Propagation
            pupd, pnew = lbp(wupd, wnew)
            # Track oriented multi-Bernoulli
            prob_existence, means_existing, covariances_existing = tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
        
            #####################
            #   M E R G I N G   #
            #####################
            if len(prob_existence) > 0:
              prob_existence, means_existing, covariances_existing = merge_gaussians(prob_existence, means_existing, covariances_existing)
            
            # if len(lambdau) > 0:
            #   lambdau, means_new, covariances_new = merge_gaussians(lambdau, means_new, covariances_new)
            
            
            #####################
            #   P R U N I N G   #
            #####################
            # Existing targets
            filter_indices = prob_existence > Parameters.pruning_threshold
            prob_existence = prob_existence[filter_indices]
            means_existing = means_existing[:, filter_indices] 
            covariances_existing = covariances_existing[filter_indices,:,:]
            
            # PPP elements
            # filter_indices = lambdau > Parameters.pruning_threshold
            # lambdau = lambdau[filter_indices]
            # means_new = means_new[:, filter_indices] 
            # covariances_new = covariances_new[filter_indices,:,:]
    
            # Eliminate filter entries if they are outside of the surveillance region
            # Create the conditions for x and y coordinates
            x_condition = (means_existing[0, :] >= Parameters.surveillance_area[0, 0]) & (means_existing[0, :] <= Parameters.surveillance_area[0, 1])
            y_condition = (means_existing[2, :] >= Parameters.surveillance_area[1, 0]) & (means_existing[2, :] <= Parameters.surveillance_area[1, 1])
            
            # Combine the conditions
            combined_condition = x_condition & y_condition # central
        
            # Filter the array based on the combined condition 
            filtered_prob_existence = prob_existence[combined_condition]
            filtered_means_existing = means_existing[:, combined_condition] 
            filtered_covaiances_existing = covariances_existing[combined_condition,:,:]
            prob_existence = filtered_prob_existence.copy()
            means_existing = filtered_means_existing.copy()
            covariances_existing = filtered_covaiances_existing.copy()
                   
            # Existence Thresholding
            ss = np.array(prob_existence) > Parameters.existence_threshold
            n = np.sum(ss)
          
            prob_existence_extracted = prob_existence[ss]
            means_existing_extracted = means_existing[:,ss]
            covariances_existing_extracted = covariances_existing[ss,:,:]
            
            
            # Store for plotting
            means_existing_for_plot = means_existing_extracted.copy()
            if means_existing_extracted.shape[1] == 0:
                means_existing_for_plot = np.array([[np.nan],
                                                    [np.nan],
                                                    [np.nan],
                                                    [np.nan]])
                
            all_means_for_plotting.append(means_existing_extracted)
            all_prob_existences.append(prob_existence_extracted)
            all_covs_for_plotting.append(covariances_existing_extracted)
            
        
            #################################################
            #   C A L C U L A T E   G O S P A   E R R O R   #
            #################################################
            #Removing placeholder nans
            # Check which columns contain np.nan values
            groundtruths = np.array(oGroundtruth.target_states)[:,:,k].T
            nan_columns_gt = np.any(np.isnan(groundtruths), axis=0)
            nan_columns_filter_estimates = np.any(np.isnan(means_existing_extracted), axis=0)
            
            # Remove columns with np.nan values
            no_nan_groundtruth = groundtruths[:, ~nan_columns_gt]
            no_nan_filter_estimates = means_existing_extracted[:, ~nan_columns_filter_estimates]
            
            gospa, tracks_to_targets, gospa_loc, gospa_miss, gospa_false = calculate_gospa(no_nan_groundtruth.T, no_nan_filter_estimates.T, Parameters.c, Parameters.p)
            
            # gospa has already been **(1/p) in the function defenition but the localisation, missed and false have not
            gospa = gospa**2
            
            all_gospa.append(gospa)
            all_loc.append(gospa_loc)
            all_miss.append(gospa_miss)
            all_fal.append(gospa_false)
            
            #################################################
            print("errors calculated ", k , "MCTS 1 \n")       
    
        all_MCTS1_gospa_error.append(all_gospa)
        all_MCTS1_loc_gospa_error.append(all_loc)
        all_MCTS1_miss_gospa_error.append(all_miss)
        all_MCTS1_fal_gospa_error.append(all_fal)
        
    average_MCTS1_error, localisation_MCTS1_error, missed_MCTS1_error, false_MCTS1_error = calculate_average_algorithm_GOSPA_error(all_MCTS1_gospa_error, all_MCTS1_loc_gospa_error, all_MCTS1_miss_gospa_error, all_MCTS1_fal_gospa_error)
    fig_MCTS1 = GOSPA_breakdown_plots(all_MCTS1_gospa_error, all_MCTS1_loc_gospa_error, all_MCTS1_miss_gospa_error, all_MCTS1_fal_gospa_error, Parameters.number_of_timesteps, Parameters.c, fig=fig_myopic, sensor_alg = "GD - MCTS1")

    # Save raw MC errors + per-timestep mean/std to ./results
    MCTS1_stats = save_error_results("MCTS1", all_MCTS1_gospa_error, all_MCTS1_loc_gospa_error, all_MCTS1_miss_gospa_error, all_MCTS1_fal_gospa_error)
    print('MCTS1: mean-over-time std(GOSPA) =', MCTS1_stats['overall_std_gospa_mean_over_time'])

    
    end_time_MCTS1 = time.time()
    elapsed_time_MCTS1 = end_time_MCTS1 - start_time_MCTS1
    
    # Creating tuple of birth density information
    birth_densities = (Parameters.birth_means.copy(), Parameters.birth_covariances.copy())
    
    # Reformatting target track information for plots
    target_tracks = oGroundtruth.target_states
    target_tracks_matrix = np.array(target_tracks).reshape(-1, Parameters.number_of_timesteps)[::4]
    
    
    #######################################
    #   2 D   A N I M A T E D   P L O T   #
    #######################################
    # Creating figure for animated 2D plotting
    fig, ax = plt.subplots()
    plt.tight_layout()
    
    ani = FuncAnimation(fig, update_anim_plot, frames=range(Parameters.number_of_timesteps), repeat=True, interval=100, fargs=(ax, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, [all_means_for_plotting], sensor_list, information_shared, planning_nonmyopically))
    
    ani.save('animated_plot_MCTS1.gif', writer='pillow', fps=10) # saves to root directory
    
    # Snapshot of scenario
    save_frame_as_image(timestep_for_saving, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, [all_means_for_plotting], sensor_list, information_shared, planning_nonmyopically, 'MCTS1_scenario_snapshot')


    #######################################
    #   3 D   A N I M A T E D   P L O T   #
    #######################################
    # Creating figure for 3D plotting
    # fig3d = plt.figure()
    # ax = fig3d.add_subplot(111, projection='3d')
    # ani3d = FuncAnimation(fig3d, update_anim_plot_3d, frames=range(Parameters.number_of_timesteps), repeat=True, interval=100, fargs=(ax, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, all_means_for_plotting, sensor_list))
    # ani3d.save('animated_plot_3d.gif', writer='pillow', fps=10) # saves to root directory
    
    
    #########################################
    #   G O S P A   E R R O R   P L O T S   #
    #########################################
    # fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1)
    # ax1.plot(range(len(all_gospa)), all_gospa)
    # ax2.plot(range(len(all_loc)), all_loc)
    # ax3.plot(range(len(all_miss)), all_miss)
    # ax4.plot(range(len(all_fal)), all_fal)
    
    # ax1.set_title('GOSPA')
    # ax2.set_title('Localisation error')
    # ax3.set_title('Missed error')
    # ax4.set_title('False error')
    # ax1.set_ylim(-5, max(all_gospa)+20)
    # ax2.set_ylim(-5, max(all_loc) + 20)
    # ax3.set_ylim(-5, max(all_miss) + 20)
    # ax4.set_ylim(-5, max(all_fal) + 20)
    # plt.tight_layout()
    # plt.savefig('GOSPA_error_plots.png')
    
    # plt.show()




# %% MCTS - Set 2
#################################
#   M C T S   -   S E T   2   #
#################################
if MCTS2 == True:
    np.random.seed(selected_seed) 
    
    all_MCTS2_gospa_error = []
    all_MCTS2_loc_gospa_error = []
    all_MCTS2_miss_gospa_error = []
    all_MCTS2_fal_gospa_error = []
    
    start_time_MCTS2 = time.time()
    myopic = False
    non_myopic = True
    
    # Filter parameters
    all_means_for_plotting = []
    all_covs_for_plotting = []
    all_prob_existences = []
    all_predictions_for_plotting = []
    all_predictions_prob_exist_for_plotting = []
    all_prediction_covs_for_plotting = []
    
    
    # Plotting lists
    information_shared = []
    planning_nonmyopically = []
    costs_list = []
    all_the_tree_roots = []
    
    # Errors
    all_gospa = []
    all_loc = [] # localisation
    all_miss = [] # missed
    all_fal = [] # false
    
    for mc_run in range(Parameters.number_of_mc_runs):
        
        # Filter parameters
        all_means_for_plotting = []
        all_covs_for_plotting = []
        all_prob_existences = []
        all_predictions_for_plotting = []
        all_predictions_prob_exist_for_plotting = []
        all_prediction_covs_for_plotting = []
    
    
        # Plotting lists
        information_shared = []
        planning_nonmyopically = []
        costs_list = []
        all_the_tree_roots = []
    
        # Errors
        all_gospa = []
        all_loc = [] # localisation
        all_miss = [] # missed
        all_fal = [] # false
        
        oSensor1 =  GOSPA_Sensor_updated_maths() 
        oSensor2 = GOSPA_Sensor_updated_maths()
        oMajorSensor = GOSPA_Sensor_updated_maths() # centralised high level decision making module (does not have a physical presence in the simulation)
        sensor_list = [oSensor1, oSensor2]
        # Initialising birth locations of sensors (randomised x, fixed y)
        sensor_birth_positions = sample_sensor_birth_positions(
            obstacles_list=Parameters.obstacles_list,
            y_values=Parameters.sensor_birth_y_values,
            margin=Parameters.sensor_birth_x_margin,
            rng=np.random,
        )
        oSensor1.current_sensor_position = sensor_birth_positions[0]  # [x,y]
        oSensor2.current_sensor_position = sensor_birth_positions[1]  # [x,y]
        
        # Setting the starting position of each sensor internally
        oSensor1.movement_model.state = oSensor1.current_sensor_position.copy()
        oSensor2.movement_model.state = oSensor2.current_sensor_position.copy()
        
        ######################################################
        # Initialising the filter parameters
        ######################################################
        # Existing targets
        state_dimensions = Parameters.birth_means.shape[0]
        estimated_number_of_targets = Parameters.prob_birth[0] * Parameters.number_of_birth_densities
        prob_existence = [0] 
        means_existing = np.zeros(Parameters.birth_means[:,[0]].shape)
        covariances_existing = np.zeros(Parameters.birth_covariances[[0],:,:].shape)
        
        # PPP components (empty in MB case)
        lambdau = np.zeros(Parameters.lambdau.shape)
        means_new = np.zeros(Parameters.birth_means.shape)
        covariances_new = np.zeros(Parameters.birth_covariances.shape)
        
        # Filtering loop - MB (empty poisson)
        
        start_time = time.time()
        for k in range(Parameters.number_of_timesteps):
            
            #####################
            #   P R E D I C T   #
            #####################
            
            prob_existence, means_existing, covariances_existing = oMajorSensor.predictor.predict_pmb_empty_poisson(prob_existence.copy(), means_existing.copy(), covariances_existing.copy())
            
            all_predictions_prob_exist_for_plotting.append(prob_existence)
            all_predictions_for_plotting.append(means_existing)
            all_prediction_covs_for_plotting.append(covariances_existing)
            
            
            #########################################
            #   S E N S O R   M A N A G E M E N T   #
            #########################################
            # If the sensors are in close proximty to eachother
            if np.linalg.norm(oSensor1.current_sensor_position - oSensor2.current_sensor_position) < Parameters.proximity_distance:
                
                # Jointly optimise the sensors
                print('jointly optimised...')
                
                if non_myopic == True:
                    Parameters.MCTS_simulation_number = MCTS2_budget_joint
                    Parameters.simulation_search_depth = MCTS2_lookahead
                    
                    planning_nonmyopically.append(True)
                    
                cost_of_each_combination, root = oMajorSensor.manage_multiple_sensors_mb_full_jointly_calculated_MCTS([oSensor1, oSensor2],  prob_existence, means_existing, covariances_existing, cost_function = 'GOSPA')
                all_the_tree_roots.append(root)
                
                information_shared.append(True)
                
        
            # otherwise, complete independent sensor management   
            else:
                print('individually optimised...')
                
                if non_myopic == True:
                    Parameters.MCTS_simulation_number = MCTS2_budget_individual
                    Parameters.simulation_search_depth = MCTS2_lookahead
                    
                    planning_nonmyopically.append(True)
              
                    
                cost_of_each_combination, root = oMajorSensor.manage_multiple_sensors_mb_individually_MCTS([oSensor1, oSensor2], prob_existence, means_existing, covariances_existing, cost_function = 'GOSPA')
                all_the_tree_roots.append(root)
                
                information_shared.append(False)
           
            costs_list.append(cost_of_each_combination)
        
            #################################################
            #   G E N E R A T E   M E A S U R E M E N T S   #
            #################################################
            # Sensor 1
            oSensor1.measurements.generate_measurements_multiple_targets(np.array(oGroundtruth.target_states)[:,:,k].T, oSensor1.current_sensor_position, k)
            
            # Sensor 2
            oSensor2.measurements.generate_measurements_multiple_targets(np.array(oGroundtruth.target_states)[:,:,k].T, oSensor2.current_sensor_position, k)
            
            # Combining sensor measurements
            arrays = [oSensor1.measurements.all_measurements[k], oSensor2.measurements.all_measurements[k]] 
            combined_measurements = merge_measurements(arrays)
            oMajorSensor.measurements.all_measurements[k] = combined_measurements # pass this information to the central filtering loop
        
            ###################
            #   U P D A T E   #
            ###################
            # If multiple sensors, each do their own update here in a sequential manner
            
            # S1
            # Update
            lambdau_temp, means_new_temp, covariances_new_temp, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = oMajorSensor.updater.update_pmb_empty_poisson(lambdau, means_new, covariances_new, prob_existence, means_existing, covariances_existing, oSensor1.measurements.all_measurements[k], [oSensor1])
            # Loopy Beleif Propagation
            pupd, pnew = lbp(wupd, wnew)
            # Track oriented multi-Bernoulli
            prob_existence, means_existing, covariances_existing = tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
            
            prob_existence = prob_existence.reshape((-1,1)) # reshaping to allow for sequential update
            
            # S2
            # Update
            lambdau, means_new, covariances_new, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = oMajorSensor.updater.update_pmb_empty_poisson(lambdau_temp, means_new_temp, covariances_new_temp, prob_existence, means_existing, covariances_existing, oSensor2.measurements.all_measurements[k], [oSensor2])
            # Loopy Beleif Propagation
            pupd, pnew = lbp(wupd, wnew)
            # Track oriented multi-Bernoulli
            prob_existence, means_existing, covariances_existing = tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
        
            #####################
            #   M E R G I N G   #
            #####################
            if len(prob_existence) > 0:
              prob_existence, means_existing, covariances_existing = merge_gaussians(prob_existence, means_existing, covariances_existing)
            
            # if len(lambdau) > 0:
            #   lambdau, means_new, covariances_new = merge_gaussians(lambdau, means_new, covariances_new)
            
            
            #####################
            #   P R U N I N G   #
            #####################
            # Existing targets
            filter_indices = prob_existence > Parameters.pruning_threshold
            prob_existence = prob_existence[filter_indices]
            means_existing = means_existing[:, filter_indices] 
            covariances_existing = covariances_existing[filter_indices,:,:]
            
            # PPP elements
            # filter_indices = lambdau > Parameters.pruning_threshold
            # lambdau = lambdau[filter_indices]
            # means_new = means_new[:, filter_indices] 
            # covariances_new = covariances_new[filter_indices,:,:]
    
            # Eliminate filter entries if they are outside of the surveillance region
            # Create the conditions for x and y coordinates
            x_condition = (means_existing[0, :] >= Parameters.surveillance_area[0, 0]) & (means_existing[0, :] <= Parameters.surveillance_area[0, 1])
            y_condition = (means_existing[2, :] >= Parameters.surveillance_area[1, 0]) & (means_existing[2, :] <= Parameters.surveillance_area[1, 1])
            
            # Combine the conditions
            combined_condition = x_condition & y_condition # central
        
            # Filter the array based on the combined condition 
            filtered_prob_existence = prob_existence[combined_condition]
            filtered_means_existing = means_existing[:, combined_condition] 
            filtered_covaiances_existing = covariances_existing[combined_condition,:,:]
            prob_existence = filtered_prob_existence.copy()
            means_existing = filtered_means_existing.copy()
            covariances_existing = filtered_covaiances_existing.copy()
                   
            # Existence Thresholding
            ss = np.array(prob_existence) > Parameters.existence_threshold
            n = np.sum(ss)
          
            prob_existence_extracted = prob_existence[ss]
            means_existing_extracted = means_existing[:,ss]
            covariances_existing_extracted = covariances_existing[ss,:,:]
            
            
            # Store for plotting
            means_existing_for_plot = means_existing_extracted.copy()
            if means_existing_extracted.shape[1] == 0:
                means_existing_for_plot = np.array([[np.nan],
                                                    [np.nan],
                                                    [np.nan],
                                                    [np.nan]])
                
            all_means_for_plotting.append(means_existing_extracted)
            all_prob_existences.append(prob_existence_extracted)
            all_covs_for_plotting.append(covariances_existing_extracted)
            
        
            #################################################
            #   C A L C U L A T E   G O S P A   E R R O R   #
            #################################################
            #Removing placeholder nans
            # Check which columns contain np.nan values
            groundtruths = np.array(oGroundtruth.target_states)[:,:,k].T
            nan_columns_gt = np.any(np.isnan(groundtruths), axis=0)
            nan_columns_filter_estimates = np.any(np.isnan(means_existing_extracted), axis=0)
            
            # Remove columns with np.nan values
            no_nan_groundtruth = groundtruths[:, ~nan_columns_gt]
            no_nan_filter_estimates = means_existing_extracted[:, ~nan_columns_filter_estimates]
            
            gospa, tracks_to_targets, gospa_loc, gospa_miss, gospa_false = calculate_gospa(no_nan_groundtruth.T, no_nan_filter_estimates.T, Parameters.c, Parameters.p)
            
            # gospa has already been **(1/p) in the function defenition but the localisation, missed and false have not
            gospa = gospa**2
            
            all_gospa.append(gospa)
            all_loc.append(gospa_loc)
            all_miss.append(gospa_miss)
            all_fal.append(gospa_false)
            
            #################################################
            print("errors calculated ", k , " MCTS 2 \n")       
    
        all_MCTS2_gospa_error.append(all_gospa)
        all_MCTS2_loc_gospa_error.append(all_loc)
        all_MCTS2_miss_gospa_error.append(all_miss)
        all_MCTS2_fal_gospa_error.append(all_fal)
        
    average_MCTS2_error, localisation_MCTS2_error, missed_MCTS2_error, false_MCTS2_error = calculate_average_algorithm_GOSPA_error(all_MCTS2_gospa_error, all_MCTS2_loc_gospa_error, all_MCTS2_miss_gospa_error, all_MCTS2_fal_gospa_error)
    fig_MCTS2 = GOSPA_breakdown_plots(all_MCTS2_gospa_error, all_MCTS2_loc_gospa_error, all_MCTS2_miss_gospa_error, all_MCTS2_fal_gospa_error, Parameters.number_of_timesteps, Parameters.c, fig=fig_MCTS1, sensor_alg = "GD - MCTS2")

    # Save raw MC errors + per-timestep mean/std to ./results
    MCTS2_stats = save_error_results("MCTS2", all_MCTS2_gospa_error, all_MCTS2_loc_gospa_error, all_MCTS2_miss_gospa_error, all_MCTS2_fal_gospa_error)
    print('MCTS2: mean-over-time std(GOSPA) =', MCTS2_stats['overall_std_gospa_mean_over_time'])

    
    end_time_MCTS2 = time.time()
    elapsed_time_MCTS2 = end_time_MCTS2 - start_time_MCTS2
    
    # Creating tuple of birth density information
    birth_densities = (Parameters.birth_means.copy(), Parameters.birth_covariances.copy())
    
    # Reformatting target track information for plots
    target_tracks = oGroundtruth.target_states
    target_tracks_matrix = np.array(target_tracks).reshape(-1, Parameters.number_of_timesteps)[::4]
    
    
    #######################################
    #   2 D   A N I M A T E D   P L O T   #
    #######################################
    # Creating figure for animated 2D plotting
    fig, ax = plt.subplots()
    plt.tight_layout()
    
    ani = FuncAnimation(fig, update_anim_plot, frames=range(Parameters.number_of_timesteps), repeat=True, interval=100, fargs=(ax, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, [all_means_for_plotting], sensor_list, information_shared, planning_nonmyopically))
    
    ani.save('animated_plot_MCTS2.gif', writer='pillow', fps=10) # saves to root directory
    
    # Snapshot of scenario
    save_frame_as_image(timestep_for_saving, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, [all_means_for_plotting], sensor_list, information_shared, planning_nonmyopically, 'MCTS2_scenario_snapshot')

    
    #######################################
    #   3 D   A N I M A T E D   P L O T   #
    #######################################
    # Creating figure for 3D plotting
    # fig3d = plt.figure()
    # ax = fig3d.add_subplot(111, projection='3d')
    # ani3d = FuncAnimation(fig3d, update_anim_plot_3d, frames=range(Parameters.number_of_timesteps), repeat=True, interval=100, fargs=(ax, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, all_means_for_plotting, sensor_list))
    # ani3d.save('animated_plot_3d.gif', writer='pillow', fps=10) # saves to root directory
    
    
    #########################################
    #   G O S P A   E R R O R   P L O T S   #
    #########################################
    # fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1)
    # ax1.plot(range(len(all_gospa)), all_gospa)
    # ax2.plot(range(len(all_loc)), all_loc)
    # ax3.plot(range(len(all_miss)), all_miss)
    # ax4.plot(range(len(all_fal)), all_fal)
    
    # ax1.set_title('GOSPA')
    # ax2.set_title('Localisation error')
    # ax3.set_title('Missed error')
    # ax4.set_title('False error')
    # ax1.set_ylim(-5, max(all_gospa)+20)
    # ax2.set_ylim(-5, max(all_loc) + 20)
    # ax3.set_ylim(-5, max(all_miss) + 20)
    # ax4.set_ylim(-5, max(all_fal) + 20)
    # plt.tight_layout()
    # plt.savefig('GOSPA_error_plots.png')
    
    # plt.show()


# %% MCTS - Set 3
#################################
#   M C T S   -   S E T   3   #
#################################
if MCTS3 == True:
    np.random.seed(selected_seed) 
    
    all_MCTS3_gospa_error = []
    all_MCTS3_loc_gospa_error = []
    all_MCTS3_miss_gospa_error = []
    all_MCTS3_fal_gospa_error = []
    
    start_time_MCTS3 = time.time()
    myopic = False
    non_myopic = True
    
    # Filter parameters
    all_means_for_plotting = []
    all_covs_for_plotting = []
    all_prob_existences = []
    all_predictions_for_plotting = []
    all_predictions_prob_exist_for_plotting = []
    all_prediction_covs_for_plotting = []
    
    
    # Plotting lists
    information_shared = []
    planning_nonmyopically = []
    costs_list = []
    all_the_tree_roots = []
    
    # Errors
    all_gospa = []
    all_loc = [] # localisation
    all_miss = [] # missed
    all_fal = [] # false
    
    for mc_run in range(Parameters.number_of_mc_runs):
        
        # Filter parameters
        all_means_for_plotting = []
        all_covs_for_plotting = []
        all_prob_existences = []
        all_predictions_for_plotting = []
        all_predictions_prob_exist_for_plotting = []
        all_prediction_covs_for_plotting = []
    
    
        # Plotting lists
        information_shared = []
        planning_nonmyopically = []
        costs_list = []
        all_the_tree_roots = []
    
        # Errors
        all_gospa = []
        all_loc = [] # localisation
        all_miss = [] # missed
        all_fal = [] # false
        
        oSensor1 =  GOSPA_Sensor_updated_maths() 
        oSensor2 = GOSPA_Sensor_updated_maths()
        oMajorSensor = GOSPA_Sensor_updated_maths() # centralised high level decision making module (does not have a physical presence in the simulation)
        sensor_list = [oSensor1, oSensor2]
        # Initialising birth locations of sensors (randomised x, fixed y)
        sensor_birth_positions = sample_sensor_birth_positions(
            obstacles_list=Parameters.obstacles_list,
            y_values=Parameters.sensor_birth_y_values,
            margin=Parameters.sensor_birth_x_margin,
            rng=np.random,
        )
        oSensor1.current_sensor_position = sensor_birth_positions[0]  # [x,y]
        oSensor2.current_sensor_position = sensor_birth_positions[1]  # [x,y]
            
        # Setting the starting position of each sensor internally
        oSensor1.movement_model.state = oSensor1.current_sensor_position.copy()
        oSensor2.movement_model.state = oSensor2.current_sensor_position.copy()
        
        ######################################################
        # Initialising the filter parameters
        ######################################################
        # Existing targets
        state_dimensions = Parameters.birth_means.shape[0]
        estimated_number_of_targets = Parameters.prob_birth[0] * Parameters.number_of_birth_densities
        prob_existence = [0] 
        means_existing = np.zeros(Parameters.birth_means[:,[0]].shape)
        covariances_existing = np.zeros(Parameters.birth_covariances[[0],:,:].shape)
        
        # PPP components (empty in MB case)
        lambdau = np.zeros(Parameters.lambdau.shape)
        means_new = np.zeros(Parameters.birth_means.shape)
        covariances_new = np.zeros(Parameters.birth_covariances.shape)
        
        # Filtering loop - MB (empty poisson)
        
        start_time = time.time()
        for k in range(Parameters.number_of_timesteps):
            
            #####################
            #   P R E D I C T   #
            #####################
            
            prob_existence, means_existing, covariances_existing = oMajorSensor.predictor.predict_pmb_empty_poisson(prob_existence.copy(), means_existing.copy(), covariances_existing.copy())
            
            all_predictions_prob_exist_for_plotting.append(prob_existence)
            all_predictions_for_plotting.append(means_existing)
            all_prediction_covs_for_plotting.append(covariances_existing)
            
            
            #########################################
            #   S E N S O R   M A N A G E M E N T   #
            #########################################
            # If the sensors are in close proximty to eachother
            if np.linalg.norm(oSensor1.current_sensor_position - oSensor2.current_sensor_position) < Parameters.proximity_distance:
                
                # Jointly optimise the sensors
                print('jointly optimised...')
                
                if non_myopic == True:
                    Parameters.MCTS_simulation_number = MCTS3_budget_joint
                    Parameters.simulation_search_depth = MCTS3_lookahead
                    
                    planning_nonmyopically.append(True)
                    
                cost_of_each_combination, root = oMajorSensor.manage_multiple_sensors_mb_full_jointly_calculated_MCTS([oSensor1, oSensor2],  prob_existence, means_existing, covariances_existing, cost_function = 'GOSPA')
                all_the_tree_roots.append(root)
                
                information_shared.append(True)
                
        
            # otherwise, complete independent sensor management   
            else:
                print('individually optimised...')
                
                if non_myopic == True:
                    Parameters.MCTS_simulation_number = MCTS3_budget_individual
                    Parameters.simulation_search_depth = MCTS3_lookahead
                    
                    planning_nonmyopically.append(True)
              
                    
                cost_of_each_combination, root = oMajorSensor.manage_multiple_sensors_mb_individually_MCTS([oSensor1, oSensor2], prob_existence, means_existing, covariances_existing, cost_function = 'GOSPA')
                all_the_tree_roots.append(root)
                
                information_shared.append(False)
           
            costs_list.append(cost_of_each_combination)
        
            #################################################
            #   G E N E R A T E   M E A S U R E M E N T S   #
            #################################################
            # Sensor 1
            oSensor1.measurements.generate_measurements_multiple_targets(np.array(oGroundtruth.target_states)[:,:,k].T, oSensor1.current_sensor_position, k)
            
            # Sensor 2
            oSensor2.measurements.generate_measurements_multiple_targets(np.array(oGroundtruth.target_states)[:,:,k].T, oSensor2.current_sensor_position, k)
            
            # Combining sensor measurements
            arrays = [oSensor1.measurements.all_measurements[k], oSensor2.measurements.all_measurements[k]] 
            combined_measurements = merge_measurements(arrays)
            oMajorSensor.measurements.all_measurements[k] = combined_measurements # pass this information to the central filtering loop
        
            ###################
            #   U P D A T E   #
            ###################
            # If multiple sensors, each do their own update here in a sequential manner
            
            # S1
            # Update
            lambdau_temp, means_new_temp, covariances_new_temp, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = oMajorSensor.updater.update_pmb_empty_poisson(lambdau, means_new, covariances_new, prob_existence, means_existing, covariances_existing, oSensor1.measurements.all_measurements[k], [oSensor1])
            # Loopy Beleif Propagation
            pupd, pnew = lbp(wupd, wnew)
            # Track oriented multi-Bernoulli
            prob_existence, means_existing, covariances_existing = tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
            
            prob_existence = prob_existence.reshape((-1,1)) # reshaping to allow for sequential update
            
            # S2
            # Update
            lambdau, means_new, covariances_new, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = oMajorSensor.updater.update_pmb_empty_poisson(lambdau_temp, means_new_temp, covariances_new_temp, prob_existence, means_existing, covariances_existing, oSensor2.measurements.all_measurements[k], [oSensor2])
            # Loopy Beleif Propagation
            pupd, pnew = lbp(wupd, wnew)
            # Track oriented multi-Bernoulli
            prob_existence, means_existing, covariances_existing = tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
        
            #####################
            #   M E R G I N G   #
            #####################
            if len(prob_existence) > 0:
              prob_existence, means_existing, covariances_existing = merge_gaussians(prob_existence, means_existing, covariances_existing)
            
            # if len(lambdau) > 0:
            #   lambdau, means_new, covariances_new = merge_gaussians(lambdau, means_new, covariances_new)
            
            
            #####################
            #   P R U N I N G   #
            #####################
            # Existing targets
            filter_indices = prob_existence > Parameters.pruning_threshold
            prob_existence = prob_existence[filter_indices]
            means_existing = means_existing[:, filter_indices] 
            covariances_existing = covariances_existing[filter_indices,:,:]
            
            # PPP elements
            # filter_indices = lambdau > Parameters.pruning_threshold
            # lambdau = lambdau[filter_indices]
            # means_new = means_new[:, filter_indices] 
            # covariances_new = covariances_new[filter_indices,:,:]
    
            # Eliminate filter entries if they are outside of the surveillance region
            # Create the conditions for x and y coordinates
            x_condition = (means_existing[0, :] >= Parameters.surveillance_area[0, 0]) & (means_existing[0, :] <= Parameters.surveillance_area[0, 1])
            y_condition = (means_existing[2, :] >= Parameters.surveillance_area[1, 0]) & (means_existing[2, :] <= Parameters.surveillance_area[1, 1])
            
            # Combine the conditions
            combined_condition = x_condition & y_condition # central
        
            # Filter the array based on the combined condition 
            filtered_prob_existence = prob_existence[combined_condition]
            filtered_means_existing = means_existing[:, combined_condition] 
            filtered_covaiances_existing = covariances_existing[combined_condition,:,:]
            prob_existence = filtered_prob_existence.copy()
            means_existing = filtered_means_existing.copy()
            covariances_existing = filtered_covaiances_existing.copy()
                   
            # Existence Thresholding
            ss = np.array(prob_existence) > Parameters.existence_threshold
            n = np.sum(ss)
          
            prob_existence_extracted = prob_existence[ss]
            means_existing_extracted = means_existing[:,ss]
            covariances_existing_extracted = covariances_existing[ss,:,:]
            
            
            # Store for plotting
            means_existing_for_plot = means_existing_extracted.copy()
            if means_existing_extracted.shape[1] == 0:
                means_existing_for_plot = np.array([[np.nan],
                                                    [np.nan],
                                                    [np.nan],
                                                    [np.nan]])
                
            all_means_for_plotting.append(means_existing_extracted)
            all_prob_existences.append(prob_existence_extracted)
            all_covs_for_plotting.append(covariances_existing_extracted)
            
        
            #################################################
            #   C A L C U L A T E   G O S P A   E R R O R   #
            #################################################
            #Removing placeholder nans
            # Check which columns contain np.nan values
            groundtruths = np.array(oGroundtruth.target_states)[:,:,k].T
            nan_columns_gt = np.any(np.isnan(groundtruths), axis=0)
            nan_columns_filter_estimates = np.any(np.isnan(means_existing_extracted), axis=0)
            
            # Remove columns with np.nan values
            no_nan_groundtruth = groundtruths[:, ~nan_columns_gt]
            no_nan_filter_estimates = means_existing_extracted[:, ~nan_columns_filter_estimates]
            
            gospa, tracks_to_targets, gospa_loc, gospa_miss, gospa_false = calculate_gospa(no_nan_groundtruth.T, no_nan_filter_estimates.T, Parameters.c, Parameters.p)
            
            # gospa has already been **(1/p) in the function defenition but the localisation, missed and false have not
            gospa = gospa**2
            
            all_gospa.append(gospa)
            all_loc.append(gospa_loc)
            all_miss.append(gospa_miss)
            all_fal.append(gospa_false)
            
            #################################################
            print("errors calculated ", k , "MCTS 3 \n")       
    
        all_MCTS3_gospa_error.append(all_gospa)
        all_MCTS3_loc_gospa_error.append(all_loc)
        all_MCTS3_miss_gospa_error.append(all_miss)
        all_MCTS3_fal_gospa_error.append(all_fal)
        
    average_MCTS3_error, localisation_MCTS3_error, missed_MCTS3_error, false_MCTS3_error = calculate_average_algorithm_GOSPA_error(all_MCTS3_gospa_error, all_MCTS3_loc_gospa_error, all_MCTS3_miss_gospa_error, all_MCTS3_fal_gospa_error)
    fig_MCTS3 = GOSPA_breakdown_plots(all_MCTS3_gospa_error, all_MCTS3_loc_gospa_error, all_MCTS3_miss_gospa_error, all_MCTS3_fal_gospa_error, Parameters.number_of_timesteps, Parameters.c, fig=fig_MCTS2, sensor_alg = "GD - MCTS3")

    # Save raw MC errors + per-timestep mean/std to ./results
    MCTS3_stats = save_error_results("MCTS3", all_MCTS3_gospa_error, all_MCTS3_loc_gospa_error, all_MCTS3_miss_gospa_error, all_MCTS3_fal_gospa_error)
    print('MCTS3: mean-over-time std(GOSPA) =', MCTS3_stats['overall_std_gospa_mean_over_time'])

    
    end_time_MCTS3 = time.time()
    elapsed_time_MCTS3 = end_time_MCTS3 - start_time_MCTS3
    
    # Creating tuple of birth density information
    birth_densities = (Parameters.birth_means.copy(), Parameters.birth_covariances.copy())
    
    # Reformatting target track information for plots
    target_tracks = oGroundtruth.target_states
    target_tracks_matrix = np.array(target_tracks).reshape(-1, Parameters.number_of_timesteps)[::4]
    
    
    #######################################
    #   2 D   A N I M A T E D   P L O T   #
    #######################################
    # Creating figure for animated 2D plotting
    fig, ax = plt.subplots()
    plt.tight_layout()
    
    ani = FuncAnimation(fig, update_anim_plot, frames=range(Parameters.number_of_timesteps), repeat=True, interval=100, fargs=(ax, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, [all_means_for_plotting], sensor_list, information_shared, planning_nonmyopically))
    
    ani.save('animated_plot_MCTS3.gif', writer='pillow', fps=10) # saves to root directory
    
    # Snapshot of scenario
    save_frame_as_image(timestep_for_saving, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, [all_means_for_plotting], sensor_list, information_shared, planning_nonmyopically, 'MCTS3_scenario_snapshot')


    #######################################
    #   3 D   A N I M A T E D   P L O T   #
    #######################################
    # Creating figure for 3D plotting
    # fig3d = plt.figure()
    # ax = fig3d.add_subplot(111, projection='3d')
    # ani3d = FuncAnimation(fig3d, update_anim_plot_3d, frames=range(Parameters.number_of_timesteps), repeat=True, interval=100, fargs=(ax, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, all_means_for_plotting, sensor_list))
    # ani3d.save('animated_plot_3d.gif', writer='pillow', fps=10) # saves to root directory
    
    
    #########################################
    #   G O S P A   E R R O R   P L O T S   #
    #########################################
    # fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1)
    # ax1.plot(range(len(all_gospa)), all_gospa)
    # ax2.plot(range(len(all_loc)), all_loc)
    # ax3.plot(range(len(all_miss)), all_miss)
    # ax4.plot(range(len(all_fal)), all_fal)
    
    # ax1.set_title('GOSPA')
    # ax2.set_title('Localisation error')
    # ax3.set_title('Missed error')
    # ax4.set_title('False error')
    # ax1.set_ylim(-5, max(all_gospa)+20)
    # ax2.set_ylim(-5, max(all_loc) + 20)
    # ax3.set_ylim(-5, max(all_miss) + 20)
    # ax4.set_ylim(-5, max(all_fal) + 20)
    # plt.tight_layout()



# %% MCTS - Set 4
#################################
#   M C T S   -   S E T   4   #
#################################
if MCTS4 == True:
    np.random.seed(selected_seed) 
    
    all_MCTS4_gospa_error = []
    all_MCTS4_loc_gospa_error = []
    all_MCTS4_miss_gospa_error = []
    all_MCTS4_fal_gospa_error = []
    
    start_time_MCTS4 = time.time()
    myopic = False
    non_myopic = True
    
    # Filter parameters
    all_means_for_plotting = []
    all_covs_for_plotting = []
    all_prob_existences = []
    all_predictions_for_plotting = []
    all_predictions_prob_exist_for_plotting = []
    all_prediction_covs_for_plotting = []
    
    
    # Plotting lists
    information_shared = []
    planning_nonmyopically = []
    costs_list = []
    all_the_tree_roots = []
    
    # Errors
    all_gospa = []
    all_loc = [] # localisation
    all_miss = [] # missed
    all_fal = [] # false
    
    for mc_run in range(Parameters.number_of_mc_runs):
        
        # Filter parameters
        all_means_for_plotting = []
        all_covs_for_plotting = []
        all_prob_existences = []
        all_predictions_for_plotting = []
        all_predictions_prob_exist_for_plotting = []
        all_prediction_covs_for_plotting = []
    
    
        # Plotting lists
        information_shared = []
        planning_nonmyopically = []
        costs_list = []
        all_the_tree_roots = []
    
        # Errors
        all_gospa = []
        all_loc = [] # localisation
        all_miss = [] # missed
        all_fal = [] # false
        
        oSensor1 =  GOSPA_Sensor_updated_maths() 
        oSensor2 = GOSPA_Sensor_updated_maths()
        oMajorSensor = GOSPA_Sensor_updated_maths() # centralised high level decision making module (does not have a physical presence in the simulation)
        sensor_list = [oSensor1, oSensor2]
        # Initialising birth locations of sensors (randomised x, fixed y)
        sensor_birth_positions = sample_sensor_birth_positions(
            obstacles_list=Parameters.obstacles_list,
            y_values=Parameters.sensor_birth_y_values,
            margin=Parameters.sensor_birth_x_margin,
            rng=np.random,
        )
        oSensor1.current_sensor_position = sensor_birth_positions[0]  # [x,y]
        oSensor2.current_sensor_position = sensor_birth_positions[1]  # [x,y]
        
        # Setting the starting position of each sensor internally
        oSensor1.movement_model.state = oSensor1.current_sensor_position.copy()
        oSensor2.movement_model.state = oSensor2.current_sensor_position.copy()
        
        ######################################################
        # Initialising the filter parameters
        ######################################################
        # Existing targets
        state_dimensions = Parameters.birth_means.shape[0]
        estimated_number_of_targets = Parameters.prob_birth[0] * Parameters.number_of_birth_densities
        prob_existence = [0] 
        means_existing = np.zeros(Parameters.birth_means[:,[0]].shape)
        covariances_existing = np.zeros(Parameters.birth_covariances[[0],:,:].shape)
        
        # PPP components (empty in MB case)
        lambdau = np.zeros(Parameters.lambdau.shape)
        means_new = np.zeros(Parameters.birth_means.shape)
        covariances_new = np.zeros(Parameters.birth_covariances.shape)
        
        # Filtering loop - MB (empty poisson)
        
        start_time = time.time()
        for k in range(Parameters.number_of_timesteps):
            
            #####################
            #   P R E D I C T   #
            #####################
            
            prob_existence, means_existing, covariances_existing = oMajorSensor.predictor.predict_pmb_empty_poisson(prob_existence.copy(), means_existing.copy(), covariances_existing.copy())
            
            all_predictions_prob_exist_for_plotting.append(prob_existence)
            all_predictions_for_plotting.append(means_existing)
            all_prediction_covs_for_plotting.append(covariances_existing)
            
            
            #########################################
            #   S E N S O R   M A N A G E M E N T   #
            #########################################
            # If the sensors are in close proximty to eachother
            if np.linalg.norm(oSensor1.current_sensor_position - oSensor2.current_sensor_position) < Parameters.proximity_distance:
                
                # Jointly optimise the sensors
                print('jointly optimised...')
                
                if non_myopic == True:
                    Parameters.MCTS_simulation_number = MCTS4_budget_joint
                    Parameters.simulation_search_depth = MCTS4_lookahead
                    
                    planning_nonmyopically.append(True)
                    
                cost_of_each_combination, root = oMajorSensor.manage_multiple_sensors_mb_full_jointly_calculated_MCTS([oSensor1, oSensor2],  prob_existence, means_existing, covariances_existing, cost_function = 'GOSPA')
                all_the_tree_roots.append(root)
                
                information_shared.append(True)
                
        
            # otherwise, complete independent sensor management   
            else:
                print('individually optimised...')
                
                if non_myopic == True:
                    Parameters.MCTS_simulation_number = MCTS4_budget_individual
                    Parameters.simulation_search_depth = MCTS4_lookahead
                    
                    planning_nonmyopically.append(True)
              
                    
                cost_of_each_combination, root = oMajorSensor.manage_multiple_sensors_mb_individually_MCTS([oSensor1, oSensor2], prob_existence, means_existing, covariances_existing, cost_function = 'GOSPA')
                all_the_tree_roots.append(root)
                
                information_shared.append(False)
           
            costs_list.append(cost_of_each_combination)
        
            #################################################
            #   G E N E R A T E   M E A S U R E M E N T S   #
            #################################################
            # Sensor 1
            oSensor1.measurements.generate_measurements_multiple_targets(np.array(oGroundtruth.target_states)[:,:,k].T, oSensor1.current_sensor_position, k)
            
            # Sensor 2
            oSensor2.measurements.generate_measurements_multiple_targets(np.array(oGroundtruth.target_states)[:,:,k].T, oSensor2.current_sensor_position, k)
            
            # Combining sensor measurements
            arrays = [oSensor1.measurements.all_measurements[k], oSensor2.measurements.all_measurements[k]] 
            combined_measurements = merge_measurements(arrays)
            oMajorSensor.measurements.all_measurements[k] = combined_measurements # pass this information to the central filtering loop
        
            ###################
            #   U P D A T E   #
            ###################
            # If multiple sensors, each do their own update here in a sequential manner
            
            # S1
            # Update
            lambdau_temp, means_new_temp, covariances_new_temp, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = oMajorSensor.updater.update_pmb_empty_poisson(lambdau, means_new, covariances_new, prob_existence, means_existing, covariances_existing, oSensor1.measurements.all_measurements[k], [oSensor1])
            # Loopy Beleif Propagation
            pupd, pnew = lbp(wupd, wnew)
            # Track oriented multi-Bernoulli
            prob_existence, means_existing, covariances_existing = tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
            
            prob_existence = prob_existence.reshape((-1,1)) # reshaping to allow for sequential update
            
            # S2
            # Update
            lambdau, means_new, covariances_new, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = oMajorSensor.updater.update_pmb_empty_poisson(lambdau_temp, means_new_temp, covariances_new_temp, prob_existence, means_existing, covariances_existing, oSensor2.measurements.all_measurements[k], [oSensor2])
            # Loopy Beleif Propagation
            pupd, pnew = lbp(wupd, wnew)
            # Track oriented multi-Bernoulli
            prob_existence, means_existing, covariances_existing = tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
        
            #####################
            #   M E R G I N G   #
            #####################
            if len(prob_existence) > 0:
              prob_existence, means_existing, covariances_existing = merge_gaussians(prob_existence, means_existing, covariances_existing)
            
            # if len(lambdau) > 0:
            #   lambdau, means_new, covariances_new = merge_gaussians(lambdau, means_new, covariances_new)
            
            
            #####################
            #   P R U N I N G   #
            #####################
            # Existing targets
            filter_indices = prob_existence > Parameters.pruning_threshold
            prob_existence = prob_existence[filter_indices]
            means_existing = means_existing[:, filter_indices] 
            covariances_existing = covariances_existing[filter_indices,:,:]
            
            # PPP elements
            # filter_indices = lambdau > Parameters.pruning_threshold
            # lambdau = lambdau[filter_indices]
            # means_new = means_new[:, filter_indices] 
            # covariances_new = covariances_new[filter_indices,:,:]
    
            # Eliminate filter entries if they are outside of the surveillance region
            # Create the conditions for x and y coordinates
            x_condition = (means_existing[0, :] >= Parameters.surveillance_area[0, 0]) & (means_existing[0, :] <= Parameters.surveillance_area[0, 1])
            y_condition = (means_existing[2, :] >= Parameters.surveillance_area[1, 0]) & (means_existing[2, :] <= Parameters.surveillance_area[1, 1])
            
            # Combine the conditions
            combined_condition = x_condition & y_condition # central
        
            # Filter the array based on the combined condition 
            filtered_prob_existence = prob_existence[combined_condition]
            filtered_means_existing = means_existing[:, combined_condition] 
            filtered_covaiances_existing = covariances_existing[combined_condition,:,:]
            prob_existence = filtered_prob_existence.copy()
            means_existing = filtered_means_existing.copy()
            covariances_existing = filtered_covaiances_existing.copy()
                   
            # Existence Thresholding
            ss = np.array(prob_existence) > Parameters.existence_threshold
            n = np.sum(ss)
          
            prob_existence_extracted = prob_existence[ss]
            means_existing_extracted = means_existing[:,ss]
            covariances_existing_extracted = covariances_existing[ss,:,:]
            
            
            # Store for plotting
            means_existing_for_plot = means_existing_extracted.copy()
            if means_existing_extracted.shape[1] == 0:
                means_existing_for_plot = np.array([[np.nan],
                                                    [np.nan],
                                                    [np.nan],
                                                    [np.nan]])
                
            all_means_for_plotting.append(means_existing_extracted)
            all_prob_existences.append(prob_existence_extracted)
            all_covs_for_plotting.append(covariances_existing_extracted)
            
        
            #################################################
            #   C A L C U L A T E   G O S P A   E R R O R   #
            #################################################
            #Removing placeholder nans
            # Check which columns contain np.nan values
            groundtruths = np.array(oGroundtruth.target_states)[:,:,k].T
            nan_columns_gt = np.any(np.isnan(groundtruths), axis=0)
            nan_columns_filter_estimates = np.any(np.isnan(means_existing_extracted), axis=0)
            
            # Remove columns with np.nan values
            no_nan_groundtruth = groundtruths[:, ~nan_columns_gt]
            no_nan_filter_estimates = means_existing_extracted[:, ~nan_columns_filter_estimates]
            
            gospa, tracks_to_targets, gospa_loc, gospa_miss, gospa_false = calculate_gospa(no_nan_groundtruth.T, no_nan_filter_estimates.T, Parameters.c, Parameters.p)
            
            # gospa has already been **(1/p) in the function defenition but the localisation, missed and false have not
            gospa = gospa**2
            
            all_gospa.append(gospa)
            all_loc.append(gospa_loc)
            all_miss.append(gospa_miss)
            all_fal.append(gospa_false)
            
            #################################################
            print("errors calculated ", k , " MCTS 4 \n")       
    
        all_MCTS4_gospa_error.append(all_gospa)
        all_MCTS4_loc_gospa_error.append(all_loc)
        all_MCTS4_miss_gospa_error.append(all_miss)
        all_MCTS4_fal_gospa_error.append(all_fal)
        
    average_MCTS4_error, localisation_MCTS4_error, missed_MCTS4_error, false_MCTS4_error = calculate_average_algorithm_GOSPA_error(all_MCTS4_gospa_error, all_MCTS4_loc_gospa_error, all_MCTS4_miss_gospa_error, all_MCTS4_fal_gospa_error)
    fig_MCTS4 = GOSPA_breakdown_plots(all_MCTS4_gospa_error, all_MCTS4_loc_gospa_error, all_MCTS4_miss_gospa_error, all_MCTS4_fal_gospa_error, Parameters.number_of_timesteps, Parameters.c, fig=fig_MCTS3, sensor_alg = "GD - MCTS4")

    # Save raw MC errors + per-timestep mean/std to ./results
    MCTS4_stats = save_error_results("MCTS4", all_MCTS4_gospa_error, all_MCTS4_loc_gospa_error, all_MCTS4_miss_gospa_error, all_MCTS4_fal_gospa_error)
    print('MCTS4: mean-over-time std(GOSPA) =', MCTS4_stats['overall_std_gospa_mean_over_time'])

    
    end_time_MCTS4 = time.time()
    elapsed_time_MCTS4 = end_time_MCTS4 - start_time_MCTS4
    
    # Creating tuple of birth density information
    birth_densities = (Parameters.birth_means.copy(), Parameters.birth_covariances.copy())
    
    # Reformatting target track information for plots
    target_tracks = oGroundtruth.target_states
    target_tracks_matrix = np.array(target_tracks).reshape(-1, Parameters.number_of_timesteps)[::4]
    
    
    #######################################
    #   2 D   A N I M A T E D   P L O T   #
    #######################################
    # Creating figure for animated 2D plotting
    fig, ax = plt.subplots()
    plt.tight_layout()
    
    ani = FuncAnimation(fig, update_anim_plot, frames=range(Parameters.number_of_timesteps), repeat=True, interval=100, fargs=(ax, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, [all_means_for_plotting], sensor_list, information_shared, planning_nonmyopically))
    
    ani.save('animated_plot_MCTS4.gif', writer='pillow', fps=10) # saves to root directory
    
    # Snapshot of scenario
    save_frame_as_image(timestep_for_saving, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, [all_means_for_plotting], sensor_list, information_shared, planning_nonmyopically, 'MCTS4_scenario_snapshot')


    #######################################
    #   3 D   A N I M A T E D   P L O T   #
    #######################################
    # Creating figure for 3D plotting
    # fig3d = plt.figure()
    # ax = fig3d.add_subplot(111, projection='3d')
    # ani3d = FuncAnimation(fig3d, update_anim_plot_3d, frames=range(Parameters.number_of_timesteps), repeat=True, interval=100, fargs=(ax, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, all_means_for_plotting, sensor_list))
    # ani3d.save('animated_plot_3d.gif', writer='pillow', fps=10) # saves to root directory
    
    
    #########################################
    #   G O S P A   E R R O R   P L O T S   #
    #########################################
    # fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1)
    # ax1.plot(range(len(all_gospa)), all_gospa)
    # ax2.plot(range(len(all_loc)), all_loc)
    # ax3.plot(range(len(all_miss)), all_miss)
    # ax4.plot(range(len(all_fal)), all_fal)
    
    # ax1.set_title('GOSPA')
    # ax2.set_title('Localisation error')
    # ax3.set_title('Missed error')
    # ax4.set_title('False error')
    # ax1.set_ylim(-5, max(all_gospa)+20)
    # ax2.set_ylim(-5, max(all_loc) + 20)
    # ax3.set_ylim(-5, max(all_miss) + 20)
    # ax4.set_ylim(-5, max(all_fal) + 20)
    # plt.tight_layout()
    
# %% MCTS - Set 5
    #################################
    #   M C T S   -   S E T   5   #
    #################################
if MCTS5 == True:
    np.random.seed(selected_seed) 
    
    all_MCTS5_gospa_error = []
    all_MCTS5_loc_gospa_error = []
    all_MCTS5_miss_gospa_error = []
    all_MCTS5_fal_gospa_error = []
    
    start_time_MCTS5 = time.time()
    myopic = False
    non_myopic = True
    
    # Filter parameters
    all_means_for_plotting = []
    all_covs_for_plotting = []
    all_prob_existences = []
    all_predictions_for_plotting = []
    all_predictions_prob_exist_for_plotting = []
    all_prediction_covs_for_plotting = []
    
    
    # Plotting lists
    information_shared = []
    planning_nonmyopically = []
    costs_list = []
    all_the_tree_roots = []
    
    # Errors
    all_gospa = []
    all_loc = [] # localisation
    all_miss = [] # missed
    all_fal = [] # false
    
    for mc_run in range(Parameters.number_of_mc_runs):
        
        # Filter parameters
        all_means_for_plotting = []
        all_covs_for_plotting = []
        all_prob_existences = []
        all_predictions_for_plotting = []
        all_predictions_prob_exist_for_plotting = []
        all_prediction_covs_for_plotting = []
    
    
        # Plotting lists
        information_shared = []
        planning_nonmyopically = []
        costs_list = []
        all_the_tree_roots = []
    
        # Errors
        all_gospa = []
        all_loc = [] # localisation
        all_miss = [] # missed
        all_fal = [] # false
        
        oSensor1 =  GOSPA_Sensor_updated_maths() 
        oSensor2 = GOSPA_Sensor_updated_maths()
        oMajorSensor = GOSPA_Sensor_updated_maths() # centralised high level decision making module (does not have a physical presence in the simulation)
        sensor_list = [oSensor1, oSensor2]
        # Initialising birth locations of sensors (randomised x, fixed y)
        sensor_birth_positions = sample_sensor_birth_positions(
            obstacles_list=Parameters.obstacles_list,
            y_values=Parameters.sensor_birth_y_values,
            margin=Parameters.sensor_birth_x_margin,
            rng=np.random,
        )
        oSensor1.current_sensor_position = sensor_birth_positions[0]  # [x,y]
        oSensor2.current_sensor_position = sensor_birth_positions[1]  # [x,y]
        
        # Setting the starting position of each sensor internally
        oSensor1.movement_model.state = oSensor1.current_sensor_position.copy()
        oSensor2.movement_model.state = oSensor2.current_sensor_position.copy()
        
        ######################################################
        # Initialising the filter parameters
        ######################################################
        # Existing targets
        state_dimensions = Parameters.birth_means.shape[0]
        estimated_number_of_targets = Parameters.prob_birth[0] * Parameters.number_of_birth_densities
        prob_existence = [0] 
        means_existing = np.zeros(Parameters.birth_means[:,[0]].shape)
        covariances_existing = np.zeros(Parameters.birth_covariances[[0],:,:].shape)
        
        # PPP components (empty in MB case)
        lambdau = np.zeros(Parameters.lambdau.shape)
        means_new = np.zeros(Parameters.birth_means.shape)
        covariances_new = np.zeros(Parameters.birth_covariances.shape)
        
        # Filtering loop - MB (empty poisson)
        
        start_time = time.time()
        for k in range(Parameters.number_of_timesteps):
            
            #####################
            #   P R E D I C T   #
            #####################
            
            prob_existence, means_existing, covariances_existing = oMajorSensor.predictor.predict_pmb_empty_poisson(prob_existence.copy(), means_existing.copy(), covariances_existing.copy())
            
            all_predictions_prob_exist_for_plotting.append(prob_existence)
            all_predictions_for_plotting.append(means_existing)
            all_prediction_covs_for_plotting.append(covariances_existing)
            
            
            #########################################
            #   S E N S O R   M A N A G E M E N T   #
            #########################################
            # If the sensors are in close proximty to eachother
            if np.linalg.norm(oSensor1.current_sensor_position - oSensor2.current_sensor_position) < Parameters.proximity_distance:
                
                # Jointly optimise the sensors
                print('jointly optimised...')
                
                if non_myopic == True:
                    Parameters.MCTS_simulation_number = MCTS5_budget_joint
                    Parameters.simulation_search_depth = MCTS5_lookahead
                    
                    planning_nonmyopically.append(True)
                    
                cost_of_each_combination, root = oMajorSensor.manage_multiple_sensors_mb_full_jointly_calculated_MCTS([oSensor1, oSensor2],  prob_existence, means_existing, covariances_existing, cost_function = 'GOSPA')
                all_the_tree_roots.append(root)
                
                information_shared.append(True)
                
        
            # otherwise, complete independent sensor management   
            else:
                print('individually optimised...')
                
                if non_myopic == True:
                    Parameters.MCTS_simulation_number = MCTS5_budget_individual
                    Parameters.simulation_search_depth = MCTS5_lookahead
                    
                    planning_nonmyopically.append(True)
              
                    
                cost_of_each_combination, root = oMajorSensor.manage_multiple_sensors_mb_individually_MCTS([oSensor1, oSensor2], prob_existence, means_existing, covariances_existing, cost_function = 'GOSPA')
                all_the_tree_roots.append(root)
                
                information_shared.append(False)
           
            costs_list.append(cost_of_each_combination)
        
            #################################################
            #   G E N E R A T E   M E A S U R E M E N T S   #
            #################################################
            # Sensor 1
            oSensor1.measurements.generate_measurements_multiple_targets(np.array(oGroundtruth.target_states)[:,:,k].T, oSensor1.current_sensor_position, k)
            
            # Sensor 2
            oSensor2.measurements.generate_measurements_multiple_targets(np.array(oGroundtruth.target_states)[:,:,k].T, oSensor2.current_sensor_position, k)
            
            # Combining sensor measurements
            arrays = [oSensor1.measurements.all_measurements[k], oSensor2.measurements.all_measurements[k]] 
            combined_measurements = merge_measurements(arrays)
            oMajorSensor.measurements.all_measurements[k] = combined_measurements # pass this information to the central filtering loop
        
            ###################
            #   U P D A T E   #
            ###################
            # If multiple sensors, each do their own update here in a sequential manner
            
            # S1
            # Update
            lambdau_temp, means_new_temp, covariances_new_temp, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = oMajorSensor.updater.update_pmb_empty_poisson(lambdau, means_new, covariances_new, prob_existence, means_existing, covariances_existing, oSensor1.measurements.all_measurements[k], [oSensor1])
            # Loopy Beleif Propagation
            pupd, pnew = lbp(wupd, wnew)
            # Track oriented multi-Bernoulli
            prob_existence, means_existing, covariances_existing = tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
            
            prob_existence = prob_existence.reshape((-1,1)) # reshaping to allow for sequential update
            
            # S2
            # Update
            lambdau, means_new, covariances_new, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = oMajorSensor.updater.update_pmb_empty_poisson(lambdau_temp, means_new_temp, covariances_new_temp, prob_existence, means_existing, covariances_existing, oSensor2.measurements.all_measurements[k], [oSensor2])
            # Loopy Beleif Propagation
            pupd, pnew = lbp(wupd, wnew)
            # Track oriented multi-Bernoulli
            prob_existence, means_existing, covariances_existing = tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
        
            #####################
            #   M E R G I N G   #
            #####################
            if len(prob_existence) > 0:
              prob_existence, means_existing, covariances_existing = merge_gaussians(prob_existence, means_existing, covariances_existing)
            
            # if len(lambdau) > 0:
            #   lambdau, means_new, covariances_new = merge_gaussians(lambdau, means_new, covariances_new)
            
            
            #####################
            #   P R U N I N G   #
            #####################
            # Existing targets
            filter_indices = prob_existence > Parameters.pruning_threshold
            prob_existence = prob_existence[filter_indices]
            means_existing = means_existing[:, filter_indices] 
            covariances_existing = covariances_existing[filter_indices,:,:]
            
            # PPP elements
            # filter_indices = lambdau > Parameters.pruning_threshold
            # lambdau = lambdau[filter_indices]
            # means_new = means_new[:, filter_indices] 
            # covariances_new = covariances_new[filter_indices,:,:]
    
            # Eliminate filter entries if they are outside of the surveillance region
            # Create the conditions for x and y coordinates
            x_condition = (means_existing[0, :] >= Parameters.surveillance_area[0, 0]) & (means_existing[0, :] <= Parameters.surveillance_area[0, 1])
            y_condition = (means_existing[2, :] >= Parameters.surveillance_area[1, 0]) & (means_existing[2, :] <= Parameters.surveillance_area[1, 1])
            
            # Combine the conditions
            combined_condition = x_condition & y_condition # central
        
            # Filter the array based on the combined condition 
            filtered_prob_existence = prob_existence[combined_condition]
            filtered_means_existing = means_existing[:, combined_condition] 
            filtered_covaiances_existing = covariances_existing[combined_condition,:,:]
            prob_existence = filtered_prob_existence.copy()
            means_existing = filtered_means_existing.copy()
            covariances_existing = filtered_covaiances_existing.copy()
                   
            # Existence Thresholding
            ss = np.array(prob_existence) > Parameters.existence_threshold
            n = np.sum(ss)
          
            prob_existence_extracted = prob_existence[ss]
            means_existing_extracted = means_existing[:,ss]
            covariances_existing_extracted = covariances_existing[ss,:,:]
            
            
            # Store for plotting
            means_existing_for_plot = means_existing_extracted.copy()
            if means_existing_extracted.shape[1] == 0:
                means_existing_for_plot = np.array([[np.nan],
                                                    [np.nan],
                                                    [np.nan],
                                                    [np.nan]])
                
            all_means_for_plotting.append(means_existing_extracted)
            all_prob_existences.append(prob_existence_extracted)
            all_covs_for_plotting.append(covariances_existing_extracted)
            
        
            #################################################
            #   C A L C U L A T E   G O S P A   E R R O R   #
            #################################################
            #Removing placeholder nans
            # Check which columns contain np.nan values
            groundtruths = np.array(oGroundtruth.target_states)[:,:,k].T
            nan_columns_gt = np.any(np.isnan(groundtruths), axis=0)
            nan_columns_filter_estimates = np.any(np.isnan(means_existing_extracted), axis=0)
            
            # Remove columns with np.nan values
            no_nan_groundtruth = groundtruths[:, ~nan_columns_gt]
            no_nan_filter_estimates = means_existing_extracted[:, ~nan_columns_filter_estimates]
            
            gospa, tracks_to_targets, gospa_loc, gospa_miss, gospa_false = calculate_gospa(no_nan_groundtruth.T, no_nan_filter_estimates.T, Parameters.c, Parameters.p)
            
            # gospa has already been **(1/p) in the function defenition but the localisation, missed and false have not
            gospa = gospa**2
            
            all_gospa.append(gospa)
            all_loc.append(gospa_loc)
            all_miss.append(gospa_miss)
            all_fal.append(gospa_false)
            
            #################################################
            print("errors calculated ", k , " MCTS 5 \n")       
    
        all_MCTS5_gospa_error.append(all_gospa)
        all_MCTS5_loc_gospa_error.append(all_loc)
        all_MCTS5_miss_gospa_error.append(all_miss)
        all_MCTS5_fal_gospa_error.append(all_fal)
        
    average_MCTS5_error, localisation_MCTS5_error, missed_MCTS5_error, false_MCTS5_error = calculate_average_algorithm_GOSPA_error(all_MCTS5_gospa_error, all_MCTS5_loc_gospa_error, all_MCTS5_miss_gospa_error, all_MCTS5_fal_gospa_error)
    fig_MCTS5 = GOSPA_breakdown_plots(all_MCTS5_gospa_error, all_MCTS5_loc_gospa_error, all_MCTS5_miss_gospa_error, all_MCTS5_fal_gospa_error, Parameters.number_of_timesteps, Parameters.c, fig=fig_MCTS4, sensor_alg = "GD - MCTS5")

    # Save raw MC errors + per-timestep mean/std to ./results
    MCTS5_stats = save_error_results("MCTS5", all_MCTS5_gospa_error, all_MCTS5_loc_gospa_error, all_MCTS5_miss_gospa_error, all_MCTS5_fal_gospa_error)
    print('MCTS5: mean-over-time std(GOSPA) =', MCTS5_stats['overall_std_gospa_mean_over_time'])

    
    end_time_MCTS5 = time.time()
    elapsed_time_MCTS5 = end_time_MCTS5 - start_time_MCTS5
    
    # Creating tuple of birth density information
    birth_densities = (Parameters.birth_means.copy(), Parameters.birth_covariances.copy())
    
    # Reformatting target track information for plots
    target_tracks = oGroundtruth.target_states
    target_tracks_matrix = np.array(target_tracks).reshape(-1, Parameters.number_of_timesteps)[::4]
    
    
    #######################################
    #   2 D   A N I M A T E D   P L O T   #
    #######################################
    # Creating figure for animated 2D plotting
    fig, ax = plt.subplots()
    plt.tight_layout()
    
    ani = FuncAnimation(fig, update_anim_plot, frames=range(Parameters.number_of_timesteps), repeat=True, interval=100, fargs=(ax, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, [all_means_for_plotting], sensor_list, information_shared, planning_nonmyopically))
    
    ani.save('animated_plot_MCTS5.gif', writer='pillow', fps=10) # saves to root directory
    
    
    # Snapshot of scenario
    save_frame_as_image(timestep_for_saving, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, [all_means_for_plotting], sensor_list, information_shared, planning_nonmyopically, 'MCTS5_scenario_snapshot')

    #######################################
    #   3 D   A N I M A T E D   P L O T   #
    #######################################
    # Creating figure for 3D plotting
    # fig3d = plt.figure()
    # ax = fig3d.add_subplot(111, projection='3d')
    # ani3d = FuncAnimation(fig3d, update_anim_plot_3d, frames=range(Parameters.number_of_timesteps), repeat=True, interval=100, fargs=(ax, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, all_means_for_plotting, sensor_list))
    # ani3d.save('animated_plot_3d.gif', writer='pillow', fps=10) # saves to root directory
    
    
    #########################################
    #   G O S P A   E R R O R   P L O T S   #
    #########################################
    # fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1)
    # ax1.plot(range(len(all_gospa)), all_gospa)
    # ax2.plot(range(len(all_loc)), all_loc)
    # ax3.plot(range(len(all_miss)), all_miss)
    # ax4.plot(range(len(all_fal)), all_fal)
    
    # ax1.set_title('GOSPA')
    # ax2.set_title('Localisation error')
    # ax3.set_title('Missed error')
    # ax4.set_title('False error')
    # ax1.set_ylim(-5, max(all_gospa)+20)
    # ax2.set_ylim(-5, max(all_loc) + 20)
    # ax3.set_ylim(-5, max(all_miss) + 20)
    # ax4.set_ylim(-5, max(all_fal) + 20)
    # plt.tight_layout()

# %% Saving plots and printing
if MCTS5 == True:
    fig_MCTS5.savefig('GOSPA_error_plots_png.png')
    fig_MCTS5.savefig('GOSPA_error_plots_eps.eps')
    print('Average Errors')
    print(f'Myopic: {average_myopic_error} \n Myopic KLD: {average_myopicKLD_error} \n MCTS1: {average_MCTS1_error} \n MCTS2: {average_MCTS2_error} \n MCTS3: {average_MCTS3_error} \n MCTS4: {average_MCTS4_error} \n MCTS5: {average_MCTS5_error}')

    times = np.array([elapsed_time_myopic, elapsed_time_myopicKLD, elapsed_time_MCTS1, elapsed_time_MCTS2, elapsed_time_MCTS3, elapsed_time_MCTS4, elapsed_time_MCTS5]) / (Parameters.number_of_mc_runs * Parameters.number_of_timesteps)
    print('Times Taken per time step')
    print(f'Myopic: {times[0]} \n Myopic KLD: {times[1]} \n MCTS1: {times[2]} \n MCTS2: {times[3]} \n MCTS3: {times[4]} \n MCTS4: {times[5]} \n MCTS5: {times[6]}')

elif MCTS4 == True:
    fig_MCTS4.savefig('GOSPA_error_plots_png.png')
    fig_MCTS4.savefig('GOSPA_error_plots_eps.eps')
    print('Average Errors')
    print(f'Myopic: {average_myopic_error} \n Myopic KLD: {average_myopicKLD_error} \n MCTS1: {average_MCTS1_error} \n MCTS2: {average_MCTS2_error} \n MCTS3: {average_MCTS3_error} \n MCTS4: {average_MCTS4_error} ')

    times = np.array([elapsed_time_myopic, elapsed_time_myopicKLD, elapsed_time_MCTS1, elapsed_time_MCTS2, elapsed_time_MCTS3, elapsed_time_MCTS4]) / (Parameters.number_of_mc_runs * Parameters.number_of_timesteps)
    print('Times Taken per time step')
    print(f'Myopic: {times[0]} \n Myopic KLD: {times[1]} \n MCTS1: {times[2]} \n MCTS2: {times[3]} \n MCTS3: {times[4]} \n MCTS4: {times[5]}')

    
elif MCTS3 == True:
    fig_MCTS3.savefig('GOSPA_error_plots_png.png')
    fig_MCTS3.savefig('GOSPA_error_plots_eps.eps')
    print('Average Errors')
    print(f'Myopic: {average_myopic_error} \n Myopic KLD: {average_myopicKLD_error} \n MCTS1: {average_MCTS1_error} \n MCTS2: {average_MCTS2_error} \n MCTS3: {average_MCTS3_error} ')

    times = np.array([elapsed_time_myopic, elapsed_time_myopicKLD,  elapsed_time_MCTS1, elapsed_time_MCTS2, elapsed_time_MCTS3]) / (Parameters.number_of_mc_runs * Parameters.number_of_timesteps)
    print('Times Taken per time step')
    print(f'Myopic: {times[0]} \n Myopic KLD: {times[1]} \n MCTS1: {times[2]} \n MCTS2: {times[3]} \n MCTS3: {times[4]}')

elif MCTS2 == True:
    fig_MCTS2.savefig('GOSPA_error_plots_png.png')
    fig_MCTS2.savefig('GOSPA_error_plots_eps.eps')
    print('Average Errors')
    print(f'Myopic: {average_myopic_error} \n Myopic KLD: {average_myopicKLD_error}  \n MCTS1: {average_MCTS1_error} \n MCTS2: {average_MCTS2_error} ')

    times = np.array([elapsed_time_myopic, elapsed_time_myopicKLD,  elapsed_time_MCTS1, elapsed_time_MCTS2]) / (Parameters.number_of_mc_runs * Parameters.number_of_timesteps)
    print('Times Taken per time step')
    print(f'Myopic: {times[0]} \n Myopic KLD: {times[1]} \n MCTS1: {times[2]} \n MCTS2: {times[3]}')
    
elif MCTS1 == True:
    fig_MCTS1.savefig('GOSPA_error_plots_png.png')
    fig_MCTS1.savefig('GOSPA_error_plots_eps.eps')
    print('Average Errors')
    print(f'Myopic: {average_myopic_error} \n Myopic KLD: {average_myopicKLD_error} \n MCTS1: {average_MCTS1_error} ')

    times = np.array([elapsed_time_myopic, elapsed_time_myopicKLD, elapsed_time_MCTS1]) / (Parameters.number_of_mc_runs * Parameters.number_of_timesteps)
    print('Times Taken per time step')
    print(f'Myopic: {times[0]} \n Myopic KLD: {times[1]} \n MCTS1: {times[2]}')
    
else:
    fig_myopic.savefig('GOSPA_error_plots_png.png')
    fig_myopic.savefig('GOSPA_error_plots_eps.eps')
    print('Average Errors')
    print(f'Myopic: {average_myopic_error}')
    
    times = np.array([elapsed_time_myopic]) / (Parameters.number_of_mc_runs * Parameters.number_of_timesteps)
    print('Times Taken per time step')
    print(f'Myopic: {times[0]}')
    
    fig_myopicKLD.savefig('GOSPA_error_plots_png.png')
    fig_myopicKLD.savefig('GOSPA_error_plots_eps.eps')
    print('Average Errors')
    print(f'Myopic KLD : {average_myopicKLD_error}')
    
    times = np.array([elapsed_time_myopicKLD]) / (Parameters.number_of_mc_runs * Parameters.number_of_timesteps)
    print('Times Taken per time step')
    print(f'Myopic KLD : {times[0]}')
    
    
    
    
plt.show()

# Save snapshot of algorithms, need to put into each algorithm to save the snapshot when variables exist
# save_frame_as_image(80, target_tracks, birth_densities, oMajorSensor.measurements.all_measurements, [all_means_for_plotting], sensor_list, information_shared, planning_nonmyopically, 'snapshot_frame_80')
