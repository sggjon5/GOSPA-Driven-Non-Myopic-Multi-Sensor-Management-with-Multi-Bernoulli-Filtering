# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:51:30 2023

@author: sggjone5
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Rectangle
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches


import Parameters





from scipy.spatial.distance import mahalanobis


# Function to save a specific frame as an image
def save_frame_as_image(frame, target_tracks, birth_densities, measurements, all_means_for_plotting, sensor_list, information_shared, planning_nonmyopically, filename_base):
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Call the update function for the specific frame
    update_anim_plot(frame, ax, target_tracks, birth_densities, measurements, all_means_for_plotting, sensor_list, information_shared, planning_nonmyopically)
    
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)

    # Save the figure as .png and .eps files
    plt.savefig(f'{filename_base}.png', format='png', bbox_inches='tight', dpi=1200)
    plt.savefig(f'{filename_base}.eps', format='eps', bbox_inches='tight', dpi=1200)
    
    # Close the plot to avoid display in a notebook or other environments
    plt.close()




def GOSPA_breakdown_plots(averageError, localisationError, missedError, falseError, number_of_timesteps, c, fig=None, sensor_alg = "Default"):
    """ 
    Takes in the lists of all the error contributions and plots the mean square error
    across each MC run
    
    Inputs:
        averageError: LIST of all GOSPA errors from each MC run
        localisationError: LIST of all localisation errors from each MC run
        missedError: LIST of all missed errors from each MC run
        falseError: LIST of all false errors from each MC run
        number_of_timesteps: RONSEAL
        fig: the figure you wish to plot them onto
        sensor_alg: the sensor algorithm which the errors have come from for legend purposes
    
    Outputs:
        fig: returns the figure in whcih it has plotted the graph so it can be
             passed to be plotted on again
    """
    timestep = range(number_of_timesteps)
    
    averageError = np.sqrt(np.mean(np.asarray(averageError), axis = 0)) 
    
    localisationError = np.sqrt(np.mean(np.asarray(localisationError), axis = 0)) 
    
    missedError = np.sqrt(np.mean(np.asarray(missedError), axis = 0)) 
    
    falseError = np.sqrt(np.mean(np.asarray(falseError), axis = 0)) 


    
    # setting up the axis
    if fig is None:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(10,16), nrows = 4, ncols = 1)
    
    ax1, ax2, ax3, ax4 = fig.axes
    
    # setting axis labels for each plot
    ax1.set_ylabel('GOSPA error', fontsize=16)
    
    ax2.set_ylabel('Localisation error', fontsize=16)
    
    ax3.set_ylabel('Missed detection error', fontsize=16)
    
    ax4.set_ylabel('False detection error', fontsize=16)
    
    ax4.set_xlabel('Time step', fontsize=16)

    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.tick_params(axis='both', which='minor', labelsize=16)

    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.tick_params(axis='both', which='minor', labelsize=16)

    ax3.tick_params(axis='both', which='major', labelsize=16)
    ax3.tick_params(axis='both', which='minor', labelsize=16)

    ax4.tick_params(axis='both', which='major', labelsize=16)
    ax4.tick_params(axis='both', which='minor', labelsize=16)
    
    largest_error = max(max(averageError), max(localisationError), max(missedError), max(falseError))
    # plotting data
    ax1.plot(timestep, averageError, label = sensor_alg)

    ax2.plot(timestep, localisationError, label = sensor_alg)

    ax3.plot(timestep, missedError, label = sensor_alg)
    
    ax4.plot(timestep, falseError, label = sensor_alg)

    
    ax1.legend(loc = 1)
    # ax2.legend(loc = 1)
    # ax3.legend(loc = 1)
    # ax4.legend(loc = 1)
    
    ax1.grid(visible=True, which = 'major', axis = 'both')
    ax2.grid(visible=True, which = 'major', axis = 'both')
    ax3.grid(visible=True, which = 'major', axis = 'both')
    ax4.grid(visible=True, which = 'major', axis = 'both')
    
    ax1.set(xlim=(0, number_of_timesteps), ylim=(0, 1.3*largest_error))
    ax2.set(xlim=(0, number_of_timesteps), ylim=(0, 1.3*largest_error))
    ax3.set(xlim=(0, number_of_timesteps), ylim=(0, 1.3*largest_error))
    ax4.set(xlim=(0, number_of_timesteps), ylim=(0, 1.3*largest_error))
    
    plt.tight_layout()
    
    
    return fig

def calculate_average_algorithm_GOSPA_error(averageError, localisationError, missedError, falseError):
    
    averageError = np.mean(np.sqrt(np.mean(np.asarray(averageError), axis = 0))) 
    
    localisationError = np.mean(np.sqrt(np.mean(np.asarray(localisationError), axis = 0))) 
    
    missedError = np.mean(np.sqrt(np.mean(np.asarray(missedError), axis = 0))) 
    
    falseError = np.mean(np.sqrt(np.mean(np.asarray(falseError), axis = 0))) 


    return averageError, localisationError, missedError, falseError

# Recursive function to traverse and plot the tree
def plot_tree(node, ax):
    if node.sensor_mean is None or len(node.sensor_mean) < 2:
        return  # Skip nodes that don't have valid sensor means or less than two dimensions

    # Convert sensor_means to a tuple of two float values (x, y)
    sensor_mean_2d = tuple(map(float, node.sensor_mean[:2]))

    # Plot the current node as a transparent circle at its sensor_means location
    circle = patches.Circle(sensor_mean_2d, radius=0.2, color='blue', alpha=0.4)
    ax.add_patch(circle)

    # Print debug info to verify traversal
    print(f"Plotting node at {sensor_mean_2d} with {len(node.children)} children")

    # Recursively plot all children
    for child in node.children:
        if child.sensor_mean is not None:
            child_sensor_mean_2d = tuple(map(float, child.sensor_mean[:2]))  # Ensure the child node's sensor_means is 2D
            
            # Draw a line from the parent (current node) to the child node
            ax.plot([sensor_mean_2d[0], child_sensor_mean_2d[0]],
                    [sensor_mean_2d[1], child_sensor_mean_2d[1]], 'k-', lw=1)

            # Call the function recursively for the child
            plot_tree(child, ax)  # This is where we descend into the child nodes

# Function to initialize the plot and call the recursive plot function
def visualize_mcts_tree(root_node):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set limits for better visibility (adjust these if necessary)
    ax.set_xlim(-220, 220)
    ax.set_ylim(-220, 220)
    
    ax.set_aspect('equal', 'box')
    ax.set_title('Monte Carlo Tree Search Visualization')
    
    # Call the recursive function to start plotting
    plot_tree(root_node, ax)
    
    # Show the plot
    plt.show()

# Example usage (assuming you have a root_node defined already)
# visualize_mcts_tree(root_node)


# Function to remove items from list_1 that are in list_2, comparing arrays properly
def remove_items_from_list(list_1, list_2):
    return [item for item in list_1 if not any(np.array_equal(item, other) for other in list_2)]



def fixed_distance_actions_transition(current_state, speed, t, theta):
        
        # returns the next state (x and y)(2D) in the sequence based on the current (2D state)
        # and also the fixed distance actions dynamics
        
        radius = speed * t
    
        x = current_state[0][0] + (radius * np.cos(theta))
        y = current_state[1][0] + (radius * np.sin(theta))
        next_state = np.array([[x],
                               [y]])
        
        
        return next_state



def unique_arrays(list1, list2):
    # Convert 2D arrays to tuples
    def array_to_tuple(arr):
        return tuple(map(tuple, arr))

    # Create sets of tuples
    set1 = {array_to_tuple(arr) for arr in list1}
    set2 = {array_to_tuple(arr) for arr in list2}

    # Union of both sets to get unique tuples
    unique_tuples = set1.union(set2)

    # Convert tuples back to 2D numpy arrays
    unique_arrays = [np.array(tpl) for tpl in unique_tuples]
    
    return unique_arrays




def merge_measurements(measurements_array_list):
    # Initialize merged_array as an empty array
    merged_array = np.empty((2, 0))
    
    # Iterate over each array in the list
    for array in measurements_array_list:
        if array.shape[1] == 0:
            # Skip empty arrays
            continue
        else:
            # Concatenate the current array with the merged_array along axis 1
            merged_array = np.concatenate((merged_array, array), axis=1)

    combined_measurements = merged_array

    return combined_measurements




def merge_similar_Gaussians(weights, means, covariances, threshold=0.5, n=2):
    """
    Merge Gaussian components with means within close proximity using Mahalanobis distance.

    Args:
    weights (np.ndarray): Weights (probabilities) of the Gaussian components, shape (n_components,).
    means (np.ndarray): Means of the Gaussian components, shape (4, n_components).
    covariances (np.ndarray): Covariances of the Gaussian components, shape (n_components, 4, 4).
    threshold (float): Proximity threshold for merging means.

    Returns:
    merged_weights (np.ndarray): Merged weights, shape (n_merged_components,).
    merged_means (np.ndarray): Merged means, shape (4, n_merged_components).
    merged_covariances (np.ndarray): Merged covariances, shape (n_merged_components, 4, 4).
    """
    def is_close(mean1, mean2, cov, threshold):
        diff = mean1 - mean2
        return mahalanobis(mean1, mean2, np.linalg.inv(cov)) < threshold

    merged_indices = []
    merged_weights = []
    merged_means = []
    merged_covariances = []

    used = np.zeros(weights.shape[0], dtype=bool)

    for i in range(weights.shape[0]):
        if used[i]:
            continue

        similar_indices = [i]
        for j in range(i + 1, weights.shape[0]):
            if not used[j] and is_close(means[:, i], means[:, j], covariances[i], threshold):
                similar_indices.append(j)
                used[j] = True

        used[i] = True
        w = weights[similar_indices]
        c = covariances[similar_indices]
        mean = means[:, similar_indices]

        # Adjust weights by 1/n before merging
        # n = len(similar_indices)
        w /= n

        merged_weight = np.sum(w)

        # if merged_weight >=1:
        #     merged_weight = 1

        # Compute merged mean
        weighted_sum_of_means = np.sum(w * mean, axis=1)
        merged_mean = weighted_sum_of_means / merged_weight
        merged_mean = merged_mean.reshape(-1, 1)

        # Compute merged covariance
        weighted_sum_of_covs = []
        for idx in range(len(w)):
            diff = mean[:, [idx]] - merged_mean
            weighted_sum_of_covs.append((w[idx] / merged_weight) * (c[idx] + diff @ diff.T))

        merged_covariance = np.sum(weighted_sum_of_covs, axis=0)

        merged_weights.append(merged_weight)
        merged_means.append(merged_mean)
        merged_covariances.append(merged_covariance)

    merged_weights = np.array(merged_weights)
    merged_means = np.hstack(merged_means)
    merged_covariances = np.array(merged_covariances)

    if len(merged_weights) < len(weights):
        print(f'Components reduced from {len(weights)} to {len(merged_weights)}')
    else:
        print('no redctions made')
    return merged_weights, merged_means, merged_covariances



def merge_gaussians(weights, means, covariances, threshold=1.0):
    """
    Merge Gaussian components with means within close proximity using Mahalanobis distance.

    Args:
    weights (np.ndarray): Weights (probabilities) of the Gaussian components, shape (n_components,).
    means (np.ndarray): Means of the Gaussian components, shape (4, n_components).
    covariances (np.ndarray): Covariances of the Gaussian components, shape (n_components, 4, 4).
    threshold (float): Proximity threshold for merging means.

    Returns:
    merged_weights (np.ndarray): Merged weights, shape (n_merged_components,).
    merged_means (np.ndarray): Merged means, shape (4, n_merged_components).
    merged_covariances (np.ndarray): Merged covariances, shape (n_merged_components, 4, 4).
    """
    def is_close(mean1, mean2, cov, threshold):
        diff = mean1 - mean2
        return mahalanobis(mean1, mean2, np.linalg.inv(cov)) < threshold

    merged_indices = []
    merged_weights = []
    merged_means = []
    merged_covariances = []

    used = np.zeros(weights.shape[0], dtype=bool)

    for i in range(weights.shape[0]):
        if used[i]:
            continue

        similar_indices = [i]
        for j in range(i + 1, weights.shape[0]):
            if not used[j] and is_close(means[:, i], means[:, j], covariances[i], threshold):
                similar_indices.append(j)
                used[j] = True

        used[i] = True
        w = weights[similar_indices]
        c = covariances[similar_indices]
        mean = means[:, similar_indices]

        # Adjust weights by 1/n before merging
        # n = len(similar_indices)
        # w /= n

        merged_weight = np.sum(w)

        # if merged_weight >=1:
        #     merged_weight = 1

        # Compute merged mean
        weighted_sum_of_means = np.sum(w * mean, axis=1)
        merged_mean = weighted_sum_of_means / merged_weight
        merged_mean = merged_mean.reshape(-1, 1)

        # Compute merged covariance
        weighted_sum_of_covs = []
        for idx in range(len(w)):
            diff = mean[:, [idx]] - merged_mean
            weighted_sum_of_covs.append((w[idx] / merged_weight) * (c[idx] + diff @ diff.T))

        merged_covariance = np.sum(weighted_sum_of_covs, axis=0)

        merged_weights.append(merged_weight)
        merged_means.append(merged_mean)
        merged_covariances.append(merged_covariance)

    merged_weights = np.array(merged_weights)
    merged_means = np.hstack(merged_means)
    merged_covariances = np.array(merged_covariances)

    # if len(merged_weights) < len(weights):
    #     print(f'Components reduced from {len(weights)} to {len(merged_weights)}')
    # else:
    #     print(f'no redctions made, component count: {len(weights)}')
    return merged_weights, merged_means, merged_covariances










def check_for_obstacles_list(obstacle_object, available_actions_list):

    
    if obstacle_object.name == "blockade":
            
        temp_actions = []
        
        
        for i, available_action in enumerate(available_actions_list):
            # > is out the obstacle an < is in
            if ((available_action[0] - obstacle_object.location_centre[0])**2 + (available_action[1] - obstacle_object.location_centre[1])**2) > obstacle_object.radius**2:
                # turn_options_list.pop(i)
                # actions_copy_blockade.pop(i)
                temp_actions.append(available_action)
                
         
        
        
    ######################################
        
    if obstacle_object.name == "rectangle blockade":
        
        temp_actions = []
        
        for i, available_action in enumerate(available_actions_list):
            
            
            if obstacle_object.check_if_point_inside(available_action) == False:
        
                temp_actions.append(available_action)
                

    ######################################
        
    if obstacle_object.name == "flat_rectangle_blockade":

        temp_actions = []
        
        for i, available_action in enumerate(available_actions_list):         
            
            
            if obstacle_object.check_if_point_inside(available_action) == False:
        
                temp_actions.append(available_action)
                
    
    return temp_actions 


    

def update_anim_plot(frame, ax, target_tracks, birth_densities, measurements, all_means_for_plotting, sensor_list, information_shared, planning_nonmyopically):
    ax.clear()
    
    # outside the SA
    OutSA = Rectangle((-700,-700), width=1400, height=1400, fill=True, color='gray', alpha=0.5, hatch='/', edgecolor='gray')
    ax.add_patch(OutSA)
    
    # Surveillance Area
    SA = Rectangle((Parameters.surveillance_area[0][0], Parameters.surveillance_area[0][0]), width=Parameters.surveillance_area[0][1]*2, height=Parameters.surveillance_area[0][1]*2, fill=True,  color='white', edgecolor='black', linewidth=10)
    ax.add_patch(SA)
    
    for matrix in target_tracks:
        ax.plot(matrix[0, :frame], matrix[2, :frame], 'g--')  # Plot target tracks
        ax.plot(matrix[0, frame], matrix[2, frame], color='black', marker='o', alpha=0.3)  # Plot target location
    
    colors = ['ro', 'bo', 'go', 'co', 'mo', 'yo', 'ko']  #
    colors_estimates = ['c*', 'r*', 'g*', 'b*', 'm*', 'y*', 'k*']
       
    # plotting the obstacles (if any)
    # plot obstacles
    if len(Parameters.obstacles_list) > 0:
        for obstacle in Parameters.obstacles_list:
            
            if obstacle.name == "blockade":
                obstacle_circle = plt.Circle((obstacle.location_centre[0], obstacle.location_centre[1]), obstacle.radius, fill = True, facecolor = 'grey', alpha = 0.3)
                ax.add_patch(obstacle_circle)
                
            if obstacle.name == "rectangle blockade":
                points = [(obstacle.A[0][0],obstacle.A[1][0]),(obstacle.B[0][0],obstacle.B[1][0]),(obstacle.C[0][0],obstacle.C[1][0]),(obstacle.D[0][0],obstacle.D[1][0])]
                rect = patches.Polygon(points, linewidth=1, edgecolor='grey', facecolor='grey', alpha = 0.3)
                ax.add_patch(rect)
                
            if obstacle.name == "flat_rectangle_blockade":
                points = [(obstacle.A[0][0],obstacle.A[1][0]),(obstacle.B[0][0],obstacle.B[1][0]),(obstacle.C[0][0],obstacle.C[1][0]),(obstacle.D[0][0],obstacle.D[1][0])]
                rect = patches.Polygon(points, linewidth=1, edgecolor='grey', facecolor='grey', alpha = 0.3)
                ax.add_patch(rect)
                
    else:
        pass
        
        
    # Sensor Radius
    for i, sensor in enumerate(sensor_list):
        # Ensure the color index wraps around if there are more sensors than colors
        color_index = i % len(colors)
        
        # Plot the sensor radius circle
        sensor1_radius_circle = Circle((sensor.all_selected_sensors[frame][0], sensor.all_selected_sensors[frame][1]), Parameters.sensor_radius, fill=True, color='yellow', alpha=0.2, edgecolor='yellow')
        ax.add_patch(sensor1_radius_circle)
        
        # Plot the sensor location with a different color marker
        marker = ax.plot(sensor.all_selected_sensors[frame][0], sensor.all_selected_sensors[frame][1], colors[color_index])
           
        ax.annotate(str(i+1), (sensor.all_selected_sensors[frame][0], sensor.all_selected_sensors[frame][1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12, color='black')
        
        if planning_nonmyopically[frame]:
            for line in marker:
                line.set_path_effects([
                    path_effects.Stroke(linewidth=5, foreground='purple', alpha=0.7),
                    path_effects.Normal()
                ])
            
        
        
        
    # Check the conditions and plot the purple line if true
    if len(information_shared) > 1 and information_shared[frame]:
        x_coords = [sensor_list[0].all_selected_sensors[frame][0], sensor_list[1].all_selected_sensors[frame][0]]
        y_coords = [sensor_list[0].all_selected_sensors[frame][1], sensor_list[1].all_selected_sensors[frame][1]]
        ax.plot(x_coords, y_coords, color='purple')
    
    
    
    birth_means, birth_covariances = birth_densities
    target_tracks_matrix = np.array(target_tracks).reshape(-1, Parameters.number_of_timesteps)[::4]
    
    for i in range(birth_means.shape[1]):
        # Create an ellipse representing the Gaussian distribution
        eigenvalues, eigenvectors = np.linalg.eig(birth_covariances[i,:,:])
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = 2 * np.sqrt(5.991 * eigenvalues[0])  # 95% confidence interval
        height = 2 * np.sqrt(5.991 * eigenvalues[1])  # 95% confidence interval  
        colour = 'blue'
        ellipse = Ellipse(xy=(birth_means[:,i][0], birth_means[:,i][2]), width=width, height=height, angle=angle, fill=True, color=colour, alpha=0.5)
        ax.add_patch(ellipse)    
    
    
    # calculating how many targets are alive at any given time
    number_of_targets = np.sum(~np.isnan(target_tracks_matrix[:,frame]))
    
    ax.set_aspect('equal')
    ax.set_xlim(Parameters.surveillance_area[0][0]-20, Parameters.surveillance_area[0][1]+20)  # Set X-axis limits
    ax.set_ylim(Parameters.surveillance_area[0][0]-20, Parameters.surveillance_area[0][1]+20)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_title(f'Frame {frame}, number of targets alive {number_of_targets}')

    for i in range(birth_means.shape[1]):
        
        # Create an ellipse representing the Gaussian distribution
        eigenvalues, eigenvectors = np.linalg.eig(birth_covariances[i,:,:])
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = 2 * np.sqrt(5.991 * eigenvalues[0])  # 95% confidence interval
        height = 2 * np.sqrt(5.991 * eigenvalues[1])  # 95% confidence interval  
        colour = 'blue'
        # if i == 0:
        #     colour = 'orange'
        # elif i == 2:
        #     colour = 'orange'
        ellipse = Ellipse(xy=(birth_means[:,i][0], birth_means[:,i][2]), width=width, height=height, angle=angle, fill=True, color=colour, alpha=0.5)
        ax.add_patch(ellipse)    
        

    # measurements
    ax.scatter(measurements[frame][0], measurements[frame][1], marker='1', label = "measurements", color='red', alpha = 0.5)
    
    # calculating how many targets are alive at any given time
    for i, means in enumerate(all_means_for_plotting):
        color_index = i % len(colors_estimates)
        ax.plot(means[frame][0], means[frame][2], colors_estimates[0], label = "estimated states", markersize=4)  


    number_of_targets = np.isnan
    number_of_targets = np.sum(~np.isnan(target_tracks_matrix[:,frame]))
    
    # number_of_estimated_targets = np.sum(~np.isnan(estimated_targets_matrix[:,frame]))
    
    ax.set_aspect('equal')
    ax.set_xlim(Parameters.surveillance_area[0][0]-20, Parameters.surveillance_area[0][1]+20)  # Set X-axis limits
    ax.set_ylim(Parameters.surveillance_area[0][0]-20, Parameters.surveillance_area[0][1]+20)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_title(f'Frame {frame}, number of targets alive {number_of_targets}')#, number of estimated targets {number_of_estimated_targets}')
    
    
    
    
    
    
    
def plot_ellipse_3d(ax, center, width, height, angle, color='blue', alpha=0.5):
    # Generate ellipse points
    t = np.linspace(0, 2 * np.pi, 100)
    x = center[0] + width * np.cos(t) * np.cos(np.radians(angle)) - height * np.sin(t) * np.sin(np.radians(angle))
    y = center[1] + width * np.cos(t) * np.sin(np.radians(angle)) + height * np.sin(t) * np.cos(np.radians(angle))
    z = np.zeros_like(x)

    # Plot ellipse in 3D
    ax.plot(x, y, z, color=color, alpha=alpha)
    
def plot_filled_ellipse_3d(ax, center, radius_x, radius_y, color='yellow', alpha=0.5):
    # Generate points within the ellipse
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = center[0] + radius_x * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius_y * np.outer(np.sin(u), np.sin(v))
    z = np.zeros_like(x)  # Set Z to 0 since it's on the ground

    # Plot filled ellipse in 3D
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='none')

# Adjust this function to create the animation with 3D plots
def update_anim_plot_3d(frame, ax, target_tracks, birth_densities, measurements, all_means_for_plotting, sensor_list):
    ax.clear()
    ax.set_xlim(Parameters.surveillance_area[0][0]-20, Parameters.surveillance_area[0][1]+20)
    ax.set_ylim(Parameters.surveillance_area[0][0]-20, Parameters.surveillance_area[0][1]+20)
    ax.set_zlim(-1, 20)  # Set limits for z-axis based on your scenario
    
    # Set the background color
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # White background for x-plane
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # White background for y-plane
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # White background for z-plane

    # Remove gridlines
    ax.grid(False)
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)  # Set x gridlines transparent
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)  # Set y gridlines transparent
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)  # Set z gridlines transparent
    
    
    
    birth_means, birth_covariances = birth_densities
    target_tracks_matrix = np.array(target_tracks).reshape(-1, Parameters.number_of_timesteps)[::4]
    # estimated_targets_matrix = np.array(all_means_for_plotting).reshape(-1, Parameters.number_of_timesteps)[::4]
    
    birth_means, birth_covariances = birth_densities
    
    for i in range(birth_means.shape[1]):
        eigenvalues, eigenvectors = np.linalg.eig(birth_covariances[i,:,:])
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = 2 * np.sqrt(5.991 * eigenvalues[0])
        height = 2 * np.sqrt(5.991 * eigenvalues[1])
        
        plot_ellipse_3d(ax, (birth_means[:,i][0], birth_means[:,i][2]), width, height, angle, color='blue', alpha=0.5)
        

    # Plotting the sensor range as cones
    for sensor in sensor_list:
        sensor_pos = [sensor.all_selected_sensors[frame][0], sensor.all_selected_sensors[frame][1], 20]  # sensor position with height
        angles = np.linspace(0, 2 * np.pi, 30)
        x_base = sensor_pos[0] + Parameters.sensor_radius * np.cos(angles)
        y_base = sensor_pos[1] + Parameters.sensor_radius * np.sin(angles)
        z_base = np.zeros_like(x_base)  # Ground level for the base of the cone
        
        # Directly plot an ellipse at the base of the cone
        plot_filled_ellipse_3d(ax, (sensor_pos[0], sensor_pos[1]), Parameters.sensor_radius, Parameters.sensor_radius, color='yellow', alpha=0.5)
        
        # Draw a line from the sensor to each point around the base
        for xb, yb in zip(x_base, y_base):
            ax.plot([sensor_pos[0], xb], [sensor_pos[1], yb], [sensor_pos[2], 0], alpha=0.2, color='yellow')
    ax.plot([sensor.all_selected_sensors[frame][0]], [sensor.all_selected_sensors[frame][1]], [20], color='black', marker='o')

        
    # Plot target tracks on the ground
    for matrix in target_tracks:
        ax.plot(matrix[0, :frame], matrix[2, :frame], np.zeros(frame), 'g--')
        ax.plot([matrix[0, frame]], [matrix[2, frame]], [0], color='black', marker='o', alpha=0.3)

    # Plot measurements on the ground
    ax.scatter(measurements[frame][0], measurements[frame][1], np.zeros_like(measurements[frame][0]), marker='1', color='red', alpha=0.5)
    
    # calculating how many targets are alive at any given time
    
    number_of_targets = np.isnan
    number_of_targets = np.sum(~np.isnan(target_tracks_matrix[:,frame]))
    
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Height')
    ax.set_title(f'Frame {frame}, number of targets alive {number_of_targets}')



#########################################
#   R E S U L T S   S A V I N G   &   S T A T S
#########################################

def _as_2d_sensor_pos(x: float, y: float):
    return np.array([[float(x)], [float(y)]])


def _infer_obstacle_x_range(obstacles_list):
    """Infer an x-range [xmin, xmax] from the first obstacle in a list."""
    if not obstacles_list:
        return None, None

    obs = obstacles_list[0]

    # obstacle_three (axis-aligned rectangle)
    if hasattr(obs, 'bl') and hasattr(obs, 'tr'):
        try:
            xmin = float(np.asarray(obs.bl).reshape(-1)[0])
            xmax = float(np.asarray(obs.tr).reshape(-1)[0])
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            return xmin, xmax
        except Exception:
            pass

    # obstacle_two (general rectangle)
    if all(hasattr(obs, k) for k in ('A','B','C','D')):
        try:
            xs = [float(np.asarray(getattr(obs, k)).reshape(-1)[0]) for k in ('A','B','C','D')]
            return min(xs), max(xs)
        except Exception:
            pass

    # obstacle_one (circle)
    if hasattr(obs, 'location_centre') and hasattr(obs, 'radius'):
        try:
            cx = float(np.asarray(obs.location_centre).reshape(-1)[0])
            r = float(obs.radius)
            return cx - r, cx + r
        except Exception:
            pass

    return None, None


def sample_sensor_birth_positions(obstacles_list, y_values, margin=20, rng=None, fallback_x_range=None):
    """Sample sensor birth positions.

    Keeps each sensor's y fixed to y_values, randomises x uniformly across the
    first obstacle's x-extent expanded by `margin`.
    """
    if rng is None:
        rng = np.random

    xmin, xmax = _infer_obstacle_x_range(obstacles_list)

    if xmin is None or xmax is None:
        if fallback_x_range is None:
            try:
                xmin = float(Parameters.surveillance_area[0,0])
                xmax = float(Parameters.surveillance_area[0,1])
            except Exception:
                xmin, xmax = -10.0, 10.0
        else:
            xmin, xmax = map(float, fallback_x_range)

    low = float(xmin) - float(margin)
    high = float(xmax) + float(margin)

    positions = []
    for y in y_values:
        x = float(rng.uniform(low, high))
        positions.append(_as_2d_sensor_pos(x, float(y)))

    return positions


def _rmse_array(mc_errors):
    arr = np.asarray(mc_errors, dtype=float)
    return np.sqrt(arr)


def gospa_error_mean_std(all_errors):
    """Per-timestep mean/std across MC runs.

    The codebase stores p-power errors (p=2), so we convert to the displayed
    RMSE-like value via sqrt before computing statistics.
    """
    rmse = _rmse_array(all_errors)
    mean = np.mean(rmse, axis=0)
    std = np.std(rmse, axis=0, ddof=1) if rmse.shape[0] > 1 else np.zeros_like(mean)

    overall_mean = float(np.mean(mean))
    overall_std = float(np.mean(std))

    return mean, std, overall_mean, overall_std


def save_error_results(prefix, gospa_errors, loc_errors, miss_errors, false_errors, out_dir='results'):
    """Save raw MC errors plus mean/std summaries.

    Saves:
      - {out_dir}/{prefix}_errors.npz
      - {out_dir}/{prefix}_summary.csv
    """
    import os

    os.makedirs(out_dir, exist_ok=True)

    gospa_mean, gospa_std, gospa_overall_mean, gospa_overall_std = gospa_error_mean_std(gospa_errors)
    loc_mean, loc_std, loc_overall_mean, loc_overall_std = gospa_error_mean_std(loc_errors)
    miss_mean, miss_std, miss_overall_mean, miss_overall_std = gospa_error_mean_std(miss_errors)
    false_mean, false_std, false_overall_mean, false_overall_std = gospa_error_mean_std(false_errors)

    np.savez(
        os.path.join(out_dir, f'{prefix}_errors.npz'),
        gospa=np.asarray(gospa_errors, dtype=float),
        localisation=np.asarray(loc_errors, dtype=float),
        missed=np.asarray(miss_errors, dtype=float),
        false=np.asarray(false_errors, dtype=float),
        gospa_mean=gospa_mean,
        gospa_std=gospa_std,
        localisation_mean=loc_mean,
        localisation_std=loc_std,
        missed_mean=miss_mean,
        missed_std=miss_std,
        false_mean=false_mean,
        false_std=false_std,
        overall_mean=np.array([gospa_overall_mean, loc_overall_mean, miss_overall_mean, false_overall_mean]),
        overall_std=np.array([gospa_overall_std, loc_overall_std, miss_overall_std, false_overall_std]),
    )

    timesteps = np.arange(len(gospa_mean))
    summary = np.column_stack([
        timesteps,
        gospa_mean, gospa_std,
        loc_mean, loc_std,
        miss_mean, miss_std,
        false_mean, false_std,
    ])

    header = 'timestep,gospa_mean,gospa_std,localisation_mean,localisation_std,missed_mean,missed_std,false_mean,false_std'
    np.savetxt(
        os.path.join(out_dir, f'{prefix}_summary.csv'),
        summary,
        delimiter=',',
        header=header,
        comments='',
    )

    return {
        'overall_std_gospa_mean_over_time': gospa_overall_std,
        'overall_std_localisation_mean_over_time': loc_overall_std,
        'overall_std_missed_mean_over_time': miss_overall_std,
        'overall_std_false_mean_over_time': false_overall_std,
    }
