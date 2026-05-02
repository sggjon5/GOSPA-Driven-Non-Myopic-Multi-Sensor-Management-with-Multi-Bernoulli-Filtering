# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:43:14 2023
@author: sggjone5
"""

import numpy as np


def lbp(wupd, wnew):
    # Get the dimensions of the input arrays
    n, mp1 = wupd.shape
    m = mp1 - 1  # m represents the number of new tracks

    # Define a threshold for convergence
    eps_conv_threshold = 1e-4

    # Initialize belief matrices
    mu = np.ones((n, m))  # mu_ba
    mu_old = np.zeros((n, m))
    nu = np.zeros((n, m))  # mu_ab

    # Initialize arrays for storing updated probabilities
    pupd = np.zeros((n, mp1))  # Updated probabilities for existing tracks
    pnew = np.zeros((m, 1))  # Updated probabilities for new tracks

    if (mu - mu_old).shape[1] != 0 and (mu - mu_old).shape[0] != 0:
        # Run Loopy Belief Propagation (LBP) iteration
        while np.max(np.abs(mu.T - mu_old.T)) > eps_conv_threshold:
            mu_old = mu

            # Update beliefs in both directions (forward and backward)
            for i in range(n):
                prd = wupd[i, 1:] * mu[i, :]  # Combine wupd and beliefs in the backward direction
                s = wupd[i, 0] + np.sum(prd)  # Calculate the sum of probabilities
                nu[i, :] = wupd[i, 1:] / (s - prd)  # Update nu beliefs

            for j in range(m):
                s = wnew[j] + np.sum(nu[:, j])  # Calculate the sum of probabilities
                mu[:, j] = 1.0 / (s - nu[:, j])  # Update mu beliefs

    # Calculate outputs, both for existing tracks and new tracks
    for i in range(n):
        s = wupd[i, 0] + np.sum(wupd[i, 1:] * mu[i, :])
        pupd[i, 0] = wupd[i, 0] / s  # Calculate updated existence probability for each existing track
        pupd[i, 1:] = wupd[i, 1:] * mu[i, :] / s  # Calculate updated probabilities for each existing track

    for j in range(m):
        s = wnew[j] + np.sum(nu[:, j])
        pnew[j] = wnew[j] / s  # Calculate updated existence probability for each new track

    return pupd, pnew  # Return updated probabilities for existing and new tracks






