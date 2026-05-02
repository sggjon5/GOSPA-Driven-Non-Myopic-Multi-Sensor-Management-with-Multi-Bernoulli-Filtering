# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:46:17 2023
@author: sggjone5
"""

import numpy as np

def tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew):
    # Define a threshold for low track existence probability
    r_threshold = 1e-4#1e-4

    # Get the dimensions of the input arrays
    nold, mp1 = pupd.shape
    stateDimensions = xnew.shape[0]
    m = mp1 - 1  # m represents the number of new tracks
    n = nold + m  # n represents the total number of tracks (existing + new)

    # Initialize arrays for the resulting tracks
    r = np.zeros(n)  # Track existence probabilities
    x = np.zeros((stateDimensions, n))  # Track states
    P = np.zeros((n, stateDimensions, stateDimensions))  # Track covariance matrices

    # Form continuing tracks from existing tracks
    for i in range(nold):
        pr = pupd[i, :] * rupd[i, :]  # Combine the existence and update probabilities
        r[i] = np.sum(pr)  # Update the track existence probability
        pr = (pr.T / r[i]).reshape(-1, 1)  # Normalize probabilities
        x[:, [i]] = xupd[:,:,i].T @ pr  # Update the track state using weighted averages
        for j in range(mp1):
            v = x[:, i] - xupd[j, :, i]
            P[i, :, :] = P[i, :, :] + pr[j] * (Pupd[i, j, :, :] + np.outer(v, v))  # Update the track covariance

    # Form new tracks (already single hypothesis)
    r[nold:] = (pnew * rnew[:, np.newaxis]).squeeze()  # Set existence probabilities for new tracks
    x[:, nold:] = xnew  # Set state for new tracks
    P[nold:, :, :] = Pnew  # Set covariance for new tracks

    # Truncate tracks with low probability of existence (not shown in the algorithm)
    ss = r > r_threshold  # Identify tracks with sufficient existence probability
    r = r[ss]  # Remove tracks with low existence probability
    x = x[:, ss]  # Remove corresponding track states
    P = P[ss, :, :]  # Remove corresponding track covariances

    return r, x, P  # Return updated track parameters




