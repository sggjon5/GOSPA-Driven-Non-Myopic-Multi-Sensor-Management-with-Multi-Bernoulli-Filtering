# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:44:25 2023
@author: sggjone5
"""

import numpy as np

def momb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew):
    # Define a threshold for low track existence probability
    r_threshold = 1e-4

    # Get the dimensions of the input arrays
    nold, mp1 = pupd.shape
    stateDimensions = xnew.shape[0]
    m = mp1 - 1  # m represents the number of new tracks
    n = nold + m  # n represents the total number of tracks (existing + new)

    # Initialize arrays for the resulting tracks
    r = np.zeros(n)  # Track existence probabilities
    x = np.zeros((stateDimensions, n))  # Track states
    P = np.zeros((n, stateDimensions, stateDimensions))  # Track covariance matrices

    # Generate legacy (missed detection) tracks from existing tracks
    r[:nold] = pupd[:, 0] * rupd[:, 0]  # Calculate existence probability for each legacy track
    x[:, :nold] = xupd[0, :, :]  # Set state for each legacy track
    P[:nold, :, :] = Pupd[:, 0, :, :]  # Set covariance for each legacy track

    # Generate updated tracks for each measurement
    for j in range(m):
        i = j + nold  # Index for new tracks
        prnew = pnew[j] * rnew[j]  # Calculate existence probability for the new track
        pr = pupd[:, j + 1] * rupd[:, j + 1]  # Calculate existence probability for existing tracks
        r[i] = np.sum(pr) + prnew  # Calculate total existence probability for the new track
        pr = pr / r[i]  # Normalize existence probabilities for existing tracks
        prnew = prnew / r[i]  # Normalize existence probability for the new track
        x[:, i] = np.dot(xupd[j+1, :, :], pr) + xnew[:, j] * prnew  # Update the state for the new track
        v = x[:, i] - xnew[:, j]  # Calculate the state difference
        P[i, :, :] = prnew * (Pnew[j, :, :] + np.outer(v, v))  # Update the covariance for the new track
        for i2 in range(nold):
            v = x[:, i] - xupd[j+1, :, i2]  # Calculate state difference for existing tracks
            P[i, :, :] += pr[i2] * (Pupd[i2, j + 1, :, :] + np.outer(v, v))  # Update the covariance for existing tracks

    # Truncate tracks with low probability of existence (not shown in the algorithm)
    ss = r > r_threshold  # Identify tracks with sufficient existence probability
    r = r[ss]  # Remove tracks with low existence probability
    x = x[:, ss]  # Remove corresponding track states
    P = P[ss, :, :]  # Remove corresponding track covariances

    return r, x, P  # Return updated track parameters
