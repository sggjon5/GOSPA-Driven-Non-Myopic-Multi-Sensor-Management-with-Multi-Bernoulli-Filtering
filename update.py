# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 20:10:50 2023

@author: sggjone5
"""

import numpy as np

import Parameters




class Updater():
    
    def __init__(self, predictor, measurements):
        
        self.predictor = predictor
        self.measurements = measurements
        
    def update_pmb_empty_poisson(self, lambdau, xu, Pu, prob_existence, x, P, z, sensor_list):
        
        prob_detection = Parameters.prob_detection
        H = Parameters.H
        R = Parameters.R
        lambda_false_alarm = Parameters.lambda_false_alarm
        lambda_birth_threshold = Parameters.lambda_birth_threshold
        
        # Get the number of existing tracks and measurements
        n = len(prob_existence)
        stateDimensions, nu = xu.shape[0], xu.shape[1]
        measDimensions, m = z.shape[0], z.shape[1]
    
        # Initialize arrays for storing updated values
        wupd = np.zeros((n, m + 1))
        rupd = np.zeros((n, m + 1))
        xupd = np.zeros((m + 1, stateDimensions, n))
        Pupd = np.zeros((n, m + 1, stateDimensions, stateDimensions))
    
        wnew = np.zeros(m)
        rnew = np.zeros(m)
        xnew = np.zeros((stateDimensions, m))
        Pnew = np.zeros((m, stateDimensions, stateDimensions))
    
        Sk = np.zeros((nu, measDimensions, measDimensions))
        Kk = np.zeros((nu, stateDimensions, measDimensions))
        Pk = np.zeros((nu, stateDimensions, stateDimensions))
        ck = np.zeros(nu)
        sqrt_det2piSk = np.zeros(nu)
        yk = np.zeros((stateDimensions, nu))
    
        # check whether the measurement is within the FOV, if yes pd=pd if no pd = 0
        for i in range(n):
            
            if np.linalg.norm(sensor_list[0].current_sensor_position - x[:,[i]][::2]) < Parameters.sensor_radius:
                prob_detection = Parameters.prob_detection
                
            else:
                # print('inside')
                prob_detection = 0
                
            # Update existing tracks and measurements  
            wupd[i, 0] = 1 - prob_existence[i] + prob_existence[i] * (1 - prob_detection)
            rupd[i, 0] = prob_existence[i] * (1 - prob_detection) / wupd[i, 0]
            xupd[0, :, i] = x[:, i]
            Pupd[i, 0, :, :] = P[i, :, :]
    
            # Calculate measurement-related values
            S = np.dot(np.dot(H, P[i, :, :]), H.T) + R
            sqrt_det2piS = np.sqrt(np.linalg.det(2 * np.pi * S))
            K = np.dot(np.dot(P[i, :, :], H.T), np.linalg.inv(S))
            Pplus = P[i, :, :] - np.dot(np.dot(K, H), P[i, :, :])
    
            for j in range(m):
                v = z[:, j] - np.dot(H, x[:, i])
                wupd[i, j + 1] = prob_existence[i] * prob_detection * np.exp(-0.5 * np.dot(v.T, np.dot(np.linalg.inv(S), v))) / sqrt_det2piS
                rupd[i, j + 1] = 1
                xupd[j + 1, :, i] = x[:, i] + np.dot(K, v)
                Pupd[i, j + 1, :, :] = Pplus
    
        for k in range(nu):
            # Calculate parameters for PPP components
            Sk[k, :, :] = np.dot(np.dot(H, Pu[k, :, :]), H.T) + R
            sqrt_det2piSk[k] = np.sqrt(np.linalg.det(2 * np.pi * Sk[k, :, :]))
            Kk[k, :, :] = np.dot(np.dot(Pu[k, :, :], H.T), np.linalg.inv(Sk[k, :, :]))
            Pk[k, :, :] = Pu[k, :, :] - np.dot(np.dot(Kk[k, :, :], H), Pu[k, :, :])
    
        for j in range(m):
            # Update PPP components using measurements
            ck = np.zeros(nu)
            for k in range(nu):
                v = z[:, j] - np.dot(H, xu[:, k])
                ck[k] = lambdau[k] * prob_detection * np.exp(-0.5 * np.dot(v.T, np.dot(np.linalg.inv(Sk[k, :, :]), v))) / sqrt_det2piSk[k]
                yk[:, k] = xu[:, k] + np.dot(Kk[k, :, :], v)
    
            C = np.sum(ck)
            wnew[j] = lambda_false_alarm # C + lambda_false_alarm
            rnew[j] = 0 # C / wnew[j]
            ck = ck / C
            xnew[:, j] = np.zeros((4,)) # np.dot(yk, ck)
    
            for k in range(nu):
                v = xnew[:, j] - yk[:, k]
                Pnew[j, :, :] = np.eye(4) # Pnew[j, :, :] + ck[k] * (Pk[k, :, :] + np.outer(v, v))
    
        lambdau = (1 - prob_detection) * lambdau
    
        # Remove components with low weight
        ss = lambdau > lambda_birth_threshold
        lambdau = lambdau[ss]
        xu = xu[:, ss]
        Pu = Pu[ss, :, :]
        
    
        return lambdau, xu, Pu, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew
        
    def update_pmb(self, lambdau, xu, Pu, prob_existence, x, P, z, sensor_list):
        
        prob_detection = Parameters.prob_detection
        H = Parameters.H
        R = Parameters.R
        lambda_false_alarm = Parameters.lambda_false_alarm
        lambda_birth_threshold = Parameters.lambda_birth_threshold
        
        # Get the number of existing tracks and measurements
        n = len(prob_existence)
        stateDimensions, nu = xu.shape[0], xu.shape[1]
        measDimensions, m = z.shape[0], z.shape[1]
    
        # Initialize arrays for storing updated values
        wupd = np.zeros((n, m + 1))
        rupd = np.zeros((n, m + 1))
        xupd = np.zeros((m + 1, stateDimensions, n))
        Pupd = np.zeros((n, m + 1, stateDimensions, stateDimensions))
    
        wnew = np.zeros(m)
        rnew = np.zeros(m)
        xnew = np.zeros((stateDimensions, m))
        Pnew = np.zeros((m, stateDimensions, stateDimensions))
    
        Sk = np.zeros((nu, measDimensions, measDimensions))
        Kk = np.zeros((nu, stateDimensions, measDimensions))
        Pk = np.zeros((nu, stateDimensions, stateDimensions))
        ck = np.zeros(nu)
        sqrt_det2piSk = np.zeros(nu)
        yk = np.zeros((stateDimensions, nu))
    
    
        for i in range(n):
            
            if np.linalg.norm(sensor_list[0].current_sensor_position - x[:,[i]][::2]) < Parameters.sensor_radius:
                prob_detection = Parameters.prob_detection
                    
            else:
                prob_detection = 0
                
            # Update existing tracks and measurements
            wupd[i, 0] = 1 - prob_existence[i] + prob_existence[i] * (1 - prob_detection)
            rupd[i, 0] = prob_existence[i] * (1 - prob_detection) / wupd[i, 0]
            xupd[0, :, i] = x[:, i]
            Pupd[i, 0, :, :] = P[i, :, :]
    
            # Calculate measurement-related values
            S = np.dot(np.dot(H, P[i, :, :]), H.T) + R
            sqrt_det2piS = np.sqrt(np.linalg.det(2 * np.pi * S))
            K = np.dot(np.dot(P[i, :, :], H.T), np.linalg.inv(S))
            Pplus = P[i, :, :] - np.dot(np.dot(K, H), P[i, :, :])
    
            for j in range(m):
                
                if np.linalg.norm(sensor_list[0].current_sensor_position - z[:,[j]]) < Parameters.sensor_radius:
                    prob_detection = Parameters.prob_detection
                        
                else:
                    prob_detection = 0
                    
                v = z[:, j] - np.dot(H, x[:, i])
                wupd[i, j + 1] = prob_existence[i] * prob_detection * np.exp(-0.5 * np.dot(v.T, np.dot(np.linalg.inv(S), v))) / sqrt_det2piS
                rupd[i, j + 1] = 1
                xupd[j + 1, :, i] = x[:, i] + np.dot(K, v)
                Pupd[i, j + 1, :, :] = Pplus
    
        for k in range(nu):
            
            # Calculate parameters for PPP components
            Sk[k, :, :] = np.dot(np.dot(H, Pu[k, :, :]), H.T) + R
            sqrt_det2piSk[k] = np.sqrt(np.linalg.det(2 * np.pi * Sk[k, :, :]))
            Kk[k, :, :] = np.dot(np.dot(Pu[k, :, :], H.T), np.linalg.inv(Sk[k, :, :]))
            Pk[k, :, :] = Pu[k, :, :] - np.dot(np.dot(Kk[k, :, :], H), Pu[k, :, :])
    
        for j in range(m):
            
            # Update PPP components using measurements
            ck = np.zeros(nu)
            for k in range(nu):
                if np.linalg.norm(sensor_list[0].current_sensor_position - xu[:,[k]][::2]) < Parameters.sensor_radius:
                    prob_detection = Parameters.prob_detection
                        
                else:
                    prob_detection = 0
                    
                v = z[:, j] - np.dot(H, xu[:, k])
                ck[k] = lambdau[k] * prob_detection * np.exp(-0.5 * np.dot(v.T, np.dot(np.linalg.inv(Sk[k, :, :]), v))) / sqrt_det2piSk[k]
                yk[:, k] = xu[:, k] + np.dot(Kk[k, :, :], v)

            C = np.sum(ck)
            wnew[j] = C + lambda_false_alarm
            rnew[j] = C / wnew[j]
            ck = ck / C
            xnew[:, j] = np.dot(yk, ck)
    
            for k in range(nu):
                v = xnew[:, j] - yk[:, k]
                Pnew[j, :, :] = Pnew[j, :, :] + ck[k] * (Pk[k, :, :] + np.outer(v, v))

        for k in range(nu):
            # need to multiply all by 1-pd but the ones we cannot see pd = 0
            
            if np.linalg.norm(sensor_list[0].current_sensor_position - xu[:,[k]][::2]) < Parameters.sensor_radius:
                prob_detection = Parameters.prob_detection
                    
            else:
                prob_detection = 0
                
            lambdau[k] = (1 - prob_detection) * lambdau[k]
    
        # Truncate low weight components
        ss = lambdau > lambda_birth_threshold
        lambdau = lambdau[ss]
        xu = xu[:, ss]
        Pu = Pu[ss, :, :]
        

        return lambdau, xu, Pu, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew

