# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:27:20 2021

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""
# Importing libraries
import pandas as pd
import numpy as np

class NTSK:
    def __init__(self, n_clusters = 5, lambda1 = 1, RLS_option = 1, omega = 1000):
        self.hyperparameters = pd.DataFrame({'n_clusters':[n_clusters], 'lambda1':[lambda1], 'omega':[omega]})
        self.parameters = pd.DataFrame(columns = ['Center', 'P', 'p_vector', 'Theta', 'NumObservations', 'tau', 'weight', 'sigma'])
        # Select consequent parameters algorithm
        self.RLS_option = RLS_option
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
        # Computing the residual square in the testing phase
        self.ResidualTestPhase = np.array([])
        # Control variables
        self.ymin = 0.
        self.ymax = 0.
        self.region = 0.
        self.last_y = 0.
         
    def fit(self, X, y):
        # Concatenate X with y
        Data = np.concatenate((X, y.reshape(-1,1), np.zeros((X.shape[0], 2))), axis=1)
        # Compute the number of attributes
        m = X.shape[1]
        # Compute the number of samples
        n = X.shape[0]
        # Compute the angles
        for row in range(1, n):
            Data[row, m + 1] = Data[row, m] - Data[row - 1, m] 
        # Compute the width of each interval
        self.ymin = min(Data[:, m + 1])
        self.ymax = max(Data[:, m + 1])
        self.region = ( self.ymax - self.ymin ) / ( self.hyperparameters.loc[0, 'n_clusters'] )
        # Compute the cluster of the inpute
        for row in range(1, n):
            if Data[row, m + 1] < self.ymax:
                rule = int( ( Data[row, m + 1] - self.ymin ) / self.region )
                Data[row, m + 2] = rule
            else:
                rule = int( ( Data[row, m + 1] - self.ymin ) / self.region )
                Data[row, m + 2] = rule - 1
        # Create a dataframe from the array
        df = pd.DataFrame(Data)
        empty = []
        # Initializing the rules
        for rule in range(self.hyperparameters.loc[0, 'n_clusters']):
            if df[df[m + 2] == rule].shape[0] == 0:
                empty.append(rule)
            dfnew = df[df[m + 2] == rule]
            center = dfnew.loc[:,:m-1].mean().values.reshape(-1,1)
            std = dfnew.loc[:,:m-1].std().values.reshape(-1,1)
            if np.isnan(std).any:
                std[np.isnan(std)] = 1.
            if 0. in std:
                std[std == 0.] = 1.
            if rule == 0:
                # Initialize the first rule
                self.Initialize_First_Cluster(center, y[0], std)
            else:
                # Initialize the first rule
                self.Initialize_Cluster(center, y[0], std)
        if empty != 0:
            self.parameters = self.parameters.drop(empty)
        for k in range(1, X.shape[0]):

            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            # Update the rule
            self.Rule_Update(df.loc[k, m + 2])
            # Compute the normalized firing level
            self.weight(x)
            # Update the consequent parameters of the rule
            if self.RLS_option == 1:
                self.RLS(x, y[k], xe)
            elif self.RLS_option == 2:
                self.wRLS(x, y[k], xe)
            # Compute the output
            Output =  xe.T @ self.parameters.loc[df.loc[k, m + 2], 'Theta']
            # Store the results
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
        return self.OutputTrainingPhase
            
    def predict(self, X):
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'Center'].shape[0])
        for k in range(X.shape[0]):
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            # Compute the normalized firing level
            self.weight(x)
            # Compute the output
            Output = 0
            for row in self.parameters.index:
                Output = Output + self.parameters.loc[row, 'weight'] * xe.T @ self.parameters.loc[row, 'Theta']
            # Store the output
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output)
        return self.OutputTestPhase[-X.shape[0]:]
        
    def Initialize_First_Cluster(self, x, y, std):
        Theta = np.insert(np.zeros((x.shape[0], 1)).T, 0, y, axis=1).T
        self.parameters = pd.DataFrame([[x, self.hyperparameters.loc[0, 'omega'] * np.eye(x.shape[0] + 1), np.zeros(Theta.shape), Theta, 1., 0., std]], columns = ['Center', 'P', 'p_vector', 'Theta', 'NumObservations', 'weight', 'sigma'])
        Output = y
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y)**2)
    
    def Initialize_Cluster(self, x, y, std):
        Theta = np.insert(np.zeros((x.shape[0], 1)).T, 0, y, axis=1).T
        NewRow = pd.DataFrame([[x, self.hyperparameters.loc[0, 'omega'] * np.eye(x.shape[0] + 1), np.zeros(Theta.shape), Theta, 1., 0., std]], columns = ['Center', 'P', 'p_vector', 'Theta', 'NumObservations', 'weight', 'sigma'])
        self.parameters = pd.concat([self.parameters, NewRow], ignore_index=True)

    def Rule_Update(self, i):
        # Update the number of observations in the rule
        self.parameters.loc[i, 'NumObservations'] = self.parameters.loc[i, 'NumObservations'] + 1
            
    def Firing_Level(self, m, x, sigma):
        return np.exp( - ( 1/2 ) * ( ( m - x )**2 ) / ( sigma**2 ) )
    
    def tau(self, x):
        for row in self.parameters.index:
            tau = np.prod( self.Firing_Level(self.parameters.loc[row, 'Center'], x, self.parameters.loc[row, 'sigma'] ) )
            # Evoid mtau with values zero
            if abs(tau) < (10 ** -100):
                tau = (10 ** -100)
            self.parameters.at[row, 'tau'] = tau
            
    def weight(self, x):
        self.tau(x)
        for row in self.parameters.index:
            self.parameters.at[row, 'weight'] = self.parameters.loc[row, 'tau'] / sum(self.parameters['tau'])

    
    def RLS(self, x, y, xe):
        """
        Conventional RLS algorithm
        Adaptive Filtering - Paulo S. R. Diniz
        
        Parameters:
            lambda: forgeting factor
    
        """
        for row in self.parameters.index:
            # K is used here just to make easier to see the equation of the covariance matrix
            K = ( self.parameters.loc[row, 'P'] @ xe ) / ( self.hyperparameters.loc[0,'lambda1'] + xe.T @ self.parameters.loc[row, 'P'] @ xe )
            self.parameters.at[row, 'P'] = ( 1 / self.hyperparameters.loc[0,'lambda1'] ) * ( self.parameters.loc[row, 'P'] - K @ xe.T @ self.parameters.loc[row, 'P'] )
            self.parameters.at[row, 'Theta'] = self.parameters.loc[row, 'Theta'] + ( self.parameters.loc[row, 'P'] @ xe ) * (y - xe.T @ self.parameters.loc[row, 'Theta'] )


    def wRLS(self, x, y, xe):
        """
        weighted Recursive Least Square (wRLS)
        An Approach to Online Identification of Takagi-Sugeno Fuzzy Models - Angelov and Filev

        """
        for row in self.parameters.index:
            self.parameters.at[row, 'P'] = self.parameters.loc[row, 'P'] - (( self.parameters.loc[row, 'weight'] * self.parameters.loc[row, 'P'] @ xe @ xe.T @ self.parameters.loc[row, 'P'])/(1 + self.parameters.loc[row, 'weight'] * xe.T @ self.parameters.loc[row, 'P'] @ xe))
            self.parameters.at[row, 'Theta'] = ( self.parameters.loc[row, 'Theta'] + (self.parameters.loc[row, 'P'] @ xe * self.parameters.loc[row, 'weight'] * (y - xe.T @ self.parameters.loc[row, 'Theta'])) )
        
        
    # def cRLS(self, x, y, xe):
    #     """
    #     weighted Recursive Least Square with forgetting factor (cRLS)
    #     Proposed

    #     """
    #     for row in self.parameters.index:
    #         self.parameters.at[row, 'P'] = ( 1 / self.hyperparameters.loc[0,'lambda1'] ) * ( self.parameters.loc[row, 'P'] - (( self.parameters.loc[row, 'weight'] * self.parameters.loc[row, 'P'] @ xe @ xe.T @ self.parameters.loc[row, 'P'])/(self.hyperparameters.loc[0,'lambda1'] + self.parameters.loc[row, 'weight'] * xe.T @ self.parameters.loc[row, 'P'] @ xe)) )
    #         self.parameters.at[row, 'Theta'] = ( self.parameters.loc[row, 'Theta'] + (self.parameters.loc[row, 'P'] @ xe * self.parameters.loc[row, 'weight'] * (y - xe.T @ self.parameters.loc[row, 'Theta'])) )
