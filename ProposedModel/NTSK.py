# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:27:20 2021

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class NTSK:
    def __init__(self, rules = 5, lambda1 = 1, adaptive_filter = "RLS", fuzzy_operator = "prod", omega = 1000):
        
        """Regression based on New Takagi-Sugeno-Kang.

        The target is predicted by creating rules, composed of fuzzy sets.
        Then, the output is computed as a weighted average of each local output 
        (output of each rule).

        Read more in the paper https://doi.org/10.1016/j.engappai.2024.108155.


        Parameters
        ----------
        rules : int, default=5
            Number of fuzzy rules will be created.

        lambda1 : float, possible values are in the interval [0,1], default=1
            Defines the forgetting factor for the algorithm to estimate the consequent parameters.
            This parameters is only used when RLS_option is "RLS"

        adaptive_filter : {'RLS', 'wRLS'}, default='RLS'
            Algorithm used to compute the consequent parameters:

            - 'RLS' will use :class:`RLS`
            - 'wRLS' will use :class:`wRLS`
        
        fuzzy_operator : {'prod', 'max', 'min'}, default='prod'
            Choose the fuzzy operator:

            - 'prod' will use :`product`
            - 'max' will use :class:`maximum value`
            - 'min' will use :class:`minimum value`

        omega : int, default=1000
            Omega is a parameters used to initialize the algorithm to estimate
            the consequent parameters

        n_jobs : int, default=None
            The number of parallel jobs to run for neighbors search.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
            Doesn't affect :meth:`fit` method.

        Attributes
        ----------
        

        n_features_in_ : int
            Number of features seen during :term:`fit`.

            .. versionadded:: 0.24

        feature_names_in_ : ndarray of shape (`n_features_in_`,)
            Names of features seen during :term:`fit`. Defined only when `X`
            has feature names that are all strings.

            .. versionadded:: 1.0

        n_samples_fit_ : int
            Number of samples in the fitted data.

        See Also
        --------
        NMC : New Mamdani Classifier. Implements a new Mamdani approach for classification.
        NMR : New Mamdani Regressor. Implements a new Mamdani approach for regression.
        KNeighborsClassifier : Classifier implementing the k-nearest neighbors vote.
        RadiusNeighborsClassifier : Classifier implementing
            a vote among neighbors within a given radius.

        Notes
        -----
        
        NMC is a specific case of NTSK for classification.

        """
        
        # Validate `rules`: positive integer
        # if not isinstance(rules, int) or rules <= 0:
        if rules <= 0:
            raise ValueError("`rules` must be a positive integer.")

        # Validate `lambda1`: [0, 1]
        if not isinstance(lambda1, (float, int)) or not (0 <= lambda1 <= 1):
            raise ValueError("`lambda1` must be a float in the interval [0, 1].")

        # Validate `adaptive_filter`: 'RLS' or 'wRLS'
        if adaptive_filter not in {"RLS", "wRLS"}:
            raise ValueError("`adaptive_filter` must be either 'RLS' or 'wRLS'.")

        # Validate `fuzzy_operator`: 'prod', 'max', 'min'
        if fuzzy_operator not in {"prod", "max", "min"}:
            raise ValueError("`fuzzy_operator` must be one of {'prod', 'max', 'min'}.")

        # Validate `omega`: positive integer
        if not isinstance(omega, int) or omega <= 0:
            raise ValueError("`omega` must be a positive integer.")
        
        # Hyperparameters
        self.rules = rules
        self.lambda1 = lambda1
        self.adaptive_filter = adaptive_filter
        self.fuzzy_operator = fuzzy_operator
        self.omega = omega
        
        # Define the rule-based structure
        if self.adaptive_filter == "RLS":
            self.parameters = pd.DataFrame(columns = ['mean', 'std', 'NumObservations'])
            self.parameters_RLS = {}
        if self.adaptive_filter == "wRLS":
            self.parameters = pd.DataFrame(columns = ['mean', 'std', 'P', 'p_vector', 'Theta', 'NumObservations', 'tau', 'weight'])
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
        self.X_ = []

    def get_params(self, deep=True):
        return {
            'rules': self.rules,
            'lambda1': self.lambda1,
            'adaptive_filter': self.adaptive_filter,
            'fuzzy_operator': self.fuzzy_operator,
            'omega': self.omega,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
         
    def fit(self, X, y):
        
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        
        # Check wheather y is 1d
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y.shape) > 1:
            y = y.ravel()
        
        # Check wheather y is 1d
        if X.shape[0] != y.shape[0]:
            raise TypeError(
                "The number of samples of X are not compatible with the number of samples in y. "
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(y):
            raise ValueError(
                "y contains incompatible values."
                " Check y for non-numeric or infinity values"
            )
        
        # Concatenate X with y
        Data = np.hstack((X, y.reshape(-1, 1), np.zeros((X.shape[0], 2))))
        
        # Compute the number of attributes and samples
        m, n = X.shape[1], X.shape[0]
        
        # Vectorized angle calculation
        Data[1:, m + 1] = np.diff(Data[:, m])
        
        # Min and max calculations and region calculation
        self.ymin, self.ymax = Data[:, m + 1].min(), Data[:, m + 1].max()
        self.region = (self.ymax - self.ymin) / self.rules
        
        # Compute the cluster of the inpute
        for row in range(1, n):
            if Data[row, m + 1] < self.ymax:
                rule = int((Data[row, m + 1] - self.ymin) / self.region)
                Data[row, m + 2] = rule
            else:
                rule = int((Data[row, m + 1] - self.ymin) / self.region)
                Data[row, m + 2] = rule - 1
                
        # Create a dataframe from the array
        df = pd.DataFrame(Data)
        empty = []
        
        # Initialize rules vectorized
        for rule in range(self.rules):
            dfnew = df[df[m + 2] == rule]
            if dfnew.empty:
                empty.append(rule)
                # continue
            mean = dfnew.iloc[:, :m].mean().values[:, None]
            self.X_.append(dfnew.iloc[:, :m].values)
            std = np.nan_to_num(dfnew.iloc[:, :m].std().values[:, None], nan=1.0)
            self.initialize_rule(mean, y[0], std, is_first=(rule == 0))
                
        if empty:
            self.parameters.drop(empty, inplace=True, errors='ignore')
            
        for k in range(1, n):

            # Prepare the k-th input vector
            x = X[k, :].reshape((1, -1)).T
            xe = np.insert(x.T, 0, 1, axis=1).T
            rule = int(df.loc[k, m + 2])
            
            # Update the rule
            self.Rule_Update(rule)
            
            # Update the consequent parameters of the rule
            if self.adaptive_filter == "RLS":
                self.RLS(x, y[k], xe)
            elif self.adaptive_filter == "wRLS":
                self.weight(x)
                self.wRLS(x, y[k], xe)
                
            try:
                if self.adaptive_filter == "RLS":
                    # Compute the output based on the most compatible rule
                    Output = xe.T @ self.parameters_RLS['Theta']
                elif self.adaptive_filter == "wRLS":
                    # Compute the output based on the most compatible rule
                    Output = xe.T @ self.parameters.at[rule, 'Theta']
                
                # Store the results
                self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
                self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, (Output - y[k]) ** 2)
            except:
                
                if self.adaptive_filter == "RLS":
                
                    # Call the model with higher lambda 
                    self.inconsistent_lambda(X, y)
                    
                    # Return the results
                    return self.OutputTrainingPhase
                
            if self.adaptive_filter == "RLS":
                if np.isnan(self.parameters_RLS['Theta']).any() or np.isinf(self.ResidualTrainingPhase).any():
                    
                    # Call the model with higher lambda 
                    self.inconsistent_lambda(X, y)
                    
                    # Return the results
                    return self.OutputTrainingPhase
            
        return self.OutputTrainingPhase
            
    def predict(self, X):
        
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
        
        # Prepare the inputs
        X = X.reshape(-1, self.parameters.loc[0, 'mean'].shape[0])
        self.OutputTestPhase = np.array([])
        
        for x in X:
            
            # Prepare the first input vector
            x = x.reshape((1, -1)).T
            
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            
            if self.adaptive_filter == "RLS":
                # Compute the output based on the most compatible rule
                Output = xe.T @ self.parameters_RLS['Theta']
            
            elif self.adaptive_filter == "wRLS":
                
                # Compute the normalized firing degree
                self.weight(x)
            
                # Compute the output
                Output = sum(self.parameters.loc[row, 'weight'] * xe.T @ self.parameters.loc[row, 'Theta'] for row in self.parameters.index)
                
            # Store the output
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output)
            
        return np.array(self.OutputTestPhase)
    
    def show_rules(self):
        rules = []
        for i in self.parameters.index:
            rule = f"Rule {i}"
            for j in range(self.parameters.loc[i,"mean"].shape[0]):
                rule = f'{rule} - {self.parameters.loc[i,"mean"][j].item():.2f} ({self.parameters.loc[i,"std"][j].item():.2f})'
            print(rule)
            rules.append(rule)
        
        return rules
    
    def plot_hist(self, bins=10):
        # Set plot-wide configurations only once
        plt.rc('font', size=30)
        plt.rc('axes', titlesize=30)
        
        # Iterate through rules and attributes
        for i, data in enumerate(self.X_):
            for j in range(data.shape[1]):
                # Create and configure the plot
                plt.figure(figsize=(19.20, 10.80))  # Larger figure for better clarity
                plt.hist(
                    data[:, j], 
                    bins=bins, 
                    alpha=0.7,  # Slight transparency for better visuals
                    color='blue', 
                    edgecolor='black'
                )
                # Add labels and titles
                plt.title(f'Rule {i} - Attribute {j}')
                plt.xlabel('Values')
                plt.ylabel('Frequency')
                plt.grid(False)
                
                # Display the plot
                plt.show()
    
    def inconsistent_lambda(self, X, y):
        
        print(f'The lambda1 of {self.lambda1:.2f} is producing inconsistent values. The new value will be set to {0.01+self.lambda1:.2f}')
        
        # Initialize the model
        model = NTSK(rules = self.rules, lambda1 = 0.01 + self.lambda1, adaptive_filter = self.adaptive_filter)
        # Train the model
        self.OutputTrainingPhase = model.fit(X, y)
        
        # Get rule-based structure
        self.parameters = model.parameters
        self.parameters_RLS = model.parameters_RLS
        # Get new lambda1
        self.lambda1 = model.lambda1
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = model.ResidualTrainingPhase
        # Control variables
        self.ymin = model.ymin
        self.ymax = model.ymax
        self.region = model.region
        self.last_y = model.last_y
    
    def is_numeric_and_finite(self, array):
        """
        Check if a NumPy array contains only numeric values and all values are finite.
    
        Parameters:
            array (numpy.ndarray): The array to check.
    
        Returns:
            bool: True if the array contains only numeric and finite values, False otherwise.
        """
        try:
            # Check if the data type of the array is numeric
            if not np.issubdtype(array.dtype, np.number):
                return False
            # Check if all elements are finite
            return np.isfinite(array).all()
        except TypeError:
            # If any element is not numeric, np.isfinite will raise a TypeError
            return False
        
    def initialize_rule(self, mean, y, std, is_first=False):
        Theta = np.insert(np.zeros(mean.shape[0]), 0, y)[:, None]
        if self.adaptive_filter == "RLS":
            rule_params = {
                'mean': mean,
                'std': std,
                'NumObservations': 1
            }

            if is_first:
                self.parameters = pd.DataFrame([rule_params])
                self.parameters_RLS['P'] = self.omega * np.eye(mean.shape[0] + 1)
                self.parameters_RLS['p_vector'] = np.zeros(Theta.shape)
                self.parameters_RLS['Theta'] = Theta
            else:
                self.parameters = pd.concat([self.parameters, pd.DataFrame([rule_params])], ignore_index=True)
        
        elif self.adaptive_filter == "wRLS":
            rule_params = {
                'mean': mean,
                'P': self.omega * np.eye(mean.shape[0] + 1),
                'p_vector': np.zeros(Theta.shape),
                'Theta': Theta,
                'NumObservations': 1,
                'weight': 0,
                'std': std
            }
            if is_first:
                self.parameters = pd.DataFrame([rule_params])
            else:
                self.parameters = pd.concat([self.parameters, pd.DataFrame([rule_params])], ignore_index=True)

    def Rule_Update(self, i):
        # Update the number of observations in the rule
        self.parameters.loc[i, 'NumObservations'] = self.parameters.loc[i, 'NumObservations'] + 1
            
    def Firing_Level(self, m, x, std):
        # Prevent division by zero by adding a small epsilon to std
        epsilon = 1e-10
        std = np.maximum(std, epsilon)  # Replace zero std with a small value to avoid division by zero
        
        return np.exp(-0.5 * ((m - x) ** 2) / (std ** 2))
    
    def tau(self, x):
        for row in self.parameters.index:
            if self.fuzzy_operator == "prod":
                tau = np.prod( self.Firing_Level(self.parameters.loc[row, 'mean'], x, self.parameters.loc[row, 'std'] ) )
            if self.fuzzy_operator == "max":
                tau = np.max( self.Firing_Level(self.parameters.loc[row, 'mean'], x, self.parameters.loc[row, 'std'] ) )
            if self.fuzzy_operator == "min":
                tau = np.min( self.Firing_Level(self.parameters.loc[row, 'mean'], x, self.parameters.loc[row, 'std'] ) )
            
            # Evoid mtau with values zero
            if abs(tau) < 1e-10:
                tau = 1e-10

            self.parameters.at[row, 'tau'] = tau
    
    def weight(self, x):
        self.tau(x)
        self.parameters['weight'] = self.parameters['weight'].astype(float)
        tau_sum = sum(self.parameters['tau'])
        if tau_sum == 0:
            tau_sum = 1 / self.parameters.shape[0]  # Small constant to avoid division by zero
        for row in self.parameters.index:
            self.parameters.at[row, 'weight'] = self.parameters.loc[row, 'tau'] / tau_sum
    
    def RLS(self, x, y, xe):
        """
        Conventional RLS algorithm
        Adaptive Filtering - Paulo S. R. Diniz
        
        Parameters:
            lambda: forgeting factor
    
        """
               
        lambda1 = 1. if self.lambda1 + xe.T @ self.parameters_RLS['P'] @ xe == 0 else self.lambda1
            
        # K is used here just to make easier to see the equation of the covariance matrix
        K = ( self.parameters_RLS['P'] @ xe ) / ( lambda1 + xe.T @ self.parameters_RLS['P'] @ xe )
        self.parameters_RLS['P'] = ( 1 / lambda1 ) * ( self.parameters_RLS['P'] - K @ xe.T @ self.parameters_RLS['P'] )
        self.parameters_RLS['Theta'] = self.parameters_RLS['Theta'] + ( self.parameters_RLS['P'] @ xe ) * (y - xe.T @ self.parameters_RLS['Theta'] )
            

    def wRLS(self, x, y, xe):
        """
        weighted Recursive Least Square (wRLS)
        An Approach to Online Identification of Takagi-Sugeno Fuzzy Models - Angelov and Filev

        """
        for row in self.parameters.index:
            self.parameters.at[row, 'P'] = self.parameters.loc[row, 'P'] - (( self.parameters.loc[row, 'weight'] * self.parameters.loc[row, 'P'] @ xe @ xe.T @ self.parameters.loc[row, 'P'])/(1 + self.parameters.loc[row, 'weight'] * xe.T @ self.parameters.loc[row, 'P'] @ xe))
            self.parameters.at[row, 'Theta'] = ( self.parameters.loc[row, 'Theta'] + (self.parameters.loc[row, 'P'] @ xe * self.parameters.loc[row, 'weight'] * (y - xe.T @ self.parameters.loc[row, 'Theta'])) )
        