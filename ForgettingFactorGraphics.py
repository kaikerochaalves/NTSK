# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 21:37:10 2024

@author: Kaike Alves
"""

# Import libraries
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import statistics as st
import matplotlib.pyplot as plt
import numpy as np

# Feature scaling
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Including to the path another fold
import sys
sys.path.append(r'ProposedModel')

# Import models
from NTSK import NTSK


# Import the series
import yfinance as yf

#-----------------------------------------------------------------------------
# Import the time series
#-----------------------------------------------------------------------------

Serie = "S&P 500"

horizon = 1
    
# Importing the data
SP500 = yf.Ticker('^GSPC')
SP500 = SP500.history(start = "2020-01-01", end = "2022-01-01", interval='1d')

# Prepare the data
columns = SP500.columns
Data = SP500[columns[:4]]

# Add the target column value
NextClose = Data.iloc[horizon:,-1].values
Data = Data.drop(Data.index[-horizon:])
Data['NextClose'] = NextClose

# Convert to array
X = Data[Data.columns[:-1]].values
y = Data[Data.columns[-1]].values

# Spliting the data into train and test
n = Data.shape[0]
training_size = round(n*0.8)
X_train, X_test = X[:training_size,:], X[training_size:,:]
y_train, y_test = y[:training_size], y[training_size:]

# Min-max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_test = scaler.transform(X_test)




# -----------------------------------------------------------------------------
# NTSK-wRLS
# -----------------------------------------------------------------------------



Model = "NTSK-RLS"

# Set hyperparameters range
l_lambda1 = np.arange(0.8,1,0.01)
adaptive_filter = "RLS"

simul = 0
lambdaRLS1 = np.array([])
rmseRLS1 = np.array([])
for lambda1 in l_lambda1:
        
    # Initialize the model
    model = NTSK(rules = 1, lambda1 = lambda1, adaptive_filter = adaptive_filter)
    # Train the model
    OutputTraining = model.fit(X_train, y_train)
    # Test the model
    y_pred1 = model.predict(X_test)
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test, y_pred1))
    # Compute the Mean Absolute Error
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(y_test.flatten())
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test, y_pred1)

        
    simul = simul + 1
    #print(f'Simulação: {simul}')
    print('.', end='', flush=True)
    
    lambdaRLS1 = np.append(lambdaRLS1, lambda1)
    rmseRLS1 = np.append(rmseRLS1, RMSE)


# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(lambdaRLS1, rmseRLS1, linewidth = 5, color = 'red', label = Serie)
plt.ylabel('RMSE')
plt.xlabel('Forgetting Factor')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/RMSE_ForgettingFactor_{Model}_{Serie}.eps', format='eps', dpi=1200)
plt.show()

#-----------------------------------------------------------------------------
# Import the time series
#-----------------------------------------------------------------------------

Serie = "NASDAQ"

horizon = 1
    
# Importing the data
NASDAQ = yf.Ticker('^IXIC')
NASDAQ = NASDAQ.history(start = "2020-01-01", end = "2022-01-01", interval='1d')

# Prepare the data
columns = NASDAQ.columns
Data = NASDAQ[columns[:4]]

# Add the target column value
NextClose = Data.iloc[horizon:,-1].values
Data = Data.drop(Data.index[-horizon:])
Data['NextClose'] = NextClose

# Convert to array
X = Data[Data.columns[:-1]].values
y = Data[Data.columns[-1]].values

# Spliting the data into train and test
n = Data.shape[0]
training_size = round(n*0.8)
X_train, X_test = X[:training_size,:], X[training_size:,:]
y_train, y_test = y[:training_size], y[training_size:]

# Min-max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_test = scaler.transform(X_test)




# -----------------------------------------------------------------------------
# NTSK-wRLS
# -----------------------------------------------------------------------------



Model = "NTSK-RLS"

# Set hyperparameters range
l_lambda1 = np.arange(0.8,1,0.01)
adaptive_filter = "RLS"

simul = 0
lambdaRLS2 = np.array([])
rmseRLS2 = np.array([])
for lambda1 in l_lambda1:
        
    # Initialize the model
    model = NTSK(rules = 1, lambda1 = lambda1, adaptive_filter = adaptive_filter)
    # Train the model
    OutputTraining = model.fit(X_train, y_train)
    # Test the model
    y_pred1 = model.predict(X_test)
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test, y_pred1))
    # Compute the Mean Absolute Error
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(y_test.flatten())
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test, y_pred1)

        
    simul = simul + 1
    #print(f'Simulação: {simul}')
    print('.', end='', flush=True)
    
    lambdaRLS2 = np.append(lambdaRLS2, lambda1)
    rmseRLS2 = np.append(rmseRLS2, RMSE)


# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(lambdaRLS2, rmseRLS2, linewidth = 5, color = 'red', label = Serie)
plt.ylabel('RMSE')
plt.xlabel('Forgetting Factor')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/RMSE_ForgettingFactor_{Model}_{Serie}.eps', format='eps', dpi=1200)
plt.show()




#-----------------------------------------------------------------------------
# Import the time series
#-----------------------------------------------------------------------------

Serie = "TAIEX"

horizon = 1
    
# Importing the data
TAIEX = yf.Ticker('^TWII')
TAIEX = TAIEX.history(start = "2020-01-01", end = "2022-01-01", interval='1d')

# Prepare the data
columns = TAIEX.columns
Data = TAIEX[columns[:4]]

# Add the target column value
NextClose = Data.iloc[horizon:,-1].values
Data = Data.drop(Data.index[-horizon:])
Data['NextClose'] = NextClose

# Convert to array
X = Data[Data.columns[:-1]].values
y = Data[Data.columns[-1]].values

# Spliting the data into train and test
n = Data.shape[0]
training_size = round(n*0.8)
X_train, X_test = X[:training_size,:], X[training_size:,:]
y_train, y_test = y[:training_size], y[training_size:]

# Min-max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_test = scaler.transform(X_test)



# -----------------------------------------------------------------------------
# NTSK-wRLS
# -----------------------------------------------------------------------------



Model = "NTSK-RLS"

# Set hyperparameters range
l_lambda1 = np.arange(0.8,1,0.01)
adaptive_filter = "RLS"

simul = 0
lambdaRLS3 = np.array([])
rmseRLS3 = np.array([])
for lambda1 in l_lambda1:
        
    # Initialize the model
    model = NTSK(rules = 1, lambda1 = lambda1, adaptive_filter = adaptive_filter)
    # Train the model
    OutputTraining = model.fit(X_train, y_train)
    # Test the model
    y_pred1 = model.predict(X_test)
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test, y_pred1))
    # Compute the Mean Absolute Error
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(y_test.flatten())
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test, y_pred1)

        
    simul = simul + 1
    #print(f'Simulação: {simul}')
    print('.', end='', flush=True)
    
    lambdaRLS3 = np.append(lambdaRLS3, lambda1)
    rmseRLS3 = np.append(rmseRLS3, RMSE)


# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(lambdaRLS3, rmseRLS3, linewidth = 5, color = 'red', label = Serie)
plt.ylabel('RMSE')
plt.xlabel('Forgetting Factor')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/RMSE_ForgettingFactor_{Model}_{Serie}.eps', format='eps', dpi=1200)
plt.show()



# -----------------------------------------------------------------------------
# NTSK-wRLS All Series
# -----------------------------------------------------------------------------


# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(lambdaRLS1, rmseRLS1, linewidth = 5, label = 'S&P 500')
plt.plot(lambdaRLS2, rmseRLS2, linewidth = 5, label = 'NASDAQ')
plt.plot(lambdaRLS3, rmseRLS3, linewidth = 5, label = 'TAIEX')
plt.ylabel('RMSE')
plt.xlabel('Number of Rules')
plt.legend(loc='upper left')
plt.savefig('Graphics/RMSE_ForgettingFactor_AllModels_NTSKwRLS.eps', format='eps', dpi=1200)
plt.show()