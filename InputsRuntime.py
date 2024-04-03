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
import time

# Feature scaling
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Including to the path another fold
import sys
sys.path.append(r'ProposedModels')

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



Model = "NTSK-wRLS"

# Set hyperparameters range
n_clusters = 4
RLS_option = 2

simul = 0
inputswRLS1 = np.array([])
runtimewRLS1 = np.array([])
for inputs in list(range(2,100,2)):
    
    # Spliting the data into train and test
    n = int(Data.shape[0] * inputs/100)
    training_size = round(n*0.8)
    X_train, X_test = X[:training_size,:], X[training_size:n,:]
    y_train, y_test = y[:training_size], y[training_size:n]

    # Min-max scaling
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train, y_train)
    X_test = scaler.transform(X_test)
    
    start = time.time()
    # Initialize the model
    model = NTSK(n_clusters = n_clusters, RLS_option = RLS_option)
    # Train the model
    OutputTraining = model.fit(X_train, y_train)
    # Test the model
    y_pred1 = model.predict(X_test)
    end = time.time()
    runtime = end - start
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test, y_pred1))
    # Compute the Mean Absolute Error
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(y_test.flatten())
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test, y_pred1)
    # Compute the number of final rules
    Rules = n_clusters
        
    simul = simul + 1
    #print(f'Simulação: {simul}')
    print('.', end='', flush=True)
    
    inputswRLS1 = np.append(inputswRLS1, n*4)
    runtimewRLS1 = np.append(runtimewRLS1, runtime)
    


# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(inputswRLS1, runtimewRLS1, linewidth = 5, color = 'red', label = Serie)
plt.ylabel('Runtime (s)')
plt.xlabel('Number of Inputs')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/Inputs_Runtime_{Model}_{Serie}.eps', format='eps', dpi=1200)
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



Model = "NTSK-wRLS"

# Set hyperparameters range
n_clusters = 4
RLS_option = 2

simul = 0
inputswRLS2 = np.array([])
runtimewRLS2 = np.array([])
for inputs in list(range(2,100,2)):
    
    # Spliting the data into train and test
    n = int(Data.shape[0] * inputs/100)
    training_size = round(n*0.8)
    X_train, X_test = X[:training_size,:], X[training_size:n,:]
    y_train, y_test = y[:training_size], y[training_size:n]

    # Min-max scaling
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train, y_train)
    X_test = scaler.transform(X_test)
    
    start = time.time()
    # Initialize the model
    model = NTSK(n_clusters = n_clusters, RLS_option = RLS_option)
    # Train the model
    OutputTraining = model.fit(X_train, y_train)
    # Test the model
    y_pred1 = model.predict(X_test)
    end = time.time()
    runtime = end - start
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test, y_pred1))
    # Compute the Mean Absolute Error
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(y_test.flatten())
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test, y_pred1)
    # Compute the number of final rules
    Rules = n_clusters
        
    simul = simul + 1
    #print(f'Simulação: {simul}')
    print('.', end='', flush=True)
    
    inputswRLS2 = np.append(inputswRLS2, n*4)
    runtimewRLS2 = np.append(runtimewRLS2, runtime)
    


# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(inputswRLS2, runtimewRLS2, linewidth = 5, color = 'red', label = Serie)
plt.ylabel('Runtime (s)')
plt.xlabel('Number of Inputs')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/Inputs_Runtime_{Model}_{Serie}.eps', format='eps', dpi=1200)
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



Model = "NTSK-wRLS"

# Set hyperparameters range
n_clusters = 4
RLS_option = 2

simul = 0
inputswRLS3 = np.array([])
runtimewRLS3 = np.array([])
for inputs in list(range(2,100,2)):
    
    # Spliting the data into train and test
    n = int(Data.shape[0] * inputs/100)
    training_size = round(n*0.8)
    X_train, X_test = X[:training_size,:], X[training_size:n,:]
    y_train, y_test = y[:training_size], y[training_size:n]

    # Min-max scaling
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train, y_train)
    X_test = scaler.transform(X_test)
    
    start = time.time()
    # Initialize the model
    model = NTSK(n_clusters = n_clusters, RLS_option = RLS_option)
    # Train the model
    OutputTraining = model.fit(X_train, y_train)
    # Test the model
    y_pred1 = model.predict(X_test)
    end = time.time()
    runtime = end - start
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test, y_pred1))
    # Compute the Mean Absolute Error
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(y_test.flatten())
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test, y_pred1)
    # Compute the number of final rules
    Rules = n_clusters
        
    simul = simul + 1
    #print(f'Simulação: {simul}')
    print('.', end='', flush=True)
    
    inputswRLS3 = np.append(inputswRLS3, n*4)
    runtimewRLS3 = np.append(runtimewRLS3, runtime)
    


# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(inputswRLS3, runtimewRLS3, linewidth = 5, color = 'red', label = Serie)
plt.ylabel('Runtime (s)')
plt.xlabel('Number of Inputs')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/Inputs_Runtime_{Model}_{Serie}.eps', format='eps', dpi=1200)
plt.show()




# -----------------------------------------------------------------------------
# NTSK-wRLS All Series
# -----------------------------------------------------------------------------


# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(inputswRLS1, runtimewRLS1, linewidth = 5, label = 'S&P 500')
plt.plot(inputswRLS2, runtimewRLS2, linewidth = 5, label = 'NASDAQ')
plt.plot(inputswRLS3, runtimewRLS3, linewidth = 5, label = 'TAIEX')
plt.ylabel('Runtime (s)')
plt.xlabel('Number of Inputs')
plt.legend(loc='upper left')
plt.savefig('Graphics/Inputs_Runtime_AllModels_NTSKwRLS.eps', format='eps', dpi=1200)
plt.show()