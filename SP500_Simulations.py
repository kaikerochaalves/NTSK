# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 21:52:41 2023

@author: Kaike Alves
"""

# Import libraries
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import statistics as st
import matplotlib.pyplot as plt

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

Serie = "SP500"

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
# Executing the Grid-Search for the SP500 time series
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# NTSK-RLS
# -----------------------------------------------------------------------------

Model = "NTSK-RLS"

# Set hyperparameters range
n_clusters = 10
lambda1 = 0.95
RLS_option = 1


# Initialize the model
model = NTSK(n_clusters = n_clusters, lambda1 = lambda1, RLS_option = RLS_option)
# Train the model
OutputTraining = model.fit(X_train, y_train)
# Test the model
y_pred1 = model.predict(X_test)

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred1))
print("RMSE:", RMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred1)
print("MAE:", MAE)
# Compute the number of final rules
Rules = n_clusters
print("Rules:", Rules)

# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.plot(y_pred1, linewidth = 5, color = 'blue', label = 'Predicted value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/{Model}_{Serie}.eps', format='eps', dpi=1200)
plt.show()

# -----------------------------------------------------------------------------
# NTSK-wRLS
# -----------------------------------------------------------------------------

Model = "NTSK-wRLS"

# Set hyperparameters range
n_clusters = 4
RLS_option = 2


# Initialize the model
model = NTSK(n_clusters = n_clusters, RLS_option = RLS_option)
# Train the model
OutputTraining = model.fit(X_train, y_train)
# Test the model
y_pred1 = model.predict(X_test)

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred1))
print("RMSE:", RMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred1)
print("MAE:", MAE)
# Compute the number of final rules
Rules = n_clusters
print("Rules:", Rules)

# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.plot(y_pred1, linewidth = 5, color = 'blue', label = 'Predicted value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/{Model}_{Serie}.eps', format='eps', dpi=1200)
plt.show()
