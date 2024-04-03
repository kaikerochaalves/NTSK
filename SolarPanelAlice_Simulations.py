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
import pandas as pd

# Feature scaling
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Including to the path another fold
import sys
sys.path.append(r'ProposedModels')

# Import models
from NTSK import NTSK


#-----------------------------------------------------------------------------
# Import the time series
#-----------------------------------------------------------------------------

Serie = "Alice_59_Site_38_QCELLS_Power"

horizon = 1
    
# Importing the data
Data = pd.read_excel(f'Datasets/{Serie}.xlsx')

# Defining the atributes and the target value
X = Data[Data.columns[2:13]].values
y = Data[Data.columns[13]].values

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
# Executing the simulations for Alice time series
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# NTSK-RLS
# -----------------------------------------------------------------------------

Model = "NTSK-RLS"

# Set hyperparameters range
n_clusters = 6
lambda1 = 1
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

# Show the rules

print(f"\n\nRules for {Model}:\n")

for i in model.parameters.index:
    rule = f'{i+1}'
    for j in range(model.parameters.loc[i,'Center'].shape[0]):
        rule += f" & {model.parameters.loc[i,'Center'][j,0]:.2f} $\pm$ {model.parameters.loc[i,'sigma'][j,0]:.2f}"
    rule += f" & ({model.parameters.loc[i,'tangent'][0]:.2f},{model.parameters.loc[i,'tangent'][1]:.2f})"
    print(rule)

# -----------------------------------------------------------------------------
# NTSK-wRLS
# -----------------------------------------------------------------------------

Model = "NTSK-wRLS"

# Set hyperparameters range
n_clusters = 19
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

# Show the rules

print(f"\n\nRules for {Model}:\n")

for i in model.parameters.index:
    rule = f'{i+1}'
    for j in range(model.parameters.loc[i,'Center'].shape[0]):
        rule += f" & {model.parameters.loc[i,'Center'][j,0]:.2f} $\pm$ {model.parameters.loc[i,'sigma'][j,0]:.2f}"
    rule += f" & ({model.parameters.loc[i,'tangent'][0]:.2f},{model.parameters.loc[i,'tangent'][1]:.2f})"
    print(rule)

#-----------------------------------------------------------------------------
# Import the time series
#-----------------------------------------------------------------------------

Serie = "Alice_91_Site_1A_Trina_Power"

horizon = 1
    
# Importing the data
Data = pd.read_excel(f'Datasets/{Serie}.xlsx')

# Defining the atributes and the target value
X = Data[Data.columns[2:13]].values
y = Data[Data.columns[13]].values

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
# Executing the Grid-Search for the TAIEX time series
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# NTSK-RLS
# -----------------------------------------------------------------------------

Model = "NTSK-RLS"

# Set hyperparameters range
n_clusters = 6
lambda1 = 1
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

# Show the rules

print(f"\n\nRules for {Model}:\n")

for i in model.parameters.index:
    rule = f'{i+1}'
    for j in range(model.parameters.loc[i,'Center'].shape[0]):
        rule += f" & {model.parameters.loc[i,'Center'][j,0]:.2f} $\pm$ {model.parameters.loc[i,'sigma'][j,0]:.2f}"
    rule += f" & ({model.parameters.loc[i,'tangent'][0]:.2f},{model.parameters.loc[i,'tangent'][1]:.2f})"
    print(rule)

# -----------------------------------------------------------------------------
# NTSK-wRLS
# -----------------------------------------------------------------------------

Model = "NTSK-wRLS"

# Set hyperparameters range
n_clusters = 1
RLS_option = 2


# Initialize the model
model = NTSK(n_clusters = n_clusters, RLS_option = RLS_option)
# Train the model
OutputTraining = model.fit(X_train, y_train)
# Test the model
y_pred2 = model.predict(X_test)

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred2))
print("RMSE:", RMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred2)
print("MAE:", MAE)
# Compute the number of final rules
Rules = n_clusters
print("Rules:", Rules)

# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.plot(y_pred2, linewidth = 5, color = 'blue', label = 'Predicted value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/{Model}_{Serie}.eps', format='eps', dpi=1200)
plt.show()

# Show the rules

print(f"\n\nRules for {Model}:\n")

for i in model.parameters.index:
    rule = f'{i+1}'
    for j in range(model.parameters.loc[i,'Center'].shape[0]):
        rule += f" & {model.parameters.loc[i,'Center'][j,0]:.2f} $\pm$ {model.parameters.loc[i,'sigma'][j,0]:.2f}"
    rule += f" & ({model.parameters.loc[i,'tangent'][0]:.2f},{model.parameters.loc[i,'tangent'][1]:.2f})"
    print(rule)

