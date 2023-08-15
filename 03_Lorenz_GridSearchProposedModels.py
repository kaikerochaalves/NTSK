# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 21:52:41 2023

@author: Kaike Alves
"""

# Import libraries
import pandas as pd
import numpy as np
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

# Including to the path another fold
sys.path.append(r'Functions')
# Import the serie generator
from LorenzAttractorGenerator import Lorenz

#-----------------------------------------------------------------------------
# Generate the time series
#-----------------------------------------------------------------------------

Serie = "Lorenz"

# Input parameters
x0 = 0.
y0 = 1.
z0 = 1.05
sigma = 10
beta = 2.667
rho=28
num_steps = 10000

# Creating the Lorenz Time Series
x, y, z = Lorenz(x0 = x0, y0 = y0, z0 = z0, sigma = sigma, beta = beta, rho = rho, num_steps = num_steps)

# Ploting the graphic
plt.rc('font', size=10)
plt.rc('axes', titlesize=15)
ax = plt.figure().add_subplot(projection='3d')
ax.plot(x, y, z, lw = 0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()

def Create_Leg(data, ncols, leg, leg_output = None):
    X = np.array(data[leg*(ncols-1):].reshape(-1,1))
    for i in range(ncols-2,-1,-1):
        X = np.append(X, data[leg*i:leg*i+X.shape[0]].reshape(-1,1), axis = 1)
    X_new = np.array(X[:,-1].reshape(-1,1))
    for col in range(ncols-2,-1,-1):
        X_new = np.append(X_new, X[:,col].reshape(-1,1), axis=1)
    if leg_output == None:
        return X_new
    else:
        y = np.array(data[leg*(ncols-1)+leg_output:].reshape(-1,1))
        return X_new[:y.shape[0],:], y

# Defining the atributes and the target value
X = np.concatenate([x[:-1].reshape(-1,1), y[:-1].reshape(-1,1), z[:-1].reshape(-1,1)], axis = 1)
y = x[1:].reshape(-1,1)

# Spliting the data into train and test
X_train, X_test = X[:8000,:], X[8000:,:]
y_train, y_test = y[:8000,:], y[8000:,:]

# Min-max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_test = scaler.transform(X_test)

# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.show()

# -----------------------------------------------------------------------------
# Executing the Grid-Search for the Lorenz time series
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# NTSK-RLS
# -----------------------------------------------------------------------------

Model = "NTSK-RLS"

# Set hyperparameters range
l_n_clusters = range(1,20,3)
l_lambda = [0.8, 0.9, 0.95, 0.99]
RLS_option = 1

simul = 0
# Creating the DataFrame to store results
columns = ['Model', 'n_clusters', 'lambda1', 'RLS_option', 'RMSE', 'NDEI', 'MAE', 'Rules']
result = pd.DataFrame(columns = columns)
for n_clusters in l_n_clusters:
    for lambda1 in l_lambda:
        
        # Initialize the model
        model = NTSK(n_clusters = n_clusters, lambda1 = lambda1, RLS_option = RLS_option)
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
        # Compute the number of final rules
        Rules = n_clusters
            
        simul = simul + 1
        #print(f'Simulação: {simul}')
        print('.', end='', flush=True)
        
        NewRow = pd.DataFrame([[Model, n_clusters, lambda1, RLS_option, RMSE, NDEI, MAE, Rules]], columns = columns)
        result = pd.concat([result, NewRow], ignore_index=True)
        
name = f'GridSearchResults\Hyperparameters_{Model}_{Serie}.xlsx'
result.to_excel(name)

# Simulating with best hyperparameters
row = result['RMSE'].astype(float).idxmin()
n = result.shape[1]
hp1 = result.loc[row,result.columns[1:n-4]]

# Initializing the model
model = NTSK(**hp1)
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
Rules = hp1.n_clusters


NTSK_RLS = f'{Model} & {RMSE:.5f} & {NDEI:.5f} & {MAE:.5f} & {Rules}'

print(f"\n{hp1}")

print(f"\n{NTSK_RLS}")

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
# NTSK-wRLS"
# -----------------------------------------------------------------------------

Model = "NTSK-wRLS"

# Set hyperparameters range
l_n_clusters = range(1,20,3)
RLS_option = 2

simul = 0
# Creating the DataFrame to store results
columns = ['Model', 'n_clusters', 'RLS_option', 'RMSE', 'NDEI', 'MAE', 'Rules']
result = pd.DataFrame(columns = columns)
for n_clusters in l_n_clusters:
        
    # Initialize the model
    model = NTSK(n_clusters = n_clusters, RLS_option = RLS_option)
    # Train the model
    OutputTraining = model.fit(X_train, y_train)
    # Test the model
    y_pred2 = model.predict(X_test)
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test, y_pred2))
    # Compute the Mean Absolute Error
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(y_test.flatten())
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test, y_pred2)
    # Compute the number of final rules
    Rules = n_clusters
        
    simul = simul + 1
    #print(f'Simulação: {simul}')
    print('.', end='', flush=True)
    
    NewRow = pd.DataFrame([[Model, n_clusters, RLS_option, RMSE, NDEI, MAE, Rules]], columns = columns)
    result = pd.concat([result, NewRow], ignore_index=True)
        
name = f'GridSearchResults\Hyperparameters_{Model}_{Serie}.xlsx'
result.to_excel(name)

# Simulating with best hyperparameters
row = result['RMSE'].astype(float).idxmin()
n = result.shape[1]
hp2 = result.loc[row,result.columns[1:n-4]]

# Initializing the model
model = NTSK(**hp2)
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
Rules = hp2.n_clusters


NTSK_wRLS = f'{Model} & {RMSE:.5f} & {NDEI:.5f} & {MAE:.5f} & {Rules}'

print(f"\n{hp2}")

print(f"\n{NTSK_wRLS}")

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


#-----------------------------------------------------------------------------
# Graphics NTSK-RLS and NTSK-wRLS
#-----------------------------------------------------------------------------

# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.plot(y_pred1, linewidth = 3, color = 'blue', label = 'NTSK-RLS', linestyle = "--")
plt.plot(y_pred2, linewidth = 3, color = 'black', label = 'NTSK-wRLS', linestyle = "-.")
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/2Models_{Serie}.eps', format='eps', dpi=1200)
plt.show()



#-----------------------------------------------------------------------------
# Print results
#-----------------------------------------------------------------------------

print(f"\n\n{NTSK_RLS}")
print(f"\n{NTSK_wRLS}")
