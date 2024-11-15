
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 21:52:41 2023

@author: Kaike Alves
"""

# Import libraries
import pandas as pd
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
sys.path.append(r'ProposedModels')

# Import models
from NTSK import NTSK


#-----------------------------------------------------------------------------
# Import the time series
#-----------------------------------------------------------------------------

l_Serie = ['Alice_59_Site_38_QCELLS_Power', 'Alice_91_Site_1A_Trina_Power']

for Serie in l_Serie:

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
    # Executing the Grid-Search for the Alice time series
    # -----------------------------------------------------------------------------
    
    # -----------------------------------------------------------------------------
    # NTSK-RLS
    # -----------------------------------------------------------------------------
    
    Model = "NTSK-RLS"
    
    # Set hyperparameters range
    rules = 1
    l_lambda = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.]
    adaptive_filter = "RLS"

    simul = 0
    # Creating the DataFrame to store results
    columns = ['Model', 'rules', 'lambda1', 'adaptive_filter', 'RMSE', 'NDEI', 'MAE', 'Rules']
    result = pd.DataFrame(columns = columns)
    for lambda1 in l_lambda:
        
        # Initialize the model
        model = NTSK(rules = rules, lambda1 = lambda1, adaptive_filter = adaptive_filter)
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
        Rules = rules
            
        simul = simul + 1
        #print(f'Simulação: {simul}')
        print('.', end='', flush=True)
        
        NewRow = pd.DataFrame([[Model, rules, lambda1, adaptive_filter, RMSE, NDEI, MAE, Rules]], columns = columns)
        if result.shape[0] == 0:
            result = NewRow
        else:  
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
    Rules = hp1.rules
    
    
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
    l_rules = range(1,20,3)
    adaptive_filter = "wRLS"
    
    simul = 0
    # Creating the DataFrame to store results
    columns = ['Model', 'rules', 'adaptive_filter', 'RMSE', 'NDEI', 'MAE', 'Rules']
    result = pd.DataFrame(columns = columns)
    for rules in l_rules:
            
        # Initialize the model
        model = NTSK(rules = rules, adaptive_filter = adaptive_filter)
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
        Rules = rules
            
        simul = simul + 1
        #print(f'Simulação: {simul}')
        print('.', end='', flush=True)
        
        NewRow = pd.DataFrame([[Model, rules, adaptive_filter, RMSE, NDEI, MAE, Rules]], columns = columns)
        if result.shape[0] == 0:
            result = NewRow
        else:  
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
    Rules = hp2.rules
    
    
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
