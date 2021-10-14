import numpy as np
import matplotlib.pyplot as plt
from get_dataset import *
from ILC_square_v2 import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# Finding weights for vectors using a simple regression
def regression_weights_image(X_train, y_train, X_test):
    
    
    X = X_train
    y = y_train
    
    X_flattened_1 = []
    X_flattened_2 = []
    X_flattened_3 = []
    X_flattened_4 = []
    X_flattened_5 = []
    X_flattened_6 = []
    y_flattened = []
    
    
    
    for i in range(len(X_test)):
        X_flattened_1 = np.append( X_flattened_1, X[i,0].flatten())
        X_flattened_2 = np.append( X_flattened_2, X[i,1].flatten())
        X_flattened_3 = np.append( X_flattened_3, X[i,2].flatten())
        X_flattened_4 = np.append( X_flattened_4, X[i,3].flatten())
        X_flattened_5 = np.append( X_flattened_5, X[i,4].flatten())
        X_flattened_6 = np.append( X_flattened_6, X[i,5].flatten())

        y_flattened = np.append(y_flattened, y[0].flatten())
    X_flattened = np.vstack([X_flattened_1, X_flattened_2, X_flattened_3, X_flattened_4, X_flattened_5, X_flattened_6]).T
    reg = LinearRegression().fit(X_flattened, y_flattened)
    test_map_output = reg.predict(np.vstack([X_test[0].flatten(), X_test[1].flatten(), X_test[2].flatten(), X_test[3].flatten(), X_test[4].flatten(), X_test[5].flatten()]).T)
    return (reg.coef_,test_map_output.reshape((301,301)))


# Using the Random forest regressor, i.e. different wights to different patches.
def regression_with_RF(X_train, y_train, X_test):
    
    X = X_train
    y = y_train
    
    X_flattened_1 = []
    X_flattened_2 = []
    X_flattened_3 = []
    X_flattened_4 = []
    X_flattened_5 = []
    X_flattened_6 = []
    y_flattened = []
    
    
    
    for i in range(len(X_test)):
        X_flattened_1 = np.append( X_flattened_1, X[i,0].flatten())
        X_flattened_2 = np.append( X_flattened_2, X[i,1].flatten())
        X_flattened_3 = np.append( X_flattened_3, X[i,2].flatten())
        X_flattened_4 = np.append( X_flattened_4, X[i,3].flatten())
        X_flattened_5 = np.append( X_flattened_5, X[i,4].flatten())
        X_flattened_6 = np.append( X_flattened_6, X[i,5].flatten())

        y_flattened = np.append(y_flattened, y[0].flatten())
    X_flattened = np.vstack([X_flattened_1, X_flattened_2, X_flattened_3, X_flattened_4, X_flattened_5, X_flattened_6]).T
    reg = RandomForestRegressor().fit(X_flattened, y_flattened)
    test_map_output = reg.predict(np.vstack([X_test[0].flatten(), X_test[1].flatten(), X_test[2].flatten(), X_test[3].flatten(), X_test[4].flatten(), X_test[5].flatten()]).T)
    return (test_map_output.reshape((301,301)))


