import numpy as np
def flatten_custom(X, y):
    
    
    
    X_flattened_1 = []
    X_flattened_2 = []
    X_flattened_3 = []
    X_flattened_4 = []
    X_flattened_5 = []
    X_flattened_6 = []
    y_flattened = []
    
    
    
    for i in range(len(X)):
        X_flattened_1 = np.append( X_flattened_1, X[i,0].flatten())
        X_flattened_2 = np.append( X_flattened_2, X[i,1].flatten())
        X_flattened_3 = np.append( X_flattened_3, X[i,2].flatten())
        X_flattened_4 = np.append( X_flattened_4, X[i,3].flatten())
        X_flattened_5 = np.append( X_flattened_5, X[i,4].flatten())
        X_flattened_6 = np.append( X_flattened_6, X[i,5].flatten())

        y_flattened = np.append(y_flattened, y[0].flatten())
    X_flattened = np.vstack([X_flattened_1, X_flattened_2, X_flattened_3, X_flattened_4, X_flattened_5, X_flattened_6]).T
    
    return X_flattened, y_flattened