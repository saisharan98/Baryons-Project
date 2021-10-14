# importing os module 
import os 
from astropy.io import fits
import numpy as np

Filelist = []


# Function to rename multiple files 
for count, filename in enumerate(os.listdir("data/")): 
	Filelist.append(filename)


#Get files
#The function returns the files in form of X, y for the particular indices. 
#Could be used to get files for test_set and train set just by passing the indices.
def get_data(indices_to_return,pxsize=301,pysize=301):
	X = np.zeros((np.size(indices_to_return), 6, pxsize, pysize))
	y = np.zeros((np.size(indices_to_return), 1, pxsize, pysize))
	j=0
	for i in indices_to_return:
		data_dir = 'data/'
		data = fits.getdata(data_dir+Filelist[i])
		X[j,:,:,:] = data[0:6]
		y[j,:,:,:] = data[6]-np.mean(data[6])
		j = j+1
	return X, y
