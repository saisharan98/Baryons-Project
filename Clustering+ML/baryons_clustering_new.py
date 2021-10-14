each import numpy as np
import pandas as pd
from get_dataset import get_data
from ILC_square_v2 import get_ILC_weights_map
from clustering_supervised import *
from flatten_custom import flatten_custom
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time
import pickle
import seaborn as sns

X_orig, y_orig = get_data(np.arange(100))
total_indices = np.arange(100)
test_indices, train_indices = train_test_split(total_indices, test_size=0.75, random_state=42)
X_train, y_train = get_data(train_indices)
X_test, y_test = get_data(test_indices)

X_flat, y_flat = flatten_custom(X_orig, y_orig)
X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.2, random_state=0)
X_flat_train, y_flat_train = flatten_custom(X_train,y_train)
X_flat_test, y_flat_test = flatten_custom(X_test,y_test)

tic = time.time()
kmeans3 = KMeans(n_clusters=3, random_state=0).fit(X_flat_train) 
toc = time.time()
kmeans3_time = toc - tic
print("Time taken for 3 clusters:", np.round(kmeans3_time,3),"secs")
pkl_filename = "kmeans3.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(kmeans3, file)
    
tic = time.time()
kmeans4 = KMeans(n_clusters=4, random_state=0).fit(X_flat_train) 
toc = time.time()
kmeans4_time = toc - tic
print("Time taken for 4 clusters:", np.round(kmeans4_time,3),"secs")
pkl_filename = "kmeans4.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(kmeans4, file)

tic = time.time()
kmeans5 = KMeans(n_clusters=5, random_state=0).fit(X_flat_train) 
toc = time.time()
kmeans5_time = toc - tic
print("Time taken for 5 clusters:", np.round(kmeans5_time,3),"secs")
pkl_filename = "kmeans5.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(kmeans5, file)
    
tic = time.time()
kmeans6 = KMeans(n_clusters=6, random_state=0).fit(X_flat_train) 
toc = time.time()
kmeans6_time = toc - tic
print("Time taken for 6 clusters:", np.round(kmeans6_time,3),"secs")
pkl_filename = "kmeans6.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(kmeans6, file)
    
    
    
# with open("kmeans3.pkl",'rb') as file:
#     kmeans3 = pickle.load(file)
# with open("kmeans4.pkl",'rb') as file:
#     kmeans4 = pickle.load(file)
# with open("kmeans5.pkl",'rb') as file:
#     kmeans5 = pickle.load(file)
# with open("kmeans6.pkl",'rb') as file:
#     kmeans6 = pickle.load(file)
    
X_test = get_data(np.array([4]))
X_test = X_test[0]
X_to_predict = np.vstack([X_test[0][0].flatten(), X_test[0][1].flatten(), X_test[0][2].flatten(), X_test[0][3].flatten(), X_test[0][4].flatten(), X_test[0][5].flatten()]).T

predicted_labels3 = kmeans3.predict(X_to_predict)
image1 = np.zeros((301,301))
print("Weights for 3 clusters:-")
for i in np.unique(predicted_labels3):
    index = np.where(predicted_labels3==i)
    if np.size(index)>0:
        w,image = get_ILC_specific_indices(X_orig,4, index)
        image1 = image1+image
print()
plt.figure(1)
plt.title("ILC constructed image for 3 clusters")
plt.imshow(image1)
plt.colorbar()

print("Weights for 4 clusters:-")
predicted_labels4 = kmeans4.predict(X_to_predict)
image2 = np.zeros((301,301))
for i in np.unique(predicted_labels4):
    index = np.where(predicted_labels4==i)
    if np.size(index)>0:
        w,image = get_ILC_specific_indices(X_orig,4, index)
        image2 = image2+image
print()
        
plt.figure(2)
plt.title("ILC constructed image for 4 clusters")
plt.imshow(image2)
plt.colorbar()

print("Weights for 5 clusters:-")
predicted_labels5 = kmeans5.predict(X_to_predict)
image3 = np.zeros((301,301))
for i in np.unique(predicted_labels5):
    index = np.where(predicted_labels5==i)
    if np.size(index)>0:
        w,image = get_ILC_specific_indices(X_orig,4, index)
        image3 = image3+image
print()
plt.figure(3)
plt.title("ILC constructed image for 5 clusters")
plt.imshow(image3)
plt.colorbar()

print("Weights for 6 clusters:-")
predicted_labels6 = kmeans6.predict(X_to_predict)
image4 = np.zeros((301,301))
for i in np.unique(predicted_labels6):
    index = np.where(predicted_labels6==i)
    if np.size(index)>0:
        w,image = get_ILC_specific_indices(X_orig,4, index)
        image4 = image4+image
print()
plt.figure(4)
plt.title("ILC constructed image for 6 clusters")
plt.imshow(image4)
plt.colorbar()


img_avg = (image1 + image2 + image3 + image4) / 4
plt.figure(5)
plt.title("Average image")
plt.imshow(img_avg)
pkl_filename = "average_image.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(img_avg, file)