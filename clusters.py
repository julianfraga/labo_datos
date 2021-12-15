# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 17:26:10 2021

@author: Julián

This script clusterizes a data set with K-means via Principal Component Analysis (PCA)
The data set is also separated into a train-test split for further analysis
In particular, this was used to categorize songs based on Spotify's metadata 
(numeric values for subjective and technical aspects fo each song) 
going from more than a dozen features to only four.
"""
import matplotlib.pyplot as plt 
plt.rcParams.update({'font.size': 22})
import numpy as np 
import pandas as pd 
import seaborn as sns

from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans   

import warnings
import os
warnings.filterwarnings('ignore')
#%%

# Dataset load
os.chdir(r'D:\Documentos\UBA\LaboDatos2021\Final')
filename = 'tracks.csv'
d = pd.read_csv(filename)
# Features in the dataset
print(d.keys()) 
#%%

# Selection of desired features and deletion of empty entries
features=['popularity','duration_ms','explicit','danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo']
X=d[features].dropna().copy() 
X.head()
X = X.values
#%%

# PCA analysis + train-test split. Set number of components and train values
components=2
valores_train=500000

pca = PCA(n_components=components)

# Fitting
valores_test=int((len(X[:,0])-valores_train))

X_train=X[:valores_train,:]
X_train= X_train.reshape(valores_train, -1)

X_test=X[valores_train:,:]
X_test= X_test.reshape(valores_test, -1)

X_train = X_train.astype('float32') 
X_test = X_test.astype('float32') 

# Scaling
scaler = MinMaxScaler()
scaler.fit(X_test)
X_test=scaler.transform(X_test)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)

pca.fit(X_train)

# Here's the description of the space X in the PCA components
X_pca = pca.transform(X_train)
#%%

plt.figure(figsize=(12,6))
plt.bar(np.arange(len(pca.components_[0])),pca.components_[0],label='comp 1',alpha=0.5)
plt.bar(np.arange(len(pca.components_[0])),pca.components_[1],label='comp 2',alpha=0.5)
plt.legend()
plt.xlabel('Feature')
plt.ylabel('Proporción')
plt.grid()
#%%
# Creación del modelo KMeans con k= #clusters

clusters=4

kmeans = KMeans(n_clusters=clusters)
# Fitting the model to reduced data in principal components
kmeans.fit(X_pca)

# Saving the positions of the centroids
centroids = kmeans.cluster_centers_

print("Shape de los centroids:",centroids.shape)
# Printing the positions of the centroids in the first two principal components
print(centroids[:,[0,1]])

# Plotting and cosmetic stuff
fig, ax = plt.subplots(figsize = (10, 7))
ax.scatter(X_pca[:, 0], X_pca[:, 1],s=1, c=kmeans.labels_, alpha=0.02)
ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=200, linewidths=1,
            c=np.unique(kmeans.labels_), edgecolors='black')
ax.set_xlabel('Primer componente principal')
ax.set_ylabel('Segunda componente principal')
#%%

# This is just for visualizing the presence of artists in the dataset by looking
# at the number of songs each one has in it.

artistas=list(d.artists.value_counts().index)
cuentas=d.artists.value_counts().values
n=30
offset=10
x=np.arange(n-offset)
fig, ax = plt.subplots()
ax.scatter(x, cuentas[offset:n])

plt.yscale('log')
plt.xlabel('artista')
plt.xticks(x, x)
plt.ylabel('Canciones en el dataset')
for i in range(n-offset):
    ax.annotate(artistas[offset+i], (x[i], cuentas[offset+i]), size=15)
