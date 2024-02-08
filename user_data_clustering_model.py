import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

import pickle

# KNN classifier
def knn (df):
    x_train = df.drop(['Cluster'], axis=1).values
    y_train = df['Cluster'].values
    
    knn = KNeighborsClassifier(n_neighbors=7)

    scaler = StandardScaler()
    scaler.fit(x_train)
    scaled_x_train = scaler.transform(x_train)

    knn.fit(scaled_x_train, y_train)

    return knn

# Main Program
# Data processing
df = pd.read_csv('dummy_v2.csv')
df.drop(['learner_type'], axis=1, inplace=True)
data = df.values # data converted to numpy array

# Normalize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Create cluster data
algo = KMeans(n_clusters=4, random_state=42)
clusters = algo.fit_predict(scaled_data)

# convert numpy arry to dataframe
clusters_df = pd.DataFrame(clusters, columns=['Cluster'])

#label mapping
label_mapping = {0: 'Poor', 1: 'Average', 2: 'Beginner', 3: 'Good'}
clusters_df['Cluster'] = clusters_df['Cluster'].map(label_mapping)

# Append to dataframe
clustered_df = pd.concat([df, clusters_df], axis=1)

# Create model
trained_model = knn(clustered_df)

# Make pickle file of our model
pickle.dump(trained_model, open("clustering_model.pkl", "wb"))