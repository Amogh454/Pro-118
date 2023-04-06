import pandas as pd
import statistics as st
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sb


Stars = pd.read_csv("stars.csv")

size=Stars["Size"].tolist()
light=Stars["Light"].tolist()

scatter = px.scatter(x=size, y=light)
scatter.show()

X= Stars.iloc[:,[0,1]].values
print(X)

WCSS = []

for i in range(1, 11):
    KMean = KMeans(n_clusters=i,init='k-means++',random_state=42)
    KMean.fit(X)
    WCSS.append(KMean.inertia_)

plt.figure(figsize=(10,5))
sb.lineplot(WCSS, marker ='o', color = 'black')
plt.title('The Elbow Method')
plt.xlabel('x = no of clusters')
plt.ylabel('WCSS')
plt.show()


kMeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_kmeans = kMeans.fit_predict(X)

plt.figure(figsize=(15,7))
sb.scatterplot(x= X[y_kmeans==0,0], y=X[y_kmeans == 0,1], color = 'yellow', label = 'cluster1')
sb.scatterplot(x= X[y_kmeans==1,0], y=X[y_kmeans == 1,1], color = 'purple', label = 'cluster1')
sb.scatterplot(x= X[y_kmeans==2,0], y=X[y_kmeans == 2,1], color = 'blue', label = 'cluster1')
sb.scatterplot(x=kMeans.cluster_centers_[:,0], y=kMeans.cluster_centers_[:,1], color = 'red', label = 'centroid', s=100, markers = ',')

plt.grid(False)
plt.title('cluster of stars')
plt.xlabel('size')
plt.ylabel('light')
plt.show()
