from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

iris_data = pd.read_csv('Iris.csv')
iris_data = iris_data.drop(columns=['SepalWidthCm', 'SepalLengthCm'])

plt.scatter(iris_data['PetalWidthCm'], iris_data['PetalLengthCm'], color='orange', label='Data Points')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Petal Dimensions')
plt.legend()
plt.show()

kmeans_model = KMeans(n_clusters=3)
clusters = kmeans_model.fit_predict(iris_data[['PetalLengthCm', 'PetalWidthCm']])
iris_data['Cluster'] = clusters

cluster_0 = iris_data[iris_data.Cluster == 0]
cluster_1 = iris_data[iris_data.Cluster == 1]
cluster_2 = iris_data[iris_data.Cluster == 2]

plt.scatter(cluster_0['PetalWidthCm'], cluster_0['PetalLengthCm'], color='green', label='Cluster 0')
plt.scatter(cluster_1['PetalWidthCm'], cluster_1['PetalLengthCm'], color='red', label='Cluster 1')
plt.scatter(cluster_2['PetalWidthCm'], cluster_2['PetalLengthCm'], color='blue', label='Cluster 2')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('KMeans Clustering (Unscaled)')
plt.legend()
plt.show()

scaler = MinMaxScaler()
iris_data[['PetalLengthCm', 'PetalWidthCm']] = scaler.fit_transform(iris_data[['PetalLengthCm', 'PetalWidthCm']])

kmeans_model_scaled = KMeans(n_clusters=3)
scaled_clusters = kmeans_model_scaled.fit_predict(iris_data[['PetalLengthCm', 'PetalWidthCm']])
iris_data['Cluster'] = scaled_clusters
cluster_centers = kmeans_model_scaled.cluster_centers_

scaled_cluster_0 = iris_data[iris_data.Cluster == 0]
scaled_cluster_1 = iris_data[iris_data.Cluster == 1]
scaled_cluster_2 = iris_data[iris_data.Cluster == 2]

plt.scatter(scaled_cluster_0['PetalWidthCm'], scaled_cluster_0['PetalLengthCm'], color='green', label='Cluster 0')
plt.scatter(scaled_cluster_1['PetalWidthCm'], scaled_cluster_1['PetalLengthCm'], color='red', label='Cluster 1')
plt.scatter(scaled_cluster_2['PetalWidthCm'], scaled_cluster_2['PetalLengthCm'], color='blue', label='Cluster 2')
plt.scatter(cluster_centers[:, 1], cluster_centers[:, 0], color='purple', label='Centroids', marker='+', s=200)
plt.xlabel('Petal Width (Normalized)')
plt.ylabel('Petal Length (Normalized)')
plt.title('KMeans Clustering (Scaled)')
plt.legend()
plt.show()

k_values = range(1, 10)
sse = []

for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(iris_data[['PetalWidthCm', 'PetalLengthCm']])
    sse.append(kmeans.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of Squared Errors')
plt.plot(k_values, sse)
plt.title('Elbow Method')
plt.show()
```
