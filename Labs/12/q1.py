import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

data, _ = make_blobs(n_samples=300, centers=5, cluster_std=1.0, random_state=42)
df = pd.DataFrame(data, columns=['X1', 'X2'])

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

sse = []
k_vals = range(1, 11)

for k in k_vals:
    model = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    model.fit(df_scaled)
    sse.append(model.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_vals, sse, marker='o')
plt.title('Elbow Curve')
plt.xlabel('Clusters')
plt.ylabel('SSE')
plt.grid(True)
plt.show()

opt_k = 5
final_model = KMeans(n_clusters=opt_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = final_model.fit_predict(df_scaled)

df['Cluster'] = clusters

plt.figure(figsize=(8, 6))

for cl in range(opt_k):
    plt.scatter(df_scaled[clusters == cl, 0], df_scaled[clusters == cl, 1], s=100, label=f'Group {cl + 1}')

plt.scatter(final_model.cluster_centers_[:, 0], final_model.cluster_centers_[:, 1], s=300, c='yellow', marker='X', label='Centroids')

plt.title(f'K-Means Clustering (k={opt_k})')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.show()

print(df.head())
