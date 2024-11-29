```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import seaborn as sns

digits_dataset = load_digits()
features = pd.DataFrame(digits_dataset.data)
labels = digits_dataset.target

first_row = features.iloc[0]
image_array = first_row.to_numpy()
image_grid = image_array.reshape(8, 8)

plt.figure(figsize=(6, 6))
plt.imshow(image_grid, cmap='gray')
plt.axis('off')
plt.title("First Digit Image")
plt.show()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(scaled_features)

variance_ratio = pca.explained_variance_ratio_
print(f"Explained variance by the first 2 components: {variance_ratio}")
print(f"Total variance explained by 2 components: {np.sum(variance_ratio):.2f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=pca_transformed[:, 0],
    y=pca_transformed[:, 1],
    hue=labels,
    palette='tab10',
    s=100,
    marker='o',
    legend='full'
)
plt.title("PCA of Digits Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Digit Label")
plt.show()
```
