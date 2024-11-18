import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

heartDataset = pd.read_csv('/content/heart.csv')
targetLabels = heartDataset.pop('target')

XTrain, XTest, YTrain, YTest = train_test_split(heartDataset, targetLabels, test_size=0.2, random_state=8)

highestAccuracy = -1
lowestAccuracy = 1
optimalK = None
suboptimalK = None

for varin in range(1, 251):
    knnModel = KNeighborsClassifier(n_neighbors=varin)
    knnModel.fit(XTrain, YTrain)
    predictions = knnModel.predict(XTest)

    accuracy = accuracy_score(YTest, predictions)

    if accuracy > highestAccuracy:
        highestAccuracy = accuracy
        optimalK = varin

    if accuracy < lowestAccuracy:
        lowestAccuracy = accuracy
        suboptimalK = varin

print("Highest Accuracy: ", highestAccuracy)
print("K-value for Highest Accuracy: ", optimalK)
print("Lowest Accuracy: ", lowestAccuracy)
print("K-value for Lowest Accuracy: ", suboptimalK)

plt.figure(figsize=(10, 6))
plt.title("KNN Accuracy for Heart Disease Prediction")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.plot(range(1, 251), [KNeighborsClassifier(n_neighbors=varin).fit(XTrain, YTrain).score(XTest, YTest) for varin in range(1, 251)], marker='o', color='b', label='Accuracy Curve')
plt.axvline(x=optimalK, color='g', linestyle='--', label=f'Highest Accuracy (K={optimalK})')
plt.axvline(x=suboptimalK, color='r', linestyle='--', label=f'Lowest Accuracy (K={suboptimalK})')
plt.legend()
plt.show()
