import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

heartData = pd.read_csv('/content/heart.csv')
targetValues = heartData.pop('target')

maxAcc = -1
minAcc = 1
accuracies = []

for seed in range(1, 11):
    XTrain, XTest, YTrain, YTest = train_test_split(heartData, targetValues, test_size=0.2, random_state=seed)
    knnClassifier = KNeighborsClassifier(n_neighbors=3)
    knnClassifier.fit(XTrain, YTrain)
    predictions = knnClassifier.predict(XTest)

    currentAcc = accuracy_score(YTest, predictions)
    accuracies.append(currentAcc)
    
    if currentAcc > maxAcc:
        maxAcc = currentAcc
    
    if currentAcc < minAcc:
        minAcc = currentAcc

print("Accuracies for each seed: ", accuracies)
print("Highest Accuracy: ", maxAcc)
print("Lowest Accuracy: ", minAcc)

