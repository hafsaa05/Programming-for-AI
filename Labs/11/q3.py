import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

columns = [
    "erythema", "scaling", "definite_borders", "itching", "koebner_phenomenon",
    "polygonal_papules", "follicular_papules", "oral_mucosal_involvement",
    "knee_elbow_involvement", "scalp_involvement", "family_history", "age", "target_class"
]
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data"
dermatology_data = pd.read_csv(url, header=None, names=columns, na_values=["?"])
dermatology_data['age'].fillna(dermatology_data['age'].mean(), inplace=True)

X_features = dermatology_data.drop(columns=['target_class'])
y_target = dermatology_data['target_class']

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_features)

knn_classifier = KNeighborsClassifier(n_neighbors=5)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
cv_accuracy_scores = cross_val_score(knn_classifier, X_normalized, y_target, cv=kfold, scoring='accuracy')

print("10-Fold Cross-Validation Accuracy Scores:", cv_accuracy_scores)
print("Mean Cross-Validation Accuracy:", np.mean(cv_accuracy_scores))

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_target, test_size=0.3, random_state=42)
knn_classifier.fit(X_train, y_train)

y_test_predictions = knn_classifier.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_test_predictions)
print("Confusion Matrix:\n", conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_target), yticklabels=np.unique(y_target))
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:\n", classification_report(y_test, y_test_predictions))
