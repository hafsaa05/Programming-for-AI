#-------------------------Simple Python----------------------------------------------------------------------------------------
# Data Types and Variables
name = "AI Student"  # String
age = 21  # Integer
height = 5.6  # Float
is_student = True  # Boolean

print(f"Name: {name}, Age: {age}, Height: {height}, Student: {is_student}")

# Nested Conditions
score = 85
if score > 90:
    print("A Grade")
elif 70 <= score <= 90:
    print("B Grade")
else:
    print("C Grade")

# Nested Loops
for i in range(1, 4):
    for j in range(1, 4):
        print(f"i={i}, j={j}")

# Function with Default Parameters
def greet(name="Student"):
    return f"Hello, {name}!"

print(greet("Hafsa"))

# OOP Example
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"My name is {self.name}, and I am {self.age} years old."

student = Person("Hafsa", 21)
print(student.introduce())

#------------------Matplotlib & Seaborn-----------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Lineplot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y, label="Sine Wave")
plt.title("Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()

# Histogram
data = np.random.randn(1000)
plt.hist(data, bins=30, color='blue', alpha=0.7)
plt.title("Histogram")
plt.show()

# Scatter Plot
plt.scatter(x, y, c='red', label="Data Points")
plt.legend()
plt.title("Scatter Plot")
plt.show()

# Box Plot
sns.boxplot(x=data)
plt.title("Box Plot")
plt.show()

# Heatmap
matrix = np.random.rand(5, 5)
sns.heatmap(matrix, annot=True, cmap='viridis')
plt.title("Heatmap")
plt.show()

#----------------Data Preprocessing--------------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Example DataFrame
df = pd.DataFrame({'Feature1': [1, 2, 3], 'Feature2': [4, 5, 6]})

# Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
print(scaled_data)

#---------------MAchine Learning-------------------
from sklearn.model_selection import train_test_split

X = np.random.rand(100, 3)
y = np.random.randint(0, 2, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape)

##KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

#q1
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

#q2
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

#q3
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

#KMeans example
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
print("Cluster Centers:", kmeans.cluster_centers_)

#Evaluation Metrics
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, predictions))

#lab task KNN
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

#PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

# Load Digits Dataset
digits_data = load_digits()
features = digits_data.data  # Pixel values
labels = digits_data.target  # Digit labels

# Standardize the Data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
principal_components = pca.fit_transform(scaled_features)

# Explained Variance
variance_ratio = pca.explained_variance_ratio_
print(f"Explained Variance Ratio: {variance_ratio}")
print(f"Total Variance Explained: {np.sum(variance_ratio):.2f}")

# Visualize in PCA Space
plt.figure(figsize=(10, 8))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=labels, cmap='tab10', s=50, alpha=0.7)
plt.colorbar(label="Digit Label")
plt.title("PCA of Handwritten Digits Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

#Lab task of PCA
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

#NLP
#eg 1
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Example Text
text = "NLP is amazing! Learn it to process language."

# Tokenization
tokens = word_tokenize(text)
print("Tokens:", tokens)

# Remove Punctuation
tokens = [word for word in tokens if word not in string.punctuation]

# Remove Stop Words
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("Filtered Tokens:", filtered_tokens)

# Convert to Vectors
vectorizer = CountVectorizer()
vectorized_text = vectorizer.fit_transform([text])
print("Count Vectorized:\n", vectorized_text.toarray())

tfidf = TfidfVectorizer()
tfidf_text = tfidf.fit_transform([text])
print("TFIDF Vectorized:\n", tfidf_text.toarray())

#lab task of NLP

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

email_data = {
    "message": [
        "Win $1000 now!", 
        "Your bank account needs verification", 
        "Meeting at 10 am", 
        "Congrats, you won a prize", 
        "Let's catch up tomorrow"
    ],
    "category": ["Spam", "Spam", "Ham", "Spam", "Ham"]
}

df = pd.DataFrame(email_data)

vectorizer = TfidfVectorizer(stop_words='english')
X_features = vectorizer.fit_transform(df['message'])
y_labels = df['category']

X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(X_features, y_labels, test_size=0.3, random_state=42)

classifier = MultinomialNB()
classifier.fit(X_train_set, y_train_set)

predictions = classifier.predict(X_test_set)

print(classification_report(y_test_set, predictions))

#eg 2 of NLP
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Iris Dataset
iris = load_iris()
features = iris.data
labels = iris.target
label_names = iris.target_names

# Standardize the Data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA
pca = PCA(n_components=2)
iris_pca = pca.fit_transform(scaled_features)

# Convert to DataFrame for Visualization
pca_df = pd.DataFrame(iris_pca, columns=['PC1', 'PC2'])
pca_df['Label'] = labels

# Plot the PCA Results
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=pca_df['Label'], palette='Set2')
plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(label_names, title="Iris Species")
plt.show()

# Explained Variance
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

#eg 3 of NLP : PCA for image compression
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load Digits Dataset
digits = load_digits()
image = digits.data[0]  # Select one image (first digit)
original_shape = (8, 8)  # Shape of the image

# Reshape and Display Original Image
image_reshaped = image.reshape(original_shape)
plt.figure(figsize=(6, 6))
plt.title("Original Image")
plt.imshow(image_reshaped, cmap='gray')
plt.axis('off')
plt.show()

# Apply PCA to Compress
pca = PCA(n_components=10)  # Keep only 10 components
compressed = pca.fit_transform(digits.data)  # Compress all data
reconstructed = pca.inverse_transform(compressed[0:1])  # Reconstruct the first image

# Reshape and Display Reconstructed Image
reconstructed_image = reconstructed.reshape(original_shape)
plt.figure(figsize=(6, 6))
plt.title("Reconstructed Image (10 Components)")
plt.imshow(reconstructed_image, cmap='gray')
plt.axis('off')
plt.show()

#PCA for feature selection
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Generate Synthetic Dataset
X, y = make_classification(n_samples=500, n_features=50, random_state=42)

# Apply PCA
pca = PCA(n_components=10)  # Reduce to 10 features
X_pca = pca.fit_transform(X)

# Compare Model Performance
model = RandomForestClassifier(random_state=42)

# Without PCA
original_score = cross_val_score(model, X, y, cv=5).mean()

# With PCA
pca_score = cross_val_score(model, X_pca, y, cv=5).mean()

print(f"Original Accuracy (Without PCA): {original_score:.2f}")
print(f"Accuracy with PCA (10 Components): {pca_score:.2f}")

#eg4: PCA for Time series data
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate Synthetic Time Series Data
np.random.seed(42)
time = np.linspace(0, 100, 500)
signals = np.sin(time) + np.random.normal(scale=0.5, size=(500,))
noisy_data = np.c_[signals, signals * 0.5 + np.random.normal(scale=0.3, size=(500,)), 
                   signals * 0.2 + np.random.normal(scale=0.2, size=(500,))]

# Apply PCA
pca = PCA(n_components=1)
reduced = pca.fit_transform(noisy_data)
reconstructed = pca.inverse_transform(reduced)

# Visualize Original and Reconstructed Signals
plt.figure(figsize=(12, 6))
plt.plot(time, signals, label="Original Signal", linewidth=2)
plt.plot(time, reconstructed[:, 0], label="Reconstructed Signal (PCA)", linestyle="--")
plt.title("PCA on Time Series Data")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

#-----------------------NUMPY----------------------
import numpy as np

# Create Arrays
arr = np.array([1, 2, 3, 4])
matrix = np.array([[1, 2], [3, 4]])

# Array Properties
print(arr.shape)  # Shape of the array
print(arr.ndim)   # Number of dimensions
print(arr.size)   # Total number of elements

# Generate Arrays
zeros = np.zeros((2, 2))
ones = np.ones((3, 3))
random = np.random.rand(2, 3)
range_array = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace_array = np.linspace(0, 1, 5)  # [0., 0.25, 0.5, 0.75, 1.]
# Reshape
reshaped = np.reshape(range_array, (5, 1))

# Indexing and Slicing
print(arr[1])      # Access second element
print(matrix[0, 1]) # Access row 0, column 1
print(arr[:2])     # First two elements

# Boolean Masking
mask = arr > 2
filtered = arr[mask]  # Elements greater than 2
# Element-wise Operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)  # [5, 7, 9]
print(a * b)  # [4, 10, 18]

# Aggregation
print(np.sum(a))      # Sum of elements
print(np.mean(b))     # Mean
print(np.max(a))      # Max
print(np.std(a))      # Standard Deviation

# Matrix Operations
dot_product = np.dot(a, b)  # Dot product
transpose = matrix.T        # Transpose
np.random.seed(42)  # Set seed for reproducibility
random_array = np.random.randint(0, 10, size=(3, 3))
print(random_array)

import numpy as np

# Generate a normal distribution
x = np.random.normal(loc=1, scale=2, size=(2, 3))
print("Normal Distribution:")
print(x)
# Generate a binomial distribution
x = np.random.binomial(n=10, p=0.5, size=10)
print("Binomial Distribution:")
print(x)
# Generate a Poisson distribution
x = np.random.poisson(lam=2, size=10)
print("Poisson Distribution:")
print(x)
# Generate a uniform distribution
x = np.random.uniform(low=0.0, high=1.0, size=(2, 3))
print("Uniform Distribution:")
print(x)
# Generate a logistic distribution
x = np.random.logistic(loc=1, scale=2, size=(2, 3))
print("Logistic Distribution:")
print(x)

#-------------PANDAS---------------------
import pandas as pd

# Create a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [24, 27, 22],
        'Score': [85, 90, 88]}
df = pd.DataFrame(data)

# Create a Series
series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])

# Inspect Data
print(df.head())    # First 5 rows
print(df.info())    # Summary
print(df.describe())  # Statistics

# Access Data
print(df['Name'])     # Access a column
print(df.iloc[1])     # Access row by index
print(df.loc[0, 'Name'])  # Access specific element
# Add a New Column
df['Passed'] = df['Score'] > 80

# Filter Rows
filtered = df[df['Age'] > 23]

# Drop Columns/Rows
df = df.drop(columns=['Passed'])
df = df.drop(index=[1])

# Sorting
sorted_df = df.sort_values(by='Score', ascending=False)
# Create a DataFrame with NaN
data_with_nan = {'Name': ['Alice', 'Bob', None],
                 'Age': [24, None, 22],
                 'Score': [85, 90, None]}
df_nan = pd.DataFrame(data_with_nan)

# Check for NaN
print(df_nan.isnull())  # Boolean mask of NaNs
print(df_nan.isnull().sum())  # Count NaNs per column

# Fill NaN
df_nan['Age'] = df_nan['Age'].fillna(df_nan['Age'].mean())

# Drop NaN
df_nan = df_nan.dropna()  # Drop rows with NaN
# Aggregations
print(df['Score'].mean())  # Mean of a column
print(df['Age'].sum())     # Sum of a column

# Grouping
grouped = df.groupby('Name')['Score'].mean()
print(grouped)

# Pivot Table
pivot = df.pivot_table(values='Score', index='Name', aggfunc='mean')
print(pivot)
# Reading CSV
df = pd.read_csv('data.csv')

# Writing CSV
df.to_csv('output.csv', index=False)

# Reading Excel
df_excel = pd.read_excel('data.xlsx')

# Writing Excel
df.to_excel('output.xlsx', index=False)
# Merge DataFrames
df1 = pd.DataFrame({'ID': [1, 2], 'Name': ['Alice', 'Bob']})
df2 = pd.DataFrame({'ID': [1, 2], 'Score': [85, 90]})
merged = pd.merge(df1, df2, on='ID')

# Concatenate DataFrames
concatenated = pd.concat([df1, df2], axis=0)
df['Score'].plot(kind='hist', title='Score Distribution')
plt.show()

df.plot(kind='scatter', x='Age', y='Score', title='Age vs Score')
plt.show()
