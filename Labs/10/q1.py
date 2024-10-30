import numpy as np
import abc as ab
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#-----------------------------------------------PAPER A----------------------------------------------------------------------------------------------------
df = pd.read_csv('heart.csv')

plt.figure(figsize=(12, 18))

plt.subplot(7, 2, 1)
plt.grid(visible=True)
sns.histplot(df, x='age')
plt.title('Age Distribution')

plt.subplot(7, 2, 2)
plt.grid(visible=True)
sns.histplot(df, x='sex')
plt.title('Sex Distribution')

plt.subplot(7, 2, 3)
plt.grid(visible=True)
sns.histplot(df, x='cp')
plt.title('Chest Pain Type')

plt.subplot(7, 2, 4)
plt.grid(visible=True)
sns.histplot(df, x='trestbps')
plt.title('Resting Blood Pressure')

plt.subplot(7, 2, 5)
plt.grid(visible=True)
sns.histplot(df, x='chol')
plt.title('Cholesterol')

plt.subplot(7, 2, 6)
plt.grid(visible=True)
sns.histplot(df, x='fbs')
plt.title('Fasting Blood Sugar')

plt.subplot(7, 2, 7)
plt.grid(visible=True)
sns.histplot(df, x='restecg')
plt.title('Resting ECG')

plt.subplot(7, 2, 8)
plt.grid(visible=True)
sns.histplot(df, x='thalach')
plt.title('Max Heart Rate')

plt.subplot(7, 2, 9)
plt.grid(visible=True)
sns.histplot(df, x='exang')
plt.title('Exercise-induced Angina')

plt.subplot(7, 2, 10)
plt.grid(visible=True)
sns.histplot(df, x='oldpeak')
plt.title('ST Depression (Oldpeak)')

plt.subplot(7, 2, 11)
plt.grid(visible=True)
sns.histplot(df, x='slope')
plt.title('Slope of Peak Exercise ST Segment')

plt.subplot(7, 2, 12)
plt.grid(visible=True)
sns.histplot(df, x='ca')
plt.title('Number of Major Vessels')

plt.subplot(7, 2, 13)
plt.grid(visible=True)
sns.histplot(df, x='thal')
plt.title('Thalassemia')

plt.subplot(7, 2, 14)
plt.grid(visible=True)
sns.histplot(df, x='target')
plt.title('Heart Disease Presence')

plt.tight_layout()
plt.show()

#-------------------------------------------------------------PAPER A-----------------------------------------------------
plt.figure()
sns.histplot(data=df,x="fbs")
plt.show()

plt.figure()
sns.boxplot(x="fbs",y="age",data=df)
plt.show()

plt.figure()
sns.scatterplot(data=df,x="fbs",y="age")
plt.show()

plt.figure()
sns.lineplot(y=df["age"],x=df["fbs"])
plt.show()

plt.figure()
sns.displot(df)
plt.ylabel("fbs")
plt.show()

plt.figure()
sns.pairplot(df,hue="age")
plt.show()
