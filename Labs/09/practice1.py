import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

df = pd.read_csv("iris.csv")
df = pd.DataFrame(df)

x_values = df["sepal_length"]
y_values = df["petal_length"]

plt.plot(x_values, y_values)
plt.show()
