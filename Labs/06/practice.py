import pandas as pd

data = pd.read_csv('heart.csv')
print(data.head(30))
print(data.tail(30))
