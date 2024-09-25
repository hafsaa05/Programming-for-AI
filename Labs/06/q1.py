import pandas as pd

df = pd.read_csv('movies.csv')

filtered = df[(df['revenue'] > 2000000) & (df['budget'] < 1000000)]
print(filtered)
