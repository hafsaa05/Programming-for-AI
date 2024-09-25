import pandas as pd

df = pd.read_csv('movies2.csv')

sort = df.sort_values(by='runtime', ascending=False)

print(sort)