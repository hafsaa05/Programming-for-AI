import pandas as pd

df = pd.read_csv("heart.csv")
df = df[(df["cp"] == 2)]
print(df)
