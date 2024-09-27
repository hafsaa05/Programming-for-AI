import pandas as pd

data = pd.read_csv("world_alcohol.csv")
consumption = data[(data['Year']==1987) | (data['Year']==1989)]
print(consumption)