import pandas as pd
import matplotlib.pyplot as plt

sea_level_records = pd.read_csv('sea_level_data.csv')
most_recent_year = sea_level_records['Year'].max()
recent_data = sea_level_records[sea_level_records['Year'] >= most_recent_year - 100]

plt.figure(figsize=(12, 6))
plt.scatter(recent_data['Year'], recent_data['SeaLevelChange'], color='green', label='Sea Level Change (mm)')

plt.title("Sea Level Changes Over the Last Century")
plt.xlabel("Year")
plt.ylabel("Change in Sea Level from 1993-2008 Average (mm)")
plt.grid(True)
plt.legend()
plt.show()
