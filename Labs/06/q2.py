import pandas as pd
data = {
    'movies': ['Deadpool', 'Pride & Prejudice', 'Glassworker', 'Harry Potter'],
    'revenue': [25000000, 15000000, 5000000, 1200000],
    'budget': [800000, 1200000, 950000, 700000]
}

df = pd.DataFrame(data)
sort = df.sort_values(by='budget' , ascending=False)

print(sort)
