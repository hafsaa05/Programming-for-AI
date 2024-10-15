import pandas as pd

# Creating a Series from a list

data = [10, 20, 30, 40, 50]
series = pd.Series(data)
print(series)

# Creating a DataFrame from a list of lists

data = [[1, 'Alice', 23], [2, 'Bob', 25], [3, 'Charlie', 22]]
df = pd.DataFrame(data, columns=['ID', 'Name', 'Age'])
print(df)

# Creating a DataFrame from a dictionary of lists

data_dict = {
    'ID': [1, 2, 3],
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [23, 25, 22]
}

df_from_dict = pd.DataFrame(data_dict)
print(df_from_dict)
print(df_from_dict.shape) #gives dimensions
