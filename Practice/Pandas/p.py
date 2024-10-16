import pandas as pd

# Series: A one-dimensional array-like structure.
data = [10, 20, 30, 40]
s = pd.Series(data)
print(s)

# DataFrame: A two-dimensional labeled data structure.
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "Salary": [50000, 60000, 70000]
}
df = pd.DataFrame(data)
print(df)

# df.head() -> First 5 rows (default).
print(df.head())

# df.tail() -> Last 5 rows (default).
print(df.tail())

# df.info() -> Summary of DataFrame.
df.info()

# df.describe() -> Statistical summary.
df.describe()

# df.shape -> Returns dimensions (rows, columns).
print(df.shape)

# df.columns -> Get column names.
print(df.columns)

# df.index -> Get row indexes.
print(df.index)

# df.dtypes -> Data types of each column.
print(df.dtypes)

# df['column_name'] -> Access column as a Series.
print(df['Age'])

# df[['column1', 'column2']] -> Access multiple columns.
print(df[['Name', 'Salary']])

# df.iloc[row_index] -> Access row by integer index.
print(df.iloc[0])

# df.loc[row_label] -> Access row by label/index name.
print(df.loc[0])

# df.loc[start:end, ['col1', 'col2']] -> Access specific rows and columns.
print(df.loc[0:2, ['Name', 'Salary']])

# Adding a new column.
df['Bonus'] = df['Salary'] * 0.10
print(df)

# Deleting a column.
df = df.drop('Bonus', axis=1)
print(df)

# Renaming columns.
df = df.rename(columns={"Name": "Employee Name", "Age": "Employee Age"})
print(df)

# Filtering data.
filtered_df = df[df['Employee Age'] > 30]
print(filtered_df)

# Sorting data.
sorted_df = df.sort_values(by='Salary', ascending=False)
print(sorted_df)

# Handling missing data.
df.loc[3] = ["David", None, None]
print(df)

# Drop rows with missing values.
df_cleaned = df.dropna()
print(df_cleaned)

# Fill missing values with a default value.
df_filled = df.fillna(0)
print(df_filled)

# Grouping and aggregating data.
grouped_df = df.groupby('Employee Name').mean()
print(grouped_df)

# Merging DataFrames.
df2 = pd.DataFrame({
    "Employee Name": ["Alice", "Bob", "Charlie"],
    "Department": ["HR", "Engineering", "Finance"]
})
merged_df = pd.merge(df, df2, on="Employee Name")
print(merged_df)

# Exporting DataFrame to a CSV file.
df.to_csv('employees.csv', index=False)

# Reading from a CSV file.
df_from_csv = pd.read_csv('employees.csv')
print(df_from_csv)
