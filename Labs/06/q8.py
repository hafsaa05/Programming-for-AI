import pandas as pd

data1 = pd.read_csv("products_extended.csv")
data2 = pd.read_csv("orders_extended.csv")

print("First 5 rows of products data:")
print(data1.head(5))
print("\nLast 10 rows of products data:")
print(data1.tail(10))

print("First 5 rows of orders data:")
print(data2.head(5))
print("\nLast 10 rows of orders data:")
print(data2.tail(10))

merge_data = pd.merge(data2, data1, on="ProductID")

merge_data['Revenue'] = merge_data['Quantity'] * merge_data['Price']
total_revenue = merge_data['Revenue'].sum()
print(f"\nTotal Revenue is {total_revenue:.2f}$")

best_selling = merge_data.groupby('ProductID').agg({'Quantity': 'sum'}).sort_values(by='Quantity', ascending=False).head(5)
low_selling = merge_data.groupby('ProductID').agg({'Quantity': 'sum'}).sort_values(by='Quantity', ascending=True).head(5)

print("\nTop 5 Best-Selling Products:")
print(best_selling)
print("\nTop 5 Low-Selling Products:")
print(low_selling)

best_selling_merged = pd.merge(best_selling, data1, on='ProductID')
most_common_category = best_selling_merged['Category'].mode()[0]
print(f"\nMost Common Category Among Top 5 Best-Selling Products: {most_common_category}")

avg_price = data1.groupby('Category')['Price'].mean().reset_index()
print("\nAverage Price of Products in Each Category:")
print(avg_price)

merge_data['Order Date'] = pd.to_datetime(merge_data['Order Date'])
merge_data['Day'] = merge_data['Order Date'].dt.day
merge_data['Month'] = merge_data['Order Date'].dt.month
merge_data['Year'] = merge_data['Order Date'].dt.year

day_revenue = merge_data.groupby('Day')['Revenue'].sum().idxmax()
month_revenue = merge_data.groupby('Month')['Revenue'].sum().idxmax()
year_revenue = merge_data.groupby('Year')['Revenue'].sum().idxmax()

print(f"\nDay with Highest Total Revenue: {day_revenue}")
print(f"Month with Highest Total Revenue: {month_revenue}")
print(f"Year with Highest Total Revenue: {year_revenue}")

monthly_revenue = merge_data.groupby('Month')['Revenue'].sum().reset_index()
print("\nTotal Revenue for Each Month:")
print(monthly_revenue)

print("\nChecking for null values:")
print(merge_data.isnull().sum())

merge_data = merge_data.dropna()

print("\nCleaned Data Overview:")
print(merge_data.head(5))
