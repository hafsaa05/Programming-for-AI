import pandas as pd

data = pd.read_excel("employee.xlsx")
emp = data[data["JoiningYear"] == 2020]
print(emp)


#do pip install pandas openpyxl
