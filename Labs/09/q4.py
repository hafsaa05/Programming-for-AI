import matplotlib.pyplot as plt
import pandas as pd

data = {
    'Age': [17, 22, 19, 30, 25, 35, 40, 28, 33, 42, 27, 21, 18, 29, 31, 26, 39, 38, 24, 23]
}

df = pd.DataFrame(data)

age_groups = ["15-25", "26-30", "31-35", "36-40", "41 and above"]
A1 = df[(df['Age'] >= 15) & (df['Age'] <= 25)].shape[0]
A2 = df[(df['Age'] > 25) & (df['Age'] <= 30)].shape[0]
A3 = df[(df['Age'] > 30) & (df['Age'] <= 35)].shape[0]
A4 = df[(df['Age'] > 35) & (df['Age'] <= 40)].shape[0]
A5 = df[df['Age'] > 40].shape[0]

age_distribution = [A1, A2, A3, A4, A5]

plt.figure(figsize=(8, 8))
plt.pie(age_distribution, labels=age_groups, autopct='%1.1f%%', startangle=140)
plt.title("Student Age Distribution")
plt.show()
