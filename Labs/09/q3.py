import matplotlib.pyplot as plt

flavors = ['Vanilla', 'Mint Chocolate Chip', 'Cookie Dough', 'Rocky Road', 'Butter Pecan']
scoops_sold = [12, 18, 22, 30, 25]  

plt.figure(figsize=(8, 8))
plt.pie(scoops_sold, labels=flavors, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Ice Cream Sales by Flavor')

plt.show()
