import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
sales = [200, 220, 250, 300, 320]
expenses = [180, 190, 210, 240, 260]

plt.plot(x, sales, 'o-', color='navy', label='Sales ($)')
plt.plot(x, expenses, 'o-', color='silver', label='Expenses ($)')

plt.xlabel('Months')
plt.ylabel('Dollars')
plt.title('Sales and Expenses Over 5 Months')
plt.legend(loc='lower right')
plt.show()

