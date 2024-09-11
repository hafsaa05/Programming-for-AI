class Employee:
    def __init__(self):
        self.name = ""
        self.salary = 0.0
        self.tax = 0.0
        
    def get_data(self, name, salary, tax):
        self.name = name
        self.salary = float(salary)
        self.tax = float(tax)
        
    def Salary_after_tax(self):
        return self.salary - (self.salary * (self.tax / 100))
        
    def update_tax_percentage(self, new_tax):
        self.tax = new_tax
        
emp = Employee()
emp.get_data("Hafsa", 1500, 2)
print(f"Salary after 2% tax: {emp.Salary_after_tax()}")

emp.update_tax_percentage(3)
print(f"Salary after 3% tax: {emp.Salary_after_tax()}")
