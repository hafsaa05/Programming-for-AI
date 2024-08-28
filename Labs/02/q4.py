def employee(name, salary=10000):
    new_salary= salary * (1 - 0.02)
    print(f"Name: {name}, Salary after tax deduction: {new_salary:.2f} $")

employee("Hafsa", 750000)  
employee("Adina")   

