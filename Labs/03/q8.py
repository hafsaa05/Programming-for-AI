try:
    num1 = int(input("Enter a number: "))
    num2 = int(input("Enter another number: "))
    
    div = num1 / num2
    print(f"The result is: {div}")

except ZeroDivisionError:
    print("Cannot divide by zero!")

except ValueError:
    print("Invalid input! Please enter valid integer numbers.")
