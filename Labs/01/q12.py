# Write a program to print a dictionary where the keys are numbers between 1 and 15 (both included) and the values are the square of the keys.

numbers = {i:i ** 2 for i in range (1,16)}
print(numbers)
