# Write a program that takes a list of numbers as input and returns the sum of all the elements in the list.

numbers = []

sum = 0

num = int(input("Enter the number of integers for the list: "))

for i in range(num):
    user_input = int(input("Enter Number: "))
    numbers.append(user_input)

for num in numbers:
    sum += num

print("Sum:", sum)
