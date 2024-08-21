# Write a Python program to get the largest number from a list input from user.

numbers = []
num = int(input("Enter the number of integers for the list: "))

for i in range(num):
    user_input = int(input("Enter Number: "))
    numbers.append(user_input)

largest = numbers[0]

for number in numbers:
    if number > largest:
        largest = number

print("Largest number:", largest)
