# Write a program to take a list and a number input from user and then delete all elements in the list less than that number.
numbers = []

num = int(input("Enter the number of integers for the list: "))
for i in range(num):
    user_input = int(input("Enter Number: "))
    numbers.append(user_input)

delete = int(input("Enter the minimum number (elements less than this will be deleted): "))

numbers = [number for number in numbers if number >= delete]

print("List after deleting elements less than", ":", numbers)
