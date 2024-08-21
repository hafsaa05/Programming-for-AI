#Write a program to take an integer list input from user and count all the even numbersin that list and print the count.

numbers = []

count = 0

num = int(input("Enter the number of integers for the list: "))

for i in range(num):
    user_input = int(input("Enter Number: "))
    numbers.append(user_input)

for number in numbers:
    if number % 2 == 0:
        count += 1

print("Count of even numbers:", count)




