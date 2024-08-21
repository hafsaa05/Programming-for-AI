#Write a program to take an integer list input from user and count all the even numbersin that list and print the count.

num = int(input("Enter the number of integers for the list: "))

user_input = list(map(int, input("Enter the integers for the list separated by spaces: ").split()))

count = 0

for i in range(num):
    if user_input[i] % 2 == 0:
        count += 1

print("Count of even numbers:", count)
