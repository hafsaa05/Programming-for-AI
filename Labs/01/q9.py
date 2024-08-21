# Write a Python program to create the multiplication table (from 1 to 10) of a number.

num = int(input("Enter a number (1 to 10): "))
for i in range (1,11):
    print(num, "*", i ,"=", (num*i) )
