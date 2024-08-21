# Write a program to make a simple calculator performing the four basic operations (+, -, *, /) on two
# numbersinputby user. The following conditionsmustbe satisfied:
# a) A ‘+’ sign must not be used foraddition.
# b) You can only use a maximum of 3 variables. No more variables are allowed.
# c) Your program should ask the user which operation he/she wants to perform and gives the
# output accordingly.
      
num1 = float(input("Enter a number: "))
num2 = float(input("Enter another number: "))
opr = input("Enter an operator (+, -, *, /): ")

if opr == '+':
    result = (num1 - (-num2))
    print("Sum:", result)
elif opr == '-':
    result = num1 - num2
    print("Subtraction:", result)
elif opr == '*':
    result = num1 * num2
    print("Multiplication:", result)
elif opr == '/':
    if num2 == 0:
        print("cant divide by zero")
    else:
        result = num1 / num2
        print("Division:", result)
else:
    print("Invalid operation.")
