def factorial(n):
    if n < 0:
        return "Enter positive number."
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

n = int(input("Enter a number: "))
print(factorial(n))
