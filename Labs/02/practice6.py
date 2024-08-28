name = input("Enter your full name: ")

split = name.split()

if len(split) >= 2:
    first = split[0]
    last = split[-1]
    print(f"First Name: {first}")
    print(f"Last Name: {last}")
else:
    print("Please enter both first and last names.")
