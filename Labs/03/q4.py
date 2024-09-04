try:
    name = input("Enter your name: ")
    cnic = input("Enter your CNIC number: ")
    age = input("Enter your age: ")
    salary = input("Enter your salary: ")

    biodata = f"Name: {name}\nCNIC: {cnic}\nAge: {age}\nSalary: {salary}\n"

    with open(r'biodata.txt', 'w') as file:
        file.write(biodata)

    number = input('Enter your contact number: ')
    with open(r'biodata.txt', 'a') as file: 
        file.write(f"Contact Number: {number}\n")

    with open(r'biodata.txt', 'r') as file:
        content = file.read()
        print("\nFile Content:\n")
        print(content)

except FileNotFoundError:
    print("Error: The file 'biodata.txt' was not found.")

except IOError:
    print("Error: An I/O error occurred while accessing the file 'biodata.txt'.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
