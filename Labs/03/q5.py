try:
    num = int(input("Enter the number of entries: "))
    d = {}

    for _ in range(num):
        name = input("Enter name: ")
        age = int(input("Enter age: "))
        d[name] = age

    with open('data.txt', 'w') as file:
        for name, age in d.items():
            file.write(f"{name}: {age}\n")

    max_age = max(d.values())
    max_age_names = [name for name, age in d.items() if age == max_age]

    print("Person(s) with the maximum age:", ", ".join(max_age_names))

    count = {}
    for age in d.values():
        if age in count:
            count[age] += 1
        else:
            count[age] = 1

    duplicate = [age for age, count in count.items() if count > 1]

    if duplicate:
        print("Duplicate age(s):", ", ".join(map(str, duplicate)))
    else:
        print("No duplicate ages found.")

except ValueError:
    print("Error: Invalid input for number of entries or age.")
except IOError:
    print("Error: An I/O error occurred while accessing the file.")