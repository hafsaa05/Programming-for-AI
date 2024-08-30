grades = {
    "Hafsa": [88, 99, 86],
    "Ahad": [77, 69, 84],
    "Amna": [78, 89, 92]
}

def add_grade(name, grade):
    if name in grades:
        grades[name].append(grade)
        print(f"Added grade {grade} to {name}.")
    else:
        print(f"Student {name} not found!")

def avg(name):
    if name in grades:
        average = sum(grades[name]) / len(grades[name])
        print(f"Average grades of Student {name} is {average}.")
    else:
        print(f"Student {name} not found!")

def add_student(name):
    if name not in grades:
        grades[name] = []
        print(f"Added new student {name}.")
    else:
        print(f"Student {name} already exists!")

def del_student(name):
    if name in grades:
        del grades[name]
        print(f"Removed student {name}.")
    else:
        print(f"Student {name} not found!")

add_grade("Ahad", 95)
avg("Hafsa")
add_student("Arwa")
del_student("Amna")
