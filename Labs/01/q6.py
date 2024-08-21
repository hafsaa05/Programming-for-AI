# Aliza has got 40 Marks in Physics, 78 in Chemistry and 82 in Maths. Take these marks as
# input from user and store them in dictionary with "key as subject name" and "value as marks". By
# accessing data stored in dictionary, print average of his marks and also print the name of subject in
# which she got highest marks.

aliza = {}

aliza['Physics'] = int(input("Enter marks for Physics: "))
aliza['Chemistry'] = int(input("Enter marks for Chemistry: "))
aliza['Maths'] = int(input("Enter marks for Maths: "))

avg = sum(aliza.values()) / len(aliza)
print(f"Average marks: {avg}")

highest_marks = -1
highest_subject = ""

for subject, marks in aliza.items():
    if marks > highest_marks:
        highest_marks = marks
        highest_subject = subject

print(f"Highest marks in: {highest_subject}")
