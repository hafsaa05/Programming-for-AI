# Write a program to take marks of 3 subjects as input by user and store them in a dictionary
# having appropriate keys. Using that dictionary calculate average and percentage of those marks.

marks = {}
subjects = ['PAI', 'DS', 'LA']

for sub in subjects:
    mark = int(input(f"Enter marks for {sub}: "))
    marks[sub] = mark

avg = sum(marks.values()) / len(marks)
print(f"Average marks: {avg}")

percentage = (sum(marks.values()) / 300) * 100
print("Percentage:", percentage)
