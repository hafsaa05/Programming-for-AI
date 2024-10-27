import matplotlib.pyplot as plt

students = ["Hafsa", "Amna", "Ahad", "Maryam", "Hamna", 
            "Ayesha", "Ahmed", "Ali", "Talha", "Sara"]
math_marks = [85, 92, 76, 88, 90, 82, 79, 95, 80, 77]
science_marks = [78, 84, 70, 89, 92, 75, 80, 91, 83, 76]

plt.figure(figsize=(10, 6))
plt.scatter(students, math_marks, color='blue', label='Mathematics', marker='o', s=100)
plt.scatter(students, science_marks, color='green', label='Science', marker='s', s=100)
plt.xlabel('Students')
plt.ylabel('Marks')
plt.title('Comparison of Mathematics and Science Marks')
plt.xticks(rotation=45)  
plt.legend()  
plt.grid(True)  
plt.show()

students_heights = ["Hafsa", "Amna", "Ahad", "Maryam", "Hamna", "Ayesha", 
                    "Ahmed", "Ali", "Talha", "Sara"]
heights = [90, 80, 62, 65, 77, 78, 86, 96, 79, 88]
clr = ["purple", "pink", "black", "red", "orange", "yellow", 
       "green", "blue", "indigo", "violet"]

plt.figure(figsize=(10, 5))
plt.bar(students_heights, heights, color=clr)
plt.xlabel('Students')
plt.ylabel('Height')
plt.title('Student Height Data')
plt.xticks(rotation=45)  
plt.show()

# Pie chart for student heights distribution
plt.figure(figsize=(8, 8))
plt.pie(heights, labels=students_heights, colors=clr, autopct='%1.1f%%')
plt.title('Student Heights Distribution')
plt.show()
