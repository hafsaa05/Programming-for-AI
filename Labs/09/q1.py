import matplotlib.pyplot as plt

students = ["Hafsa", "Amna", "Ahad", "Maryam", "Hamna", "Ayesha", "Ahmed", "Ali", 
            "Talha", "Sara", "Omer", "Zain", "Aliya", "Raza", "Sana", "Faisal", 
            "Asad", "Tariq", "Nida", "Bilal"]
heights = [90, 80, 62, 65, 77, 78, 86, 96, 79, 88, 75, 72, 81, 93, 70, 82, 85, 89, 76, 66]
clr = ["purple", "pink", "black", "red", "orange", "yellow", "green", "blue", 
       "indigo", "violet", "cyan", "magenta", "brown", "lime", "navy", "teal", 
       "coral", "gold", "silver", "gray"]

plt.figure(figsize=(10, 5))  # for better visualization
plt.bar(students, heights, color=clr)
plt.xlabel('Students')
plt.ylabel('Height')
plt.title('Student Height Data')
plt.xticks(rotation=45)  # Rotate student names for better readability
plt.show()

plt.figure(figsize=(8, 8))  # for better visualization
plt.pie(heights, labels=students, colors=clr, autopct='%1.1f%%')
plt.title('Student Heights Distribution')
plt.show()
