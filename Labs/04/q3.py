class Student:
    name=""
    age=""
    def __init__(self, name, age):
        Student.name=name
        Student.age=age
        
    def Print_biodata(self):
        print("Name: {}".format(self.name))
        print("Age: {}".format(self.age))
        
stu = Student("Hafsa", "19")
stu.Print_biodata()
