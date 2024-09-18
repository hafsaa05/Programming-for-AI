class student:
    def __init__(self, name, id):
        self.name = name
        self.id = id
    
    def display_info(self):
        print(f"Name: {self.name}")
        print(f"Student ID: {self.id}")    
        
class marks(student):
    def __init__(self, name, id, marks_algo , marks_dataScience, marks_calculus):
        super().__init__(name, id)
        self.marks_algo = marks_algo
        self.marks_dataScience = marks_dataScience
        self.marks_calculus = marks_calculus
        
    def display_marks(self):
        print(f"Algorithm marks: {self.marks_algo}")
        print(f"Data Science marks: {self.marks_dataScience}") 
        print(f"Calculus marks: {self.marks_calculus}") 
        
class result(marks):
    def __init__(self, name, id, marks_algo , marks_dataScience, marks_calculus):
        super().__init__(name, id, marks_algo , marks_dataScience, marks_calculus) 
        
    def display_result(self):
         self.total =  self.marks_algo +  self.marks_dataScience + self.marks_calculus  
         print(f"Total marks: {self.total}")
         self.avg = self.total/3
         print(f"Average marks: {self.avg}") 
 
stu = student("Hafsa", 64)
stu.display_info()   

mark = marks("Hafsa", 64, 86, 87, 88)
mark.display_marks()

res = result("Hafsa", 64, 86, 87, 88)
res.display_result()
