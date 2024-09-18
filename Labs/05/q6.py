from abc import ABC, abstractmethod
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
    
    @abstractmethod
    def calculate_bonus(self, bonus):
        pass

class manager(Employee):
    def __init__(self, name, salary):
        super().__init__(name, salary) 
    
    def calculate_bonus(self):
         self.salary += self.salary*0.2
         return self.salary

    def hire(self):
        print("Manager is hiring someone.")

class developers(Employee):
    def __init__(self, name, salary):
        super().__init__(name, salary) 
    
    def calculate_bonus(self):
        self.salary += self.salary*0.1
        return self.salary

    def hire(self):
        print("Developer is writing code.")

class SeniorManager(manager):
    def calculate_bonus(self):
         self.salary += self.salary*0.3
         return self.salary

man = manager("Hafsa", 2500)
print(f"Salary with bonus of Manager is {man.calculate_bonus()}")
man.hire()

dev = developers("Hafsa", 2500)
print(f"Salary with bonus of Developer is {dev.calculate_bonus()}")
dev.hire()

sen = SeniorManager("Hafsa", 2500)
print(f"Salary with bonus of Senior Manager is {sen.calculate_bonus()}")


