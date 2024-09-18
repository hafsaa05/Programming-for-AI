from abc import ABC, abstractmethod

class shape:
    @abstractmethod    
    def area(self, area):   
        pass
    
class rectangle(shape):
    def area(self, l, b):
        return l*b

class triangle(shape):
    def area(self, l, b):
        return (l*b)/2

class sqr(shape):
    def area(self, l):
        return l*l

rec = rectangle()
print(f" Area of rectangle: {rec.area(5,6)} ") 

tri = triangle()
print(f" Area of triangle: {tri.area(5,6)} ")

sqr = sqr()
print(f" Area of square: {sqr.area(5)} ")
