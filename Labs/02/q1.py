import math

def trapezoid(x, y, z):
    print('Area of trapezoid: ' ,((x+y)*z)/2)

def Parallelogram(a, b):
    print("Area of Parallelogram: " ,(a*b))

def cylinder(r, h):
    print("Volume of cylinder: ", (math.pi*r*r*h))
    print("Surface area of cylinder: ", (2*math.pi*r * (r+h)))

trapezoid(6, 6, 8)
Parallelogram(7, 9)
cylinder(2, 3)
