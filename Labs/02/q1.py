
def trapezoid(x, y, z):
    print('Area of trapezoid: ' ,((x+y)*z)/2)

def Parallelogram(a, b):
    print("Area of Parallelogram: " ,(a*b))

def cylinder(r, h):
    print("Volume of cylinder: ", (3.142*r*r*h))
    print("Surface area of cylinder: ", (2*3.142*r * (r+h)))

trapezoid(6, 6, 8)
Parallelogram(7, 9)
cylinder(2, 3)
