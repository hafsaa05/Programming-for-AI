# Create 3x3 matrix containing only even numbers. Multiply each element of this array with
# 4 and then multiply resultant matrix with 3x3 identity matrix (identity matrix should not be
# hardcoded).
import numpy as np

arr = np.arange(2, 20, 2)
new_arr = arr.reshape(3, 3)
print(f'Array: {new_arr}')

mul_arr = arr*4
mul_arr = mul_arr.reshape(3,3)
print(f'Multiplied array: {mul_arr}')

identity = np.eye(3)
res = mul_arr*identity
print(f'Resultant matrix: {res}')
        
