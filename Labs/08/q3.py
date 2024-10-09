# Create 3x3 matrix containing only even numbers. Multiply each element of this array with
# 4 and then multiply resultant matrix with 3x3 identity matrix (identity matrix should not be
# hardcoded).
import numpy as np

arr = np.arange(2, 20, 2).reshape(3, 3)
print(f'Array: {arr}')

mul_arr = (arr*4).reshape(3,3)
print(f'Multiplied array: {mul_arr}')

res = mul_arr*(np.eye(3))
print(f'Resultant matrix: {res}')
        
