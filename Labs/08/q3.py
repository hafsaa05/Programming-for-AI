# Create 3x3 matrix containing only even numbers. Multiply each element of this array with
# 4 and then multiply resultant matrix with 3x3 identity matrix (identity matrix should not be
# hardcoded).
import numpy as np

arr = np.arange(2, 20, 2).reshape(3, 3)
print(f'Array:\n{arr}')

mul_arr = np.multiply(arr, 4)
print(f'Multiplied array:\n{mul_arr}')

identity = np.eye(3)
res = np.multiply(mul_arr, identity)
print(f'Resultant matrix:\n{res}')
