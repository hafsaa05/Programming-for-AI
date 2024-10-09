# Generate 4x4 matrix with random numbers only from (2,5,9,10) and then multiply this
# matrix with identity matrix. Finally add this matrix to any 4x4 matrix having only odd
# numbers.
import numpy as np

arr = [2, 5, 9, 10]
new_arr = np.random.choice(arr, size=(4, 4))
print(new_arr)

identity = np.eye(4,4)
res = new_arr*identity
print(f'Multiplying with identity matrix: \n{res}')

arr2 = np.arange(1, 33, 2).reshape(4, 4)
res2 = np.add(res,arr2)
print(arr2)
print(f'Adding matrices: \n{res2}')
