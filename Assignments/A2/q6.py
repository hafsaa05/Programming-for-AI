import numpy as np

arr = np.arange(2,6).reshape(2,2)
print(f'Determinant: {np.linalg.det(arr)}')
print(f'Inverse\n: {np.linalg.inv(arr)}')
