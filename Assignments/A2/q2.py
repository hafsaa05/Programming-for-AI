import numpy as np

arr = np.random.randint(1, 50, size = 9)
arr = arr.reshape(3, 3)
print(f'Original\n: {arr}')
print(f'Transpose\n: {arr.transpose()}')
