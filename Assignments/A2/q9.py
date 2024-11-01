import numpy as np

arr = np.random.randint(1, 100, 25)
print(f'Original : {arr}')
print(f'75th Percentile : {np.percentile(arr, 75)}')
