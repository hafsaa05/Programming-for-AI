import numpy as np

arr = np.random.randint(1,100,50)
print(f'Original : {arr}')
print(f'Index of max value : {arr.argmax()}')
print(f'Index of min value : {arr.argmin()}')
