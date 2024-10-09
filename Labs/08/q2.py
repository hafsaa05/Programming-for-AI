# Create a multi-dimensional array of 3 rows and 3 columns having odd numbers from 1 to
# 19 including 1 and excluding 19. After that, iterate over this array to print all elements.
import numpy as np
arr = np.arange(1, 19, 2)
new_arr = arr.reshape(3, 3)
print(f'Array: {new_arr}')

print("\nElements in the array:")
for r in new_arr:
    for e in r:
        print(e)
