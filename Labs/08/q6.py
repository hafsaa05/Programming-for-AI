# Write a NumPy program to multiply a 5x3 matrix by a 3x2.
import numpy as np

arr1 = np.arange(2, 17).reshape(5, 3)
arr2 = np.arange(2, 8).reshape(3, 2)

result = np.dot(arr1, arr2)

print("Matrix 1 (5x3):\n", arr1)
print("\nMatrix 2 (3x2):\n", arr2)
print("\nResultant (5x2):\n", result)
