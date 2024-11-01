import numpy as np

arr1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
arr2 = np.array([2,4,6])

sol = np.linalg.solve(arr1, arr2)
print("Solution:", sol)
