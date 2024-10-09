# Write a NumPy program to create a random array with 1000 elements and compute the
# average, variance, standard deviation of the array elements. Save all the results in text file
# created at runtime.

import numpy as np

arr = np.random.rand(1000)

avg, var, std_dev = np.mean(arr), np.var(arr), np.std(arr)

with open("stats_file.txt", 'w') as f:
    f.write(f"Average: {avg}\nVariance: {var}\nStandard Deviation: {std_dev}\n")

print("Results saved to stats_file.txt")
