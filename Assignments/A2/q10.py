import numpy as np

def normalize(arr):
    mean = np.mean(arr)
    std_dev = np.std(arr)
    normalized_arr = (arr - mean) / std_dev
    return normalized_arr

arr = np.array([2, 4, 6, 8, 10])
normalized_arr = normalize(arr)

print(f'Original: {arr}')
print(f'Normalized Array: {normalized_arr}')
