import numpy as np

data_type = [('name', 'U20'), ('height', 'f'), ('class', 'i')]
arr = np.array([('Hafsa', 5.3, 12),
                ('Amna', 5.2, 11),
                ('Ahad', 5.9, 10)], dtype=data_type)
sort = np.sort(arr, order=['class', 'height'])
print(sort)
