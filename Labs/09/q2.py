import matplotlib.pyplot as plt

city_names = ["Tokyo", "Delhi", "Shanghai", "Sao Paulo", "Mumbai", 
              "Cairo", "Dhaka", "Mexico City", "Beijing", "Osaka"]
city_populations = [13929286, 30290936, 26317104, 12325232, 20500000, 
                    9500000, 21480000, 9209944, 21540000, 8839469]

plt.figure(figsize=(10, 6))
plt.barh(city_names, city_populations, color='lightgreen')
plt.xlabel("Population (in millions)")
plt.ylabel("Cities")
plt.title("City Populations Around the World")

plt.show()
