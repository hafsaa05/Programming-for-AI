temperatures = [72, 75, 78, 77, 80]

average_temperature = sum(temperatures) / len(temperatures)
print(f"Average Temperature: {average_temperature:.2f}°F")

highest_temperature = max(temperatures)
lowest_temperature = min(temperatures)
print(f"Highest Temperature: {highest_temperature}°F")
print(f"Lowest Temperature: {lowest_temperature}°F")

sorted_temperatures = sorted(temperatures)
print(f"Temperatures in Ascending Order: {sorted_temperatures}")

day_to_remove = 9
if 0 <= day_to_remove < len(temperatures):
    removed_temperature = temperatures.pop(day_to_remove)
    print(f"Removed Temperature for Day {day_to_remove + 1}: {removed_temperature}°F")
    print(f"Updated Temperature List: {temperatures}")
else:
    print("Invalid day to remove.")
