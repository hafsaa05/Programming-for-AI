temps = [34, 38, 42, 39, 43, 35, 45, 22, 56, 77, 34, 99]

def avg(t):
    return sum(t) / len(t)

def max_t(t):
    return max(t)

def min_t(t):
    return min(t)

print("Average temperature:", avg(temps))
print("Highest temperature:", max_t(temps))
print("Lowest temperature:", min_t(temps))

temps.sort()
print("Sorted temperatures:", temps)

del temps[4]
print("Temps after removing 5th day's record:", temps)
