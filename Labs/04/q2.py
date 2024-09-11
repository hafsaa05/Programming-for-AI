list1 = ["Hello ", "take "]
list2 = ["Dear", "Sir"]
x = []
for var in range(len(list1)):
    for var1 in range(len(list2)):
        x.append(list1[var] + list2[var1])

print(x)
