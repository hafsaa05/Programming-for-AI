def dictionary(keys, values):
    dictionary = {} 
    
    for i in range(len(keys)): 
        dictionary[keys[i]] = values[i]
    
    return dictionary

list1 = input("Enter elements for list1: ").split(' ')
list2 = input("Enter elements for list2: ").split(' ')

if len(list1) != len(list2):
    print("Error: The lists must have the same number of elements.")
else:
    print(dictionary(list1, list2)) 
