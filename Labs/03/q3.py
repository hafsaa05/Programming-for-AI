try:
    list1 = input("Enter elements for list1 (separated by space): ").split()
    list2 = input("Enter elements for list2 (separated by space): ").split()

    if len(list1) != len(list2):
        raise ValueError("The lists must have the same number of elements.")

    result_dict = dict(zip(list1, list2))
    print("Dictionary created:", result_dict)

    with open('dictionary.txt', 'w') as file:
        for key, value in result_dict.items():
            file.write(f"{key}: {value}\n")

    print("Dictionary has been saved to 'dictionary.txt'.")

except ValueError as ve:
    print(f"ValueError: {ve}")

except IOError:
    print("Error: An I/O error occurred while accessing the file 'dictionary.txt'.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
