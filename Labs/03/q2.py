# Create a program that reads a text file, searches for a specified word or phrase, and replaces
# all occurrences with another word or phrase. Write the modified content back to the file.
# Handle all possible exceptions as well.
try:
    with open(r'search.txt', 'r') as file:
        text = file.read()

    search = input("Enter the word you want to search: ")

    if search in text:
        rep = input("Enter the word to replace: ")

        replaced = text.replace(search, rep)

        with open(r'search.txt', 'w') as file:
            file.write(replaced)
        print(f"All occurrences of '{search}' have been replaced with '{rep}'.")
    else:
        print("Word not found!")

except FileNotFoundError:
    print("Error: The file 'search.txt' was not found.")

except IOError:
    print("Error: An I/O error occurred while accessing the file 'search.txt'.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
