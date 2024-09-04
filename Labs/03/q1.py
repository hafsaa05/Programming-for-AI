# #Write a Python program that reads a text file, counts the number of characters in it, and
# prints the total word count. Handle all possible exceptions as well.
try:
    with open(r'example.txt') as file:
        a = file.read()

    letters = ''.join(filter(str.isalpha, a))
    print("Number of characters:", len(letters))
    print("Number of words:", len(a.split()))

except FileNotFoundError:
    print("Error: The file 'example.txt' was not found.")

except IOError:
    print("Error: An I/O error occurred while accessing the file 'example.txt'.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
