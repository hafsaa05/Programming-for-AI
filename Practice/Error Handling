# try-except block is used to catch and handle exceptions
try:
    # Code that may raise an error
    f = open("non_existing_file.txt", "r")
except FileNotFoundError:
    # Handle specific FileNotFoundError
    print("File not found.")
except Exception as e:
    # Catch all other exceptions
    print(f"An error occurred: {e}")
finally:
    # This block will always run (useful for closing resources)
    print("This will always execute.")

#example
try:
    with open("example.txt", "r") as f:
        content = f.read()
        print(content)
except FileNotFoundError:
    print("Error: The file was not found.")
except PermissionError:
    print("Error: You don't have permission to access this file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Common Exceptions:
# FileNotFoundError: Raised when trying to open a non-existing file.
# IOError: Raised for input/output operations failures.
# PermissionError: Raised when the user does not have permission to perform a file operation.
# ValueError: Raised when a function receives an argument of the right type but inappropriate value.
# ZeroDivisionError: Raised when dividing by zero.

with open("newfile.txt", "w") as f:
    f.write("This is a new file.\n")
    f.write("Appending another line.\n")
# os.remove() -> Delete a file
import os
os.remove("newfile.txt") 

# os.path.exists() -> Check if a file exists
if os.path.exists("example.txt"):
    print("File exists.")
else:
    print("File does not exist.")
