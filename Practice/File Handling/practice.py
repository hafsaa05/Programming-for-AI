# 'r' -> Open a file for reading (default mode)
# 'w' -> Open a file for writing (overwrites existing content)
# 'a' -> Open a file for appending (content is added to the end)
# 'x' -> Create a new file, will raise an error if the file already exists
# 'b' -> Binary mode (useful for images, videos, etc.)
# 't' -> Text mode (default mode)

# open() -> Open a file
f = open("example.txt", "r")

# f.read() -> Reads the entire content of the file
content = f.read()
print(content)

# f.readline() -> Reads one line from the file
line = f.readline()

# f.readlines() -> Reads all lines in a file and returns them as a list
lines = f.readlines()

# f.write("Text") -> Write content to the file (in 'w' or 'a' mode)
f.write("This is an example text.")

# f.close() -> Always close the file after you're done to free up resources
f.close()

# 'with' automatically closes the file after the block ends
with open("example.txt", "r") as f:
    content = f.read()
    print(content)
