# Write a program that accepts a word from the user and reversesit using loop. For example,‘Pakistan’ becomes‘natsikaP’.
word = input("Enter a word: ")
rev = ''

for i in range(len(word) - 1, -1, -1):
    rev += word[i]

print("Reversed word:", rev)
