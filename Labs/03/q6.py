# Create a function that asks the user to enter a sentence then writes the sentence to a text file
# named "questions.txt" if it's a question. Handle all possible exceptions as well.

sentence = input("Enter a sentence: ")

def Sent(sentence):
    if sentence[-1] == '?':
         with open('questions.txt', 'w') as file: 
              file.write(sentence)
    else:
         print("Not a question!")

Sent(sentence)
              
