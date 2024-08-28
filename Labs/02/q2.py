#Write a Python function to check if the last letter of user input string is a vowel or a consonant.
def last_letter(word):
    vowels = 'aeiouAEIOU'
    
    if word and word.isalpha():
        last_char = word[-1]
        if last_char in vowels:
            return "Vowel"
        else:
            return "Consonant"
    else:
        return "Invalid input!"

word = input(("Enter a word: "))
print(last_letter(word))
