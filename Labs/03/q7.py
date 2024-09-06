

with open ("replacement_needed.txt", 'r+') as file:
    
    content = file.read()
    print(content)
    replace = input("Enter the character you want to replace: ")
    replacement = input("Enter the character you want to replace it with: ")
    for character in content:
        if character == replace:
            new_content = content.replace(replace, replacement)
    
    with open ("replacement_needed.txt", 'w') as file2:        
        file2.write(new_content)
        print(new_content)
        