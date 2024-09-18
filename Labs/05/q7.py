import re

emails = []
text = '''reach me out on hafsa@gmail.com or amna.gmail.com. Our mail company.gmail.com. handles all recruitment queries. '''
list = re.split(' ',text)

for var in list:
    if(re.search('.com',var)):
        emails.append(var)
        
print(emails)
