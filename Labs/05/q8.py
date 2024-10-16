import re

text = '''
This is a sample text containing variously formatted dates.
12/09/2023
2023-09-12
Sep 12, 2023
'''

def extract_dates(text):
    # Corrected regex pattern
    date_pattern = r'(\b\d{2}/\d{2}/\d{4}\b)|(\b\d{4}-\d{2}-\d{2}\b)|(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}, \d{4}\b)'
    
    dates = re.findall(date_pattern, text)
    
    # Flattening the list of tuples
    extracted_dates = []
    for match in dates:
        for date in match:
            if date:  # Check if the date is not an empty string
                extracted_dates.append(date)
                
    return extracted_dates

dates = extract_dates(text)

print("All dates:")
for date in dates:
    print(date)
