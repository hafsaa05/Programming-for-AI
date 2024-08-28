def convert(url):
    split= url.split("http://www.")
    if len(split) > 1:
        user_url = split[1].rstrip(".com")
        return f"{user_url}.com"
    else:
        return "Invalid format."

url = input("Enter any URL starting with 'http://www.': ")
print(convert(url))

#another solution
# def convert(url):
#     if url.startswith("http://www."):
#         url = url[11:]
#         return url + ".com"  
#     else:
#         return "Invalid format."

# url = input("Enter any URL starting with'http://www.': ")
# print(convert(url))
