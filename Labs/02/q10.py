def build_message(**info):
    message = ""
    for key, value in info.items():
        message += f"{key}: {value}, "
    return message.rstrip(", ") 

name = input("Enter the your name: ")
age = int(input("Enter the your age: "))
city = input("Enter your city: ")
occupation = input("Enter Enter your occupation: ")

info = build_message(Name = name, Age = age, City = city , occupation = occupation)
print(info)
