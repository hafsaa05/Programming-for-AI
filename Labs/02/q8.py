# Dictionary of currency exchange rates
rates = {
    'Euro': 0.93,
    'Dollar': 1.0,
    'PkR': 280.0,
    'INR': 82.0,
    'Yen': 139.0
}

cur = input("Enter the currency to convert (Euro, Dollar, PkR, INR, Yen): ")
amount = float(input("Enter the amount: "))
convert = input("Enter the currency to be converted to (Euro, Dollar, PkR, INR, Yen): ")

if cur in rates and convert in rates:
    converted_amount = (amount / rates[cur]) * rates[convert]
    print(f"{amount} {cur} is equal to {converted_amount:.2f} {convert}.")
else:
    print("Invalid currency entered. Please try again.")
