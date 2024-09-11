class BankAccount:
    def __init__(self, account, balance, opening, name):
        self.account = account
        self.balance = balance
        self.opening = opening
        self.name = name

    def deposit(self, amount):
        self.balance += amount
        print(f"{amount} amount deposited. New balance: {self.balance}")

    def withdraw(self, amount):
        if amount > self.balance:
            print("Insufficient balance.")
        else:
            self.balance -= amount
            print(f"{amount} amount withdrawn. New balance: {self.balance}")

    def check_balance(self):
        print(f"Balance: {self.balance}")

acc = BankAccount(64, 1500, "11 Sept 2024", "Hafsa")
acc.deposit(99)      
acc.withdraw(50)     
acc.check_balance()  
