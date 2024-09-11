class Account:
    def __init__(self, account_no, account_bal, security_code):
        self.__account_no = account_no      
        self.__account_bal = account_bal     
        self.__security_code = security_code 
    
    def info(self):
        print(f"Account No: {self.__account_no}")
        print(f"Balance: {self.__account_bal} $")
        print(f"Security code: {self.__security_code}")
        
acc = Account(64, 2300, "1234") 
acc.info()
