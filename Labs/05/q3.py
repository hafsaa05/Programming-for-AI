class Account:
    def __init__(self):
        self.__account_no = 0
        self.__account_bal = 0.0
        self.__security_code = 0
    
    def set_data(self, account_no, account_bal, security_code):
        self.__account_no = account_no      
        self.__account_bal = account_bal     
        self.__security_code = security_code 
    
    def get_data(self):
        print(f"Account No: {self.__account_no}")
        print(f"Balance: {self.__account_bal} $")
        print(f"Security code: {self.__security_code}")
        
acc = Account() 
acc.set_data(64, 2300, 1357)
acc.get_data()
