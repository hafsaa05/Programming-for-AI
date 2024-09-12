class Restaurant:
    def __init__(self):
        self.menu_items = {}       
        self.book_table = []       
        self.customer_orders = {}  
        
    def add_item_to_menu(self, item, price):
        self.menu_items[item] = price   
        
    def book_tables(self, table):
        if table in self.book_table:
            print(f"Table {table} is already booked.")
        else:
            self.book_table.append(table)  
            print(f"Table {table} has been booked.")
    
    def customer_order(self, table, order):
        if table in self.book_table:
            self.customer_orders[table] = order  
            print(f"Order for table {table} has been noted.")
        else:
            print(f"Table {table} is not booked. Please book it first.")
    
    def print_menu(self):
        print("Menu Items:")
        for item_name, price in self.menu_items.items():
            print(f"{item_name}: ${price}")
    
    def print_table_reservations(self):
        print("Booked Tables:")
        for table in self.book_table:
            print(f"Table {table}")
    
    def print_customer_orders(self):
        print("Customer Orders:")
        for table, order in self.customer_orders.items():
            print(f"Table {table}: {order}")


res = Restaurant()

res.add_item_to_menu("Biryani", 12)
res.add_item_to_menu("Burger", 17)
res.add_item_to_menu("Shawarma", 8)

res.book_tables(1)
res.book_tables(2)

res.customer_order(1, ["Biryani", "Shawarma"])
res.customer_order(3, ["Burger"]) 

res.print_menu()

res.print_table_reservations()

res.print_customer_orders()
