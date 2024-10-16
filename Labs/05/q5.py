from datetime import datetime

class Vehicle:
    def __init__(self, make, model, rental_price_per_day):
        self.__make = make
        self.__model = model
        self.__rental_price_per_day = rental_price_per_day
        self.__available = True 
    
    def check_availability(self):
        return self.__available
    
    def rent_vehicle(self):
        if self.__available:
            self.__available = False
            return True
        return False
    
    def return_vehicle(self):
        self.__available = True
    
    def calculate_rental_cost(self, rental_days):
        return rental_days * self.__rental_price_per_day
    
    def display_vehicle_details(self):
        status = "Available" if self.__available else "Not Available"
        return (f"Make: {self.__make}, Model: {self.__model}, "
                f"Price/Day: ${self.__rental_price_per_day}, Status: {status}")
    
    def get_rental_price_per_day(self):
        return self.__rental_price_per_day

class Car(Vehicle):
    def __init__(self, make, model, rental_price_per_day):
        super().__init__(make, model, rental_price_per_day)

class SUV(Vehicle):
    def __init__(self, make, model, rental_price_per_day):
        super().__init__(make, model, rental_price_per_day)

class Truck(Vehicle):
    def __init__(self, make, model, rental_price_per_day):
        super().__init__(make, model, rental_price_per_day)

class RentalReservation:
    def __init__(self, customer, vehicle, start_date, end_date):
        self.customer = customer
        self.vehicle = vehicle
        self.start_date = start_date
        self.end_date = end_date
        self.rental_days = (self.end_date - self.start_date).days
        self.total_cost = self.vehicle.calculate_rental_cost(self.rental_days)

    def display_reservation_details(self):
        vehicle_details = self.vehicle.display_vehicle_details()
        return (f"Reservation for {self.customer.get_name()}:\n"
                f"Vehicle: {vehicle_details}\n"
                f"Rental Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}\n"
                f"Total Cost: ${self.total_cost}")

class Customer:
    def __init__(self, name, contact_info):
        self.__name = name
        self.__contact_info = contact_info
        self.rental_history = []
    
    def add_rental_history(self, rental_reservation):
        self.rental_history.append(rental_reservation)

    def display_rental_history(self):
        print(f"Rental history for {self.__name}:")
        if not self.rental_history:
            print("No rentals found.")
        else:
            for reservation in self.rental_history:
                print(reservation.display_reservation_details())

    def get_name(self):
        return self.__name
    
    def get_contact_info(self):
        return self.__contact_info

def display_details(item):
    if isinstance(item, Vehicle):
        print(item.display_vehicle_details())
    elif isinstance(item, RentalReservation):
        print(item.display_reservation_details())

if __name__ == "__main__":
    car = Car("Toyota", "Corolla", 40)
    suv = SUV("Honda", "CR-V", 60)
    truck = Truck("Ford", "F-150", 80)
    
    customer = Customer("Hafsa", "hafsa.rashid@gmail.com")

    # Simulating the rental process
    if car.check_availability():
        start_date = datetime(2023, 9, 1)
        end_date = datetime(2023, 9, 5)
        car.rent_vehicle()
        reservation = RentalReservation(customer, car, start_date, end_date)
        customer.add_rental_history(reservation)

    customer.display_rental_history()

    display_details(car)
    
    display_details(reservation)

    car.return_vehicle()
    print(f"Vehicle available after return: {car.check_availability()}")
