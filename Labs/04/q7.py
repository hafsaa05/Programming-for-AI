
class Vehicle:
    def __init__ (self, capacity):
        self.capacity = capacity
    
    def fare(self):
        Fare = self.capacity*100
        return Fare
        
class Bus(Vehicle):
    def __init__ (self, capacity):
        self.capacity = capacity
        
    def bus_fare(self):
        Bus_fare = self.fare() + (self.fare()*0.010)
        print(f"Total fare: {Bus_fare}")
        
bus = Bus(7)
print(bus.fare())
bus.bus_fare()
