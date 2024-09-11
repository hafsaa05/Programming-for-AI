
class Vehicle:
    def __init__ (self, capacity):
        self.capacity = capacity
    
    def fare(self):
        fare = self.capacity*100
        print(f"Total fare: {fare}")
        
class Bus(Vehicle):
    def __init__ (self, capacity):
        self.capacity = capacity
        
    def fare(self):
        fare = self.capacity*100 + self.capacity*100*0.01
        print(f"Total fare: {fare}")
        
bus = Bus(7)
bus.fare()
        
        
    
