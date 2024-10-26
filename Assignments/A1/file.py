class Animal:
    def __init__(self, name, age, habitat):
        self.name = name
        self.age = age
        self.habitat = habitat
        self.is_available = True

    def set_availability(self, available):
        self.is_available = available

    def display(self):
        availability_status = "Available for viewing" if self.is_available else "In quarantine"
        return f"Name: {self.name}, Age: {self.age}, Habitat: {self.habitat}, Status: {availability_status}"

class Mammal(Animal):
    def __init__(self, name, age, habitat, fur_length, diet_type):
        super().__init__(name, age, habitat)
        self.fur_length = fur_length
        self.diet_type = diet_type

    def display(self):
        return (super().display() +
                f", Fur Length: {self.fur_length}, Diet Type: {self.diet_type}")

class Bird(Animal):
    def __init__(self, name, age, habitat, wingspan, flight_altitude):
        super().__init__(name, age, habitat)
        self.wingspan = wingspan
        self.flight_altitude = flight_altitude

    def display(self):
        return (super().display() +
                f", Wingspan: {self.wingspan}, Flight Altitude: {self.flight_altitude}")

class Reptile(Animal):
    def __init__(self, name, age, habitat, scale_pattern, venomous_status):
        super().__init__(name, age, habitat)
        self.scale_pattern = scale_pattern
        self.venomous_status = venomous_status

    def display(self):
        return (super().display() +
                f", Scale Pattern: {self.scale_pattern}, Venomous Status: {self.venomous_status}")

mammal = Mammal("Lioness", 5, "Grassland", "Short", "Carnivore")
bird = Bird("Penguin", 3, "Antarctic", "0.5 meters", "0 meters")
reptile = Reptile("Turtle", 10, "Tropical Beach", "Rough", "Non-venomous")

print(mammal.display())
print(bird.display())
print(reptile.display())

mammal.set_availability(False)
bird.set_availability(True)
reptile.set_availability(False)  

print("\nAfter availability status changes:")
print(mammal.display())
print(bird.display())
print(reptile.display())
