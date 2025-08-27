"""
Python Metaclasses and Advanced OOP Demo
This script demonstrates metaclasses, advanced inheritance, and design patterns.
"""

import abc
from typing import Any, Dict, List
from enum import Enum, auto

# Metaclass Examples
class SingletonMeta(type):
    """Metaclass that creates singleton instances."""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class ValidatedMeta(type):
    """Metaclass that adds validation to class creation."""
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Validate that all methods have docstrings
        for key, value in namespace.items():
            if callable(value) and not key.startswith('_') and not getattr(value, '__doc__', None):
                raise ValueError(f"Method {key} in class {name} must have a docstring")
        
        return super().__new__(mcs, name, bases, namespace)

class AutoPropertyMeta(type):
    """Metaclass that automatically creates properties for private attributes."""
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Find private attributes and create properties
        for key, value in list(namespace.items()):
            if key.startswith('_') and not key.startswith('__') and not callable(value):
                property_name = key[1:]  # Remove leading underscore
                
                def make_getter(attr_name):
                    def getter(self):
                        return getattr(self, attr_name)
                    return getter
                
                def make_setter(attr_name):
                    def setter(self, value):
                        setattr(self, attr_name, value)
                    return setter
                
                if property_name not in namespace:
                    namespace[property_name] = property(
                        make_getter(key),
                        make_setter(key)
                    )
        
        return super().__new__(mcs, name, bases, namespace)

# Singleton example
class DatabaseConnection(metaclass=SingletonMeta):
    """Database connection using singleton pattern."""
    
    def __init__(self):
        self.connection_string = "database://localhost:5432"
        self.is_connected = False
    
    def connect(self):
        """Connect to the database."""
        if not self.is_connected:
            print(f"Connecting to {self.connection_string}")
            self.is_connected = True
        else:
            print("Already connected")
    
    def disconnect(self):
        """Disconnect from the database."""
        if self.is_connected:
            print("Disconnecting from database")
            self.is_connected = False

# Abstract Base Classes
class Shape(abc.ABC):
    """Abstract base class for shapes."""
    
    @abc.abstractmethod
    def area(self) -> float:
        """Calculate the area of the shape."""
        pass
    
    @abc.abstractmethod
    def perimeter(self) -> float:
        """Calculate the perimeter of the shape."""
        pass
    
    @abc.abstractproperty
    def name(self) -> str:
        """Get the name of the shape."""
        pass

class Rectangle(Shape):
    """Rectangle implementation of Shape."""
    
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def area(self) -> float:
        """Calculate rectangle area."""
        return self.width * self.height
    
    def perimeter(self) -> float:
        """Calculate rectangle perimeter."""
        return 2 * (self.width + self.height)
    
    @property
    def name(self) -> str:
        """Get shape name."""
        return "Rectangle"

class Circle(Shape):
    """Circle implementation of Shape."""
    
    def __init__(self, radius: float):
        self.radius = radius
    
    def area(self) -> float:
        """Calculate circle area."""
        import math
        return math.pi * self.radius ** 2
    
    def perimeter(self) -> float:
        """Calculate circle perimeter."""
        import math
        return 2 * math.pi * self.radius
    
    @property
    def name(self) -> str:
        """Get shape name."""
        return "Circle"

# Multiple Inheritance and Method Resolution Order
class Flyable:
    """Mixin for objects that can fly."""
    
    def fly(self):
        """Make the object fly."""
        return f"{self.__class__.__name__} is flying!"

class Swimmable:
    """Mixin for objects that can swim."""
    
    def swim(self):
        """Make the object swim."""
        return f"{self.__class__.__name__} is swimming!"

class Animal:
    """Base animal class."""
    
    def __init__(self, name: str):
        self.name = name
    
    def speak(self):
        """Make the animal speak."""
        return f"{self.name} makes a sound"

class Duck(Animal, Flyable, Swimmable):
    """Duck class with multiple inheritance."""
    
    def __init__(self, name: str):
        super().__init__(name)
    
    def speak(self):
        """Duck-specific speak method."""
        return f"{self.name} says quack!"

class Fish(Animal, Swimmable):
    """Fish class that can swim."""
    
    def speak(self):
        """Fish don't really speak."""
        return f"{self.name} blows bubbles"

# Design Patterns
class Observer(abc.ABC):
    """Observer interface for observer pattern."""
    
    @abc.abstractmethod
    def update(self, subject: 'Subject', event: str, data: Any = None):
        """Update method called by subject."""
        pass

class Subject:
    """Subject class for observer pattern."""
    
    def __init__(self):
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer):
        """Attach an observer."""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer: Observer):
        """Detach an observer."""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify(self, event: str, data: Any = None):
        """Notify all observers."""
        for observer in self._observers:
            observer.update(self, event, data)

class NewsAgency(Subject):
    """News agency that publishes news."""
    
    def __init__(self):
        super().__init__()
        self._news = ""
    
    def set_news(self, news: str):
        """Set and publish news."""
        self._news = news
        self.notify("news_update", news)
    
    @property
    def news(self):
        """Get current news."""
        return self._news

class NewsChannel(Observer):
    """News channel that receives news updates."""
    
    def __init__(self, name: str):
        self.name = name
        self.latest_news = ""
    
    def update(self, subject: Subject, event: str, data: Any = None):
        """Update with latest news."""
        if event == "news_update":
            self.latest_news = data
            print(f"{self.name} received news: {data}")

# Factory Pattern
class VehicleType(Enum):
    """Vehicle types enumeration."""
    CAR = auto()
    MOTORCYCLE = auto()
    TRUCK = auto()

class Vehicle(abc.ABC):
    """Abstract vehicle class."""
    
    @abc.abstractmethod
    def start_engine(self) -> str:
        """Start the vehicle engine."""
        pass
    
    @abc.abstractmethod
    def get_info(self) -> str:
        """Get vehicle information."""
        pass

class Car(Vehicle):
    """Car implementation."""
    
    def start_engine(self) -> str:
        """Start car engine."""
        return "Car engine started with a gentle purr"
    
    def get_info(self) -> str:
        """Get car info."""
        return "This is a car with 4 wheels"

class Motorcycle(Vehicle):
    """Motorcycle implementation."""
    
    def start_engine(self) -> str:
        """Start motorcycle engine."""
        return "Motorcycle engine started with a roar"
    
    def get_info(self) -> str:
        """Get motorcycle info."""
        return "This is a motorcycle with 2 wheels"

class Truck(Vehicle):
    """Truck implementation."""
    
    def start_engine(self) -> str:
        """Start truck engine."""
        return "Truck engine started with a rumble"
    
    def get_info(self) -> str:
        """Get truck info."""
        return "This is a truck for heavy loads"

class VehicleFactory:
    """Factory for creating vehicles."""
    
    @staticmethod
    def create_vehicle(vehicle_type: VehicleType) -> Vehicle:
        """Create a vehicle based on type."""
        if vehicle_type == VehicleType.CAR:
            return Car()
        elif vehicle_type == VehicleType.MOTORCYCLE:
            return Motorcycle()
        elif vehicle_type == VehicleType.TRUCK:
            return Truck()
        else:
            raise ValueError(f"Unknown vehicle type: {vehicle_type}")

# Auto-property example
class Person(metaclass=AutoPropertyMeta):
    """Person class with auto-generated properties."""
    
    def __init__(self, name: str, age: int):
        self._name = name
        self._age = age
        self._email = ""

def demonstrate_metaclasses():
    """Demonstrate metaclass examples."""
    print("=== Metaclasses Demo ===")
    
    # Singleton pattern
    print("\n--- Singleton Pattern ---")
    db1 = DatabaseConnection()
    db2 = DatabaseConnection()
    
    print(f"db1 is db2: {db1 is db2}")  # Should be True
    
    db1.connect()
    db2.connect()  # Should show "Already connected"
    
    # Auto-property metaclass
    print("\n--- Auto-Property Metaclass ---")
    person = Person("Alice", 30)
    print(f"Name: {person.name}")  # Auto-generated property
    print(f"Age: {person.age}")    # Auto-generated property
    
    person.name = "Alice Smith"
    person.age = 31
    print(f"Updated - Name: {person.name}, Age: {person.age}")

def demonstrate_abstract_classes():
    """Demonstrate abstract base classes."""
    print("\n=== Abstract Base Classes Demo ===")
    
    shapes = [
        Rectangle(5, 3),
        Circle(4)
    ]
    
    for shape in shapes:
        print(f"{shape.name}: Area = {shape.area():.2f}, Perimeter = {shape.perimeter():.2f}")

def demonstrate_multiple_inheritance():
    """Demonstrate multiple inheritance and MRO."""
    print("\n=== Multiple Inheritance Demo ===")
    
    duck = Duck("Donald")
    fish = Fish("Nemo")
    
    print(f"Duck MRO: {Duck.__mro__}")
    
    print(duck.speak())
    print(duck.fly())
    print(duck.swim())
    
    print(fish.speak())
    print(fish.swim())

def demonstrate_observer_pattern():
    """Demonstrate observer design pattern."""
    print("\n=== Observer Pattern Demo ===")
    
    news_agency = NewsAgency()
    
    cnn = NewsChannel("CNN")
    bbc = NewsChannel("BBC")
    fox = NewsChannel("Fox News")
    
    news_agency.attach(cnn)
    news_agency.attach(bbc)
    news_agency.attach(fox)
    
    news_agency.set_news("Breaking: Python 4.0 Released!")
    news_agency.set_news("Update: New metaclass features added")
    
def demonstrate_factory_pattern():
    """Demonstrate factory design pattern."""
    print("\n=== Factory Pattern Demo ===")
    
    vehicle_types = [VehicleType.CAR, VehicleType.MOTORCYCLE, VehicleType.TRUCK]
    
    for vehicle_type in vehicle_types:
        vehicle = VehicleFactory.create_vehicle(vehicle_type)
        print(f"\n{vehicle_type.name}:")
        print(f"  {vehicle.get_info()}")
        print(f"  {vehicle.start_engine()}")

def main():
    """Main demonstration function."""
    print("=== Python Metaclasses and Advanced OOP Demo ===")
    
    demonstrate_metaclasses()
    demonstrate_abstract_classes()
    demonstrate_multiple_inheritance()
    demonstrate_observer_pattern()
    demonstrate_factory_pattern()

if __name__ == "__main__":
    main()
