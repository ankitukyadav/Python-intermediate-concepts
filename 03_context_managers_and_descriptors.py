"""
Python Context Managers and Descriptors Demo
This script demonstrates context managers, descriptors, and advanced Python features.
"""

import time
import sqlite3
from contextlib import contextmanager
from typing import Any

class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, description="Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"Starting {self.description}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type is not None:
            print(f"{self.description} failed after {duration:.4f} seconds")
            print(f"Exception: {exc_type.__name__}: {exc_val}")
            return False  # Don't suppress the exception
        else:
            print(f"{self.description} completed in {duration:.4f} seconds")
        
        return False
    
    @property
    def duration(self):
        """Get the duration of the timed operation."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

class FileManager:
    """Context manager for safe file operations."""
    
    def __init__(self, filename, mode='r', encoding='utf-8'):
        self.filename = filename
        self.mode = mode
        self.encoding = encoding
        self.file = None
    
    def __enter__(self):
        try:
            self.file = open(self.filename, self.mode, encoding=self.encoding)
            print(f"Opened file: {self.filename}")
            return self.file
        except Exception as e:
            print(f"Failed to open file {self.filename}: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
            print(f"Closed file: {self.filename}")
        
        if exc_type is not None:
            print(f"Exception occurred while working with {self.filename}: {exc_val}")
        
        return False

class DatabaseConnection:
    """Context manager for database operations with transaction support."""
    
    def __init__(self, db_name):
        self.db_name = db_name
        self.connection = None
        self.cursor = None
    
    def __enter__(self):
        self.connection = sqlite3.connect(self.db_name)
        self.cursor = self.connection.cursor()
        print(f"Connected to database: {self.db_name}")
        return self.cursor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.connection.rollback()
            print("Transaction rolled back due to exception")
        else:
            self.connection.commit()
            print("Transaction committed successfully")
        
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("Database connection closed")
        
        return False

@contextmanager
def temporary_directory():
    """Context manager using contextlib.contextmanager decorator."""
    import tempfile
    import shutil
    import os
    
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")

@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout temporarily."""
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        yield
    finally:
        sys.stdout = old_stdout

# Descriptor Examples
class ValidatedAttribute:
    """Descriptor for validated attributes."""
    
    def __init__(self, validator=None, default=None):
        self.validator = validator
        self.default = default
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name, self.default)
    
    def __set__(self, obj, value):
        if self.validator and not self.validator(value):
            raise ValueError(f"Invalid value for {self.name}: {value}")
        obj.__dict__[self.name] = value
    
    def __delete__(self, obj):
        if self.name in obj.__dict__:
            del obj.__dict__[self.name]

class TypedAttribute:
    """Descriptor that enforces type checking."""
    
    def __init__(self, expected_type, default=None):
        self.expected_type = expected_type
        self.default = default
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name, self.default)
    
    def __set__(self, obj, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f"{self.name} must be of type {self.expected_type.__name__}, "
                          f"got {type(value).__name__}")
        obj.__dict__[self.name] = value

class LoggedAttribute:
    """Descriptor that logs attribute access."""
    
    def __init__(self, default=None):
        self.default = default
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.name, self.default)
        print(f"Getting {self.name}: {value}")
        return value
    
    def __set__(self, obj, value):
        print(f"Setting {self.name}: {value}")
        obj.__dict__[self.name] = value

# Example class using descriptors
class Person:
    """Example class demonstrating descriptor usage."""
    
    # Type-checked attributes
    name = TypedAttribute(str, "Unknown")
    age = TypedAttribute(int, 0)
    
    # Validated attributes
    email = ValidatedAttribute(
        validator=lambda x: "@" in x and "." in x,
        default=""
    )
    
    # Logged attribute
    salary = LoggedAttribute(0)
    
    def __init__(self, name, age, email="", salary=0):
        self.name = name
        self.age = age
        self.email = email
        self.salary = salary
    
    def __repr__(self):
        return f"Person(name='{self.name}', age={self.age}, email='{self.email}', salary={self.salary})"

def demonstrate_context_managers():
    """Demonstrate various context manager examples."""
    print("=== Context Managers Demo ===")
    
    # Timer context manager
    print("\n--- Timer Context Manager ---")
    with Timer("Heavy computation"):
        # Simulate some work
        sum(i**2 for i in range(100000))
    
    # File manager context manager
    print("\n--- File Manager Context Manager ---")
    with FileManager("test_file.txt", "w") as f:
        f.write("Hello, Context Managers!\n")
        f.write("This file will be properly closed.\n")
    
    with FileManager("test_file.txt", "r") as f:
        content = f.read()
        print(f"File content: {content.strip()}")
    
    # Database context manager
    print("\n--- Database Context Manager ---")
    with DatabaseConnection("test.db") as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT
            )
        """)
        cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", 
                      ("Alice", "alice@example.com"))
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()
        print(f"Users in database: {users}")
    
    # Temporary directory context manager
    print("\n--- Temporary Directory Context Manager ---")
    with temporary_directory() as temp_dir:
        import os
        test_file = os.path.join(temp_dir, "temp_file.txt")
        with open(test_file, "w") as f:
            f.write("Temporary file content")
        print(f"Created file in temp directory: {test_file}")
    
    # Suppress stdout context manager
    print("\n--- Suppress Stdout Context Manager ---")
    print("This will be printed")
    with suppress_stdout():
        print("This will NOT be printed")
        print("Neither will this")
    print("This will be printed again")

def demonstrate_descriptors():
    """Demonstrate descriptor examples."""
    print("\n=== Descriptors Demo ===")
    
    # Create person instances
    try:
        person1 = Person("Alice", 30, "alice@example.com", 50000)
        print(f"Created: {person1}")
        
        # Demonstrate type checking
        print("\n--- Type Checking ---")
        person1.age = 31  # Valid
        print(f"Updated age: {person1.age}")
        
        try:
            person1.age = "thirty-two"  # Invalid - should raise TypeError
        except TypeError as e:
            print(f"Type error caught: {e}")
        
        # Demonstrate validation
        print("\n--- Validation ---")
        person1.email = "alice.new@company.com"  # Valid
        print(f"Updated email: {person1.email}")
        
        try:
            person1.email = "invalid-email"  # Invalid - should raise ValueError
        except ValueError as e:
            print(f"Validation error caught: {e}")
        
        # Demonstrate logging
        print("\n--- Logged Attribute ---")
        person1.salary = 55000  # This will be logged
        current_salary = person1.salary  # This will also be logged
        
    except Exception as e:
        print(f"Error creating person: {e}")

def cleanup_demo_files():
    """Clean up demonstration files."""
    import os
    files_to_remove = ["test_file.txt", "test.db"]
    
    for filename in files_to_remove:
        try:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Removed {filename}")
        except Exception as e:
            print(f"Error removing {filename}: {e}")

def main():
    """Main demonstration function."""
    print("=== Python Context Managers and Descriptors Demo ===")
    
    demonstrate_context_managers()
    demonstrate_descriptors()
    
    print("\n--- Cleanup ---")
    cleanup_demo_files()

if __name__ == "__main__":
    main()
