"""
Python Testing and Debugging Demo
This script demonstrates unit testing, debugging techniques, and code quality tools.
"""

import unittest
import doctest
import logging
import traceback
import time
import functools
from typing import List, Any
from unittest.mock import Mock, patch, MagicMock

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Example classes and functions to test
class Calculator:
    """A simple calculator class for demonstration."""
    
    def add(self, a: float, b: float) -> float:
        """
        Add two numbers.
        
        >>> calc = Calculator()
        >>> calc.add(2, 3)
        5.0
        >>> calc.add(-1, 1)
        0.0
        """
        logger.debug(f"Adding {a} + {b}")
        return float(a + b)
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        logger.debug(f"Subtracting {a} - {b}")
        return float(a - b)
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        logger.debug(f"Multiplying {a} * {b}")
        return float(a * b)
    
    def divide(self, a: float, b: float) -> float:
        """
        Divide a by b.
        
        >>> calc = Calculator()
        >>> calc.divide(10, 2)
        5.0
        >>> calc.divide(1, 0)
        Traceback (most recent call last):
        ...
        ValueError: Cannot divide by zero
        """
        if b == 0:
            logger.error("Attempted division by zero")
            raise ValueError("Cannot divide by zero")
        
        logger.debug(f"Dividing {a} / {b}")
        return float(a / b)

class BankAccount:
    """Bank account class for testing."""
    
    def __init__(self, initial_balance: float = 0.0):
        self.balance = initial_balance
        self.transaction_history = []
    
    def deposit(self, amount: float) -> None:
        """Deposit money to account."""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        
        self.balance += amount
        self.transaction_history.append(f"Deposit: +${amount}")
        logger.info(f"Deposited ${amount}. New balance: ${self.balance}")
    
    def withdraw(self, amount: float) -> None:
        """Withdraw money from account."""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        
        self.balance -= amount
        self.transaction_history.append(f"Withdrawal: -${amount}")
        logger.info(f"Withdrew ${amount}. New balance: ${self.balance}")
    
    def get_balance(self) -> float:
        """Get current balance."""
        return self.balance

# Debugging decorators
def debug_calls(func):
    """Decorator to debug function calls."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned: {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise
    return wrapper

def timing_debug(func):
    """Decorator to measure and log execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

# Example functions with debugging
@debug_calls
@timing_debug
def fibonacci_recursive(n: int) -> int:
    """Calculate Fibonacci number recursively (for debugging demo)."""
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

@debug_calls
def process_data(data: List[Any]) -> List[Any]:
    """Process a list of data with potential errors."""
    result = []
    for i, item in enumerate(data):
        try:
            if isinstance(item, str):
                result.append(item.upper())
            elif isinstance(item, (int, float)):
                result.append(item * 2)
            else:
                logger.warning(f"Unknown data type at index {i}: {type(item)}")
                result.append(str(item))
        except Exception as e:
            logger.error(f"Error processing item at index {i}: {e}")
            result.append(None)
    
    return result

# Unit Tests
class TestCalculator(unittest.TestCase):
    """Unit tests for Calculator class."""
    
    def setUp(
