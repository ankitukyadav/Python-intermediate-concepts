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
    
        def setUp(self):
        """Set up test fixtures before each test method."""
        self.calc = Calculator()
    
    def test_add(self):
        """Test addition functionality."""
        self.assertEqual(self.calc.add(2, 3), 5.0)
        self.assertEqual(self.calc.add(-1, 1), 0.0)
        self.assertEqual(self.calc.add(0, 0), 0.0)
        self.assertAlmostEqual(self.calc.add(0.1, 0.2), 0.3, places=7)
    
    def test_subtract(self):
        """Test subtraction functionality."""
        self.assertEqual(self.calc.subtract(5, 3), 2.0)
        self.assertEqual(self.calc.subtract(1, 1), 0.0)
        self.assertEqual(self.calc.subtract(-1, -1), 0.0)
    
    def test_multiply(self):
        """Test multiplication functionality."""
        self.assertEqual(self.calc.multiply(3, 4), 12.0)
        self.assertEqual(self.calc.multiply(-2, 3), -6.0)
        self.assertEqual(self.calc.multiply(0, 100), 0.0)
    
    def test_divide(self):
        """Test division functionality."""
        self.assertEqual(self.calc.divide(10, 2), 5.0)
        self.assertEqual(self.calc.divide(7, 2), 3.5)
        
        # Test division by zero
        with self.assertRaises(ValueError) as context:
            self.calc.divide(10, 0)
        
        self.assertEqual(str(context.exception), "Cannot divide by zero")
    
    def test_divide_edge_cases(self):
        """Test division edge cases."""
        self.assertEqual(self.calc.divide(0, 5), 0.0)
        self.assertAlmostEqual(self.calc.divide(1, 3), 0.3333333333333333)

class TestBankAccount(unittest.TestCase):
    """Unit tests for BankAccount class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.account = BankAccount(100.0)
    
    def test_initial_balance(self):
        """Test initial balance setting."""
        new_account = BankAccount()
        self.assertEqual(new_account.get_balance(), 0.0)
        
        account_with_balance = BankAccount(50.0)
        self.assertEqual(account_with_balance.get_balance(), 50.0)
    
    def test_deposit(self):
        """Test deposit functionality."""
        initial_balance = self.account.get_balance()
        self.account.deposit(50.0)
        self.assertEqual(self.account.get_balance(), initial_balance + 50.0)
        
        # Test transaction history
        self.assertIn("Deposit: +$50.0", self.account.transaction_history)
    
    def test_deposit_invalid_amount(self):
        """Test deposit with invalid amounts."""
        with self.assertRaises(ValueError):
            self.account.deposit(0)
        
        with self.assertRaises(ValueError):
            self.account.deposit(-10)
    
    def test_withdraw(self):
        """Test withdrawal functionality."""
        initial_balance = self.account.get_balance()
        self.account.withdraw(30.0)
        self.assertEqual(self.account.get_balance(), initial_balance - 30.0)
        
        # Test transaction history
        self.assertIn("Withdrawal: -$30.0", self.account.transaction_history)
    
    def test_withdraw_insufficient_funds(self):
        """Test withdrawal with insufficient funds."""
        with self.assertRaises(ValueError) as context:
            self.account.withdraw(200.0)
        
        self.assertEqual(str(context.exception), "Insufficient funds")
    
    def test_withdraw_invalid_amount(self):
        """Test withdrawal with invalid amounts."""
        with self.assertRaises(ValueError):
            self.account.withdraw(0)
        
        with self.assertRaises(ValueError):
            self.account.withdraw(-10)

# Mock Testing Examples
class EmailService:
    """Email service for testing with mocks."""
    
    def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send an email (would normally connect to email server)."""
        # In real implementation, this would connect to email server
        logger.info(f"Sending email to {to}: {subject}")
        return True

class NotificationService:
    """Service that uses EmailService."""
    
    def __init__(self, email_service: EmailService):
        self.email_service = email_service
    
    def send_welcome_email(self, user_email: str, username: str) -> bool:
        """Send welcome email to new user."""
        subject = "Welcome to our service!"
        body = f"Hello {username}, welcome to our amazing service!"
        
        try:
            return self.email_service.send_email(user_email, subject, body)
        except Exception as e:
            logger.error(f"Failed to send welcome email: {e}")
            return False

class TestNotificationService(unittest.TestCase):
    """Test NotificationService with mocks."""
    
    def setUp(self):
        """Set up test fixtures with mocks."""
        self.mock_email_service = Mock(spec=EmailService)
        self.notification_service = NotificationService(self.mock_email_service)
    
    def test_send_welcome_email_success(self):
        """Test successful welcome email sending."""
        # Configure mock to return True
        self.mock_email_service.send_email.return_value = True
        
        result = self.notification_service.send_welcome_email("user@example.com", "John")
        
        # Assert the result
        self.assertTrue(result)
        
        # Assert the mock was called correctly
        self.mock_email_service.send_email.assert_called_once_with(
            "user@example.com",
            "Welcome to our service!",
            "Hello John, welcome to our amazing service!"
        )
    
    def test_send_welcome_email_failure(self):
        """Test welcome email sending failure."""
        # Configure mock to raise an exception
        self.mock_email_service.send_email.side_effect = Exception("Network error")
        
        result = self.notification_service.send_welcome_email("user@example.com", "John")
        
        # Assert the result
        self.assertFalse(result)
        
        # Assert the mock was called
        self.mock_email_service.send_email.assert_called_once()

# Patch decorator examples
class FileProcessor:
    """File processor for testing with patches."""
    
    def read_file(self, filename: str) -> str:
        """Read content from file."""
        with open(filename, 'r') as f:
            return f.read()
    
    def process_file(self, filename: str) -> int:
        """Process file and return word count."""
        content = self.read_file(filename)
        words = content.split()
        return len(words)

class TestFileProcessor(unittest.TestCase):
    """Test FileProcessor with patches."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = FileProcessor()
    
    @patch('builtins.open')
    def test_read_file(self, mock_open):
        """Test file reading with mocked open."""
        # Configure mock
        mock_file = MagicMock()
        mock_file.read.return_value = "Hello world from file"
        mock_open.return_value.__enter__.return_value = mock_file
        
        result = self.processor.read_file("test.txt")
        
        self.assertEqual(result, "Hello world from file")
        mock_open.assert_called_once_with("test.txt", 'r')
    
    @patch.object(FileProcessor, 'read_file')
    def test_process_file(self, mock_read_file):
        """Test file processing with mocked read_file."""
        # Configure mock
        mock_read_file.return_value = "Hello world this is a test file"
        
        result = self.processor.process_file("test.txt")
        
        self.assertEqual(result, 8)  # 8 words
        mock_read_file.assert_called_once_with("test.txt")

# Custom Test Suite
def create_test_suite():
    """Create a custom test suite."""
    suite = unittest.TestSuite()
    
    # Add specific tests
    suite.addTest(TestCalculator('test_add'))
    suite.addTest(TestCalculator('test_divide'))
    suite.addTest(TestBankAccount('test_deposit'))
    suite.addTest(TestBankAccount('test_withdraw'))
    
    return suite

# Debugging utilities
class DebugUtils:
    """Utilities for debugging."""
    
    @staticmethod
    def print_stack_trace():
        """Print current stack trace."""
        print("=== Stack Trace ===")
        traceback.print_stack()
    
    @staticmethod
    def debug_variables(**kwargs):
        """Print debug information for variables."""
        print("=== Debug Variables ===")
        for name, value in kwargs.items():
            print(f"{name}: {value} (type: {type(value).__name__})")
    
    @staticmethod
    def memory_usage():
        """Get memory usage information."""
        import sys
        import gc
        
        print("=== Memory Usage ===")
        print(f"Reference count for current frame: {sys.getrefcount(sys._getframe())}")
        print(f"Garbage collection counts: {gc.get_count()}")
        
        # Get size of some objects
        objects = gc.get_objects()
        print(f"Total objects in memory: {len(objects)}")

# Performance testing
class PerformanceTest:
    """Performance testing utilities."""
    
    @staticmethod
    def time_function(func, *args, **kwargs):
        """Time a function execution."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"Function {func.__name__} took {execution_time:.6f} seconds")
        return result, execution_time
    
    @staticmethod
    def compare_functions(functions, *args, **kwargs):
        """Compare execution times of multiple functions."""
        results = {}
        
        for name, func in functions.items():
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            execution_time = end_time - start_time
            results[name] = {
                'result': result,
                'time': execution_time
            }
        
        # Print comparison
        print("=== Performance Comparison ===")
        sorted_results = sorted(results.items(), key=lambda x: x[1]['time'])
        
        for name, data in sorted_results:
            print(f"{name}: {data['time']:.6f} seconds")
        
        return results

def demonstrate_unit_testing():
    """Demonstrate unit testing."""
    print("=== Unit Testing Demo ===")
    
    # Run specific test class
    print("\n--- Running Calculator Tests ---")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCalculator)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

def demonstrate_mocking():
    """Demonstrate mocking in tests."""
    print("\n=== Mocking Demo ===")
    
    # Run mock tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNotificationService)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

def demonstrate_debugging():
    """Demonstrate debugging techniques."""
    print("\n=== Debugging Demo ===")
    
    # Debug function calls
    print("\n--- Debug Function Calls ---")
    result = fibonacci_recursive(5)
    print(f"Fibonacci(5) = {result}")
    
    # Debug data processing
    print("\n--- Debug Data Processing ---")
    test_data = ["hello", 42, 3.14, None, [1, 2, 3]]
    processed = process_data(test_data)
    print(f"Processed data: {processed}")
    
    # Debug utilities
    print("\n--- Debug Utilities ---")
    DebugUtils.debug_variables(
        result=result,
        processed=processed,
        test_data=test_data
    )

def demonstrate_performance_testing():
    """Demonstrate performance testing."""
    print("\n=== Performance Testing Demo ===")
    
    # Time a single function
    print("\n--- Single Function Timing ---")
    calc = Calculator()
    result, exec_time = PerformanceTest.time_function(calc.multiply, 123, 456)
    print(f"Result: {result}")
    
    # Compare different implementations
    print("\n--- Function Comparison ---")
    
    def list_comprehension_squares(n):
        return [i**2 for i in range(n)]
    
    def loop_squares(n):
        result = []
        for i in range(n):
            result.append(i**2)
        return result
    
    def map_squares(n):
        return list(map(lambda x: x**2, range(n)))
    
    functions = {
        'list_comprehension': list_comprehension_squares,
        'loop': loop_squares,
        'map': map_squares
    }
    
    PerformanceTest.compare_functions(functions, 10000)

def demonstrate_doctest():
    """Demonstrate doctest functionality."""
    print("\n=== Doctest Demo ===")
    
    # Run doctests for Calculator class
    print("Running doctests for Calculator...")
    doctest.run_docstring_examples(Calculator, globals(), verbose=True)

def run_all_tests():
    """Run all tests in the module."""
    print("\n=== Running All Tests ===")
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    suite = loader.discover('.', pattern='*test*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def main():
    """Main demonstration function."""
    print("=== Python Testing and Debugging Demo ===")
    
    # Set logging level for demo
    logging.getLogger().setLevel(logging.INFO)
    
    demonstrate_unit_testing()
    demonstrate_mocking()
    demonstrate_debugging()
    demonstrate_performance_testing()
    demonstrate_doctest()
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()
