"""
Python Decorators and Closures Demo
This script demonstrates decorators, closures, and advanced function concepts.
"""

import time
import functools
from datetime import datetime

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result
    return wrapper

def retry_decorator(max_attempts=3, delay=1):
    """Decorator with parameters to retry function execution."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        print(f"Function '{func.__name__}' failed after {max_attempts} attempts")
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator

def log_calls(log_file="function_calls.log"):
    """Decorator to log function calls to a file."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] Called {func.__name__} with args={args}, kwargs={kwargs}\n"
            
            try:
                with open(log_file, "a") as f:
                    f.write(log_entry)
            except Exception as e:
                print(f"Logging failed: {e}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def create_multiplier(factor):
    """Closure example: create a multiplier function."""
    def multiplier(number):
        return number * factor
    return multiplier

def create_counter(initial_value=0):
    """Closure example: create a counter with state."""
    count = initial_value
    
    def counter():
        nonlocal count
        count += 1
        return count
    
    def get_count():
        return count
    
    def reset():
        nonlocal count
        count = initial_value
    
    # Return a dictionary of functions
    return {
        'increment': counter,
        'get': get_count,
        'reset': reset
    }

class CacheDecorator:
    """Class-based decorator for caching function results."""
    
    def __init__(self, max_size=128):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
    
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from arguments
            key = str(args) + str(sorted(kwargs.items()))
            
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                print(f"Cache hit for {func.__name__}")
                return self.cache[key]
            
            # Calculate result
            result = func(*args, **kwargs)
            
            # Add to cache
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
            
            self.cache[key] = result
            self.access_order.append(key)
            print(f"Cache miss for {func.__name__} - result cached")
            return result
        
        wrapper.cache_info = lambda: {
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'hits': len([k for k in self.access_order if k in self.cache])
        }
        wrapper.clear_cache = lambda: self.cache.clear() or self.access_order.clear()
        
        return wrapper

# Example functions using decorators
@timing_decorator
@log_calls("math_operations.log")
def fibonacci_recursive(n):
    """Calculate Fibonacci number recursively (inefficient for demo)."""
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

@CacheDecorator(max_size=50)
def fibonacci_cached(n):
    """Calculate Fibonacci number with caching."""
    if n <= 1:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)

@retry_decorator(max_attempts=3, delay=0.5)
def unreliable_network_call(success_rate=0.3):
    """Simulate an unreliable network call."""
    import random
    if random.random() < success_rate:
        return "Success! Data retrieved."
    else:
        raise ConnectionError("Network request failed")

def demonstrate_decorators():
    """Demonstrate various decorator examples."""
    print("=== Python Decorators and Closures Demo ===")
    
    # Basic decorator usage
    print("\n--- Timing Decorator ---")
    result = fibonacci_recursive(10)
    print(f"Fibonacci(10) = {result}")
    
    # Cached function
    print("\n--- Cached Function ---")
    print(f"Fibonacci cached(30) = {fibonacci_cached(30)}")
    print(f"Fibonacci cached(30) = {fibonacci_cached(30)}")  # Should be cached
    print(f"Cache info: {fibonacci_cached.cache_info()}")
    
    # Retry decorator
    print("\n--- Retry Decorator ---")
    try:
        result = unreliable_network_call(success_rate=0.7)
        print(result)
    except Exception as e:
        print(f"Final error: {e}")
    
    # Closures
    print("\n--- Closures ---")
    double = create_multiplier(2)
    triple = create_multiplier(3)
    
    print(f"Double 5: {double(5)}")
    print(f"Triple 5: {triple(5)}")
    
    # Counter closure
    counter1 = create_counter(10)
    counter2 = create_counter()
    
    print(f"Counter1: {counter1['increment']()}")  # 11
    print(f"Counter1: {counter1['increment']()}")  # 12
    print(f"Counter2: {counter2['increment']()}")  # 1
    print(f"Counter1 current: {counter1['get']()}")  # 12

if __name__ == "__main__":
    demonstrate_decorators()
