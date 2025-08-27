"""
Python Asynchronous Programming Demo
This script demonstrates async/await, coroutines, and concurrent programming.
"""

import asyncio
import aiohttp
import time
import random
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

# Basic async/await examples
async def simple_async_function():
    """Simple async function demonstration."""
    print("Starting async function")
    await asyncio.sleep(1)  # Simulate async operation
    print("Async function completed")
    return "Hello from async!"

async def fetch_data(url: str, delay: float = 1.0) -> Dict[str, Any]:
    """Simulate fetching data from a URL."""
    print(f"Fetching data from {url}")
    await asyncio.sleep(delay)  # Simulate network delay
    
    # Simulate different responses
    responses = {
        "api.example1.com": {"data": "User data", "status": 200},
        "api.example2.com": {"data": "Product data", "status": 200},
        "api.example3.com": {"data": "Order data", "status": 200},
        "slow-api.com": {"data": "Slow response", "status": 200},
    }
    
    return responses.get(url, {"data": "Default data", "status": 404})

async def process_data(data: Dict[str, Any], processing_time: float = 0.5) -> str:
    """Simulate processing fetched data."""
    print(f"Processing data: {data.get('data', 'No data')}")
    await asyncio.sleep(processing_time)
    return f"Processed: {data.get('data', 'No data')}"

# Async context manager
class AsyncDatabaseConnection:
    """Async context manager for database connections."""
    
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.connection = None
    
    async def __aenter__(self):
        print(f"Connecting to async database: {self.db_name}")
        await asyncio.sleep(0.1)  # Simulate connection time
        self.connection = f"connection_to_{self.db_name}"
        return self.connection
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print(f"Closing async database connection: {self.db_name}")
        await asyncio.sleep(0.1)  # Simulate cleanup time
        self.connection = None

# Async iterator
class AsyncNumberGenerator:
    """Async iterator that generates numbers."""
    
    def __init__(self, start: int, end: int, delay: float = 0.1):
        self.start = start
        self.end = end
        self.delay = delay
        self.current = start
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.current >= self.end:
            raise StopAsyncIteration
        
        await asyncio.sleep(self.delay)
        value = self.current
        self.current += 1
        return value

# Producer-Consumer pattern with asyncio
class AsyncQueue:
    """Async queue for producer-consumer pattern."""
    
    def __init__(self, maxsize: int = 10):
        self.queue = asyncio.Queue(maxsize=maxsize)
    
    async def producer(self, name: str, items: List[Any]):
        """Producer coroutine."""
        for item in items:
            await asyncio.sleep(random.uniform(0.1, 0.5))  # Simulate work
            await self.queue.put(item)
            print(f"Producer {name} produced: {item}")
        
        # Signal completion
        await self.queue.put(None)
        print(f"Producer {name} finished")
    
    async def consumer(self, name: str):
        """Consumer coroutine."""
        consumed_items = []
        
        while True:
            item = await self.queue.get()
            
            if item is None:
                # Received termination signal
                await self.queue.put(None)  # Pass signal to other consumers
                break
            
            await asyncio.sleep(random.uniform(0.1, 0.3))  # Simulate processing
            consumed_items.append(item)
            print(f"Consumer {name} consumed: {item}")
            self.queue.task_done()
        
        print(f"Consumer {name} finished. Consumed: {consumed_items}")
        return consumed_items

# Web scraping simulation with aiohttp
async def fetch_url_content(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
    """Fetch content from URL using aiohttp."""
    try:
        # Simulate different URLs with mock responses
        mock_responses = {
            "https://httpbin.org/delay/1": {"title": "Delayed Response", "size": 1024},
            "https://httpbin.org/json": {"title": "JSON Response", "size": 512},
            "https://httpbin.org/html": {"title": "HTML Response", "size": 2048},
        }
        
        # Simulate network delay
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        response_data = mock_responses.get(url, {"title": "Unknown", "size": 0})
        
        return {
            "url": url,
            "status": 200,
            "title": response_data["title"],
            "content_size": response_data["size"]
        }
    
    except Exception as e:
        return {
            "url": url,
            "status": 500,
            "error": str(e),
            "content_size": 0
        }

async def batch_fetch_urls(urls: List[str]) -> List[Dict[str, Any]]:
    """Fetch multiple URLs concurrently."""
    # Note: In a real scenario, you would use aiohttp.ClientSession()
    # For this demo, we'll simulate the session
    class MockSession:
        pass
    
    session = MockSession()
    
    # Create tasks for concurrent execution
    tasks = [fetch_url_content(session, url) for url in urls]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return [result for result in results if not isinstance(result, Exception)]

# Async with threading
def cpu_bound_task(n: int) -> int:
    """CPU-bound task that calculates sum of squares."""
    return sum(i * i for i in range(n))

async def run_cpu_bound_async(numbers: List[int]) -> List[int]:
    """Run CPU-bound tasks asynchronously using thread pool."""
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        tasks = [
            loop.run_in_executor(executor, cpu_bound_task, n)
            for n in numbers
        ]
        
        results = await asyncio.gather(*tasks)
        return results

# Rate limiting with asyncio
class AsyncRateLimiter:
    """Async rate limiter using semaphore."""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.semaphore = asyncio.Semaphore(max_calls)
        self.call_times = []
    
    async def __aenter__(self):
        await self.semaphore.acquire()
        
        # Clean old call times
        current_time = time.time()
        self.call_times = [
            call_time for call_time in self.call_times
            if current_time - call_time < self.time_window
        ]
        
        # If we're at the limit, wait
        if len(self.call_times) >= self.max_calls:
            sleep_time = self.time_window - (current_time - self.call_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.call_times.append(current_time)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.semaphore.release()

async def rate_limited_api_call(api_endpoint: str, rate_limiter: AsyncRateLimiter) -> str:
    """Make a rate-limited API call."""
    async with rate_limiter:
        print(f"Calling API: {api_endpoint}")
        await asyncio.sleep(0.1)  # Simulate API call
        return f"Response from {api_endpoint}"

async def demonstrate_basic_async():
    """Demonstrate basic async/await functionality."""
    print("=== Basic Async/Await Demo ===")
    
    # Simple async function
    result = await simple_async_function()
    print(f"Result: {result}")
    
    # Multiple async operations
    print("\n--- Concurrent Data Fetching ---")
    urls = ["api.example1.com", "api.example2.com", "api.example3.com"]
    
    # Sequential execution
    start_time = time.time()
    sequential_results = []
    for url in urls:
        result = await fetch_data(url, 0.5)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    print(f"Sequential execution took: {sequential_time:.2f} seconds")
    
    # Concurrent execution
    start_time = time.time()
    concurrent_results = await asyncio.gather(*[fetch_data(url, 0.5) for url in urls])
    concurrent_time = time.time() - start_time
    
    print(f"Concurrent execution took: {concurrent_time:.2f} seconds")
    print(f"Speedup: {sequential_time / concurrent_time:.2f}x")

async def demonstrate_async_context_manager():
    """Demonstrate async context managers."""
    print("\n=== Async Context Manager Demo ===")
    
    async with AsyncDatabaseConnection("user_db") as connection:
        print(f"Using connection: {connection}")
        await asyncio.sleep(0.2)  # Simulate database operations

async def demonstrate_async_iterator():
    """Demonstrate async iterators."""
    print("\n=== Async Iterator Demo ===")
    
    print("Async number generation:")
    async for number in AsyncNumberGenerator(1, 6, 0.2):
        print(f"Generated: {number}")

async def demonstrate_producer_consumer():
    """Demonstrate producer-consumer pattern."""
    print("\n=== Producer-Consumer Pattern Demo ===")
    
    queue_system = AsyncQueue(maxsize=5)
    
    # Create producers and consumers
    producer_task = asyncio.create_task(
        queue_system.producer("P1", ["item1", "item2", "item3", "item4"])
    )
    
    consumer_tasks = [
        asyncio.create_task(queue_system.consumer("C1")),
        asyncio.create_task(queue_system.consumer("C2"))
    ]
    
    # Wait for all tasks to complete
    await producer_task
    await asyncio.gather(*consumer_tasks)

async def demonstrate_web_scraping():
    """Demonstrate concurrent web scraping simulation."""
    print("\n=== Concurrent Web Scraping Demo ===")
    
    urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/json",
        "https://httpbin.org/html"
    ]
    
    start_time = time.time()
    results = await batch_fetch_urls(urls)
    end_time = time.time()
    
    print(f"Fetched {len(results)} URLs in {end_time - start_time:.2f} seconds")
    for result in results:
        print(f"  {result['url']}: {result['title']} ({result['content_size']} bytes)")

async def demonstrate_cpu_bound_async():
    """Demonstrate handling CPU-bound tasks with async."""
    print("\n=== CPU-Bound Tasks with Async Demo ===")
    
    numbers = [100000, 200000, 150000, 300000]
    
    start_time = time.time()
    results = await run_cpu_bound_async(numbers)
    end_time = time.time()
    
    print(f"Calculated sum of squares for {len(numbers)} numbers in {end_time - start_time:.2f} seconds")
    for i, (num, result) in enumerate(zip(numbers, results)):
        print(f"  Sum of squares up to {num}: {result}")

async def demonstrate_rate_limiting():
    """Demonstrate rate limiting with async."""
    print("\n=== Rate Limiting Demo ===")
    
    # Create rate limiter: max 3 calls per 2 seconds
    rate_limiter = AsyncRateLimiter(max_calls=3, time_window=2.0)
    
    api_endpoints = [f"api/endpoint{i}" for i in range(6)]
    
    start_time = time.time()
    
    # Make rate-limited API calls
    tasks = [rate_limited_api_call(endpoint, rate_limiter) for endpoint in api_endpoints]
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    
    print(f"Made {len(results)} rate-limited API calls in {end_time - start_time:.2f} seconds")

async def main():
    """Main async function to run all demonstrations."""
    print("=== Python Asynchronous Programming Demo ===")
    
    await demonstrate_basic_async()
    await demonstrate_async_context_manager()
    await demonstrate_async_iterator()
    await demonstrate_producer_consumer()
    await demonstrate_web_scraping()
    await demonstrate_cpu_bound_async()
    await demonstrate_rate_limiting()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
