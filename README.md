# Intermediate Python Concepts

This directory contains advanced Python programming examples that build upon the basic concepts. These scripts demonstrate intermediate to advanced Python features and patterns.

## 📚 Contents

### 1. Decorators and Closures (`01_decorators_and_closures.py`)
- Function decorators with and without parameters
- Class-based decorators
- Closures and lexical scoping
- Practical decorator examples (timing, logging, caching, retry)

### 2. Generators and Iterators (`02_generators_and_iterators.py`)
- Custom iterator classes
- Generator functions and expressions
- Advanced generator features (send, throw, close)
- Generator pipelines for data processing
- Memory-efficient data processing

### 3. Context Managers and Descriptors (`03_context_managers_and_descriptors.py`)
- Custom context managers (`__enter__`, `__exit__`)
- Context managers using `contextlib`
- Descriptor protocol (`__get__`, `__set__`, `__delete__`)
- Property descriptors and validation
- Practical examples (file handling, database connections)

### 4. Metaclasses and Advanced OOP (`04_metaclasses_and_advanced_oop.py`)
- Metaclass creation and usage
- Abstract base classes (ABC)
- Multiple inheritance and Method Resolution Order (MRO)
- Design patterns (Singleton, Observer, Factory)
- Advanced inheritance techniques

### 5. Asynchronous Programming (`05_async_programming.py`)
- `async`/`await` syntax
- Coroutines and event loops
- Async context managers and iterators
- Concurrent programming with `asyncio`
- Producer-consumer patterns
- Rate limiting and performance optimization

### 6. Data Structures and Algorithms (`06_data_structures_and_algorithms.py`)
- Custom data structure implementations
- Linked lists, stacks, queues
- Binary search trees and graphs
- Sorting algorithms (bubble, quick, merge)
- Search algorithms (linear, binary)
- Dynamic programming examples

### 7. Testing and Debugging (`07_testing_and_debugging.py`)
- Unit testing with `unittest`
- Mocking and patching
- Test fixtures and test suites
- Debugging decorators and utilities
- Performance testing and profiling
- Doctest integration

### 8. Advanced Modules and Packages (`08_advanced_modules_and_packages.py`)
- Dynamic module creation
- Module introspection and inspection
- Custom import hooks
- Plugin systems and architecture
- Configuration management
- Lazy loading patterns

## 🎯 Learning Objectives

These examples will help you understand:
- **Advanced Function Concepts**: Decorators, closures, and higher-order functions
- **Memory Management**: Generators, iterators, and lazy evaluation
- **Object-Oriented Design**: Metaclasses, descriptors, and design patterns
- **Concurrent Programming**: Async/await, coroutines, and parallel processing
- **Code Quality**: Testing, debugging, and performance optimization
- **Software Architecture**: Modules, packages, and plugin systems

## 📖 Prerequisites
- Solid understanding of basic Python concepts
- Familiarity with object-oriented programming
- Basic knowledge of Python's standard library

## 🔧 Dependencies
Most examples use only Python's standard library. Some scripts may require:
- aiohttp (for async examples - can be simulated without it)
- Standard library modules: unittest, asyncio, collections, etc.

## 💡 Tips for Learning
1.Run the examples: Execute each script to see the concepts in action
2.Modify the code: Experiment with different parameters and scenarios
3.Read the comments: Each example is thoroughly documented
4.Practice: Try implementing similar patterns in your own projects
5.Combine concepts: Many real-world applications use multiple patterns together

## 🚀 How to Run

Each script is self-contained and can be run independently:

```bash
python intermediate_python_concepts/01_decorators_and_closures.py
python intermediate_python_concepts/02_generators_and_iterators.py
# ... and so on
