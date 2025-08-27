"""
Python Advanced Modules and Packages Demo
This script demonstrates module creation, package structure, and advanced import techniques.
"""

import sys
import os
import importlib
import pkgutil
from types import ModuleType
from typing import Any, Dict, List
import inspect

# Dynamic module creation
def create_dynamic_module(name: str, code: str) -> ModuleType:
    """Create a module dynamically from code string."""
    module = ModuleType(name)
    exec(code, module.__dict__)
    sys.modules[name] = module
    return module

# Module introspection utilities
class ModuleInspector:
    """Utilities for inspecting modules and packages."""
    
    @staticmethod
    def get_module_info(module) -> Dict[str, Any]:
        """Get comprehensive information about a module."""
        info = {
            'name': getattr(module, '__name__', 'Unknown'),
            'file': getattr(module, '__file__', 'Unknown'),
            'doc': getattr(module, '__doc__', 'No documentation'),
            'package': getattr(module, '__package__', None),
            'version': getattr(module, '__version__', 'Unknown'),
        }
        
        # Get all attributes
        attributes = {}
        functions = {}
        classes = {}
        
        for name in dir(module):
            if not name.startswith('_'):
                obj = getattr(module, name)
                
                if inspect.isfunction(obj):
                    functions[name] = {
                        'doc': obj.__doc__,
                        'signature': str(inspect.signature(obj)) if hasattr(inspect, 'signature') else 'N/A'
                    }
                elif inspect.isclass(obj):
                    classes[name] = {
                        'doc': obj.__doc__,
                        'methods': [method for method in dir(obj) if not method.startswith('_')]
                    }
                else:
                    attributes[name] = type(obj).__name__
        
        info.update({
            'attributes': attributes,
            'functions': functions,
            'classes': classes
        })
        
        return info
    
    @staticmethod
    def list_package_modules(package_name: str) -> List[str]:
        """List all modules in a package."""
        try:
            package = importlib.import_module(package_name)
            if hasattr(package, '__path__'):
                modules = []
                for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
                    modules.append(f"{package_name}.{modname}")
                return modules
            else:
                return [package_name]
        except ImportError:
            return []

# Custom import hook
class CustomImportHook:
    """Custom import hook for demonstration."""
    
    def __init__(self):
        self.modules = {}
    
    def add_virtual_module(self, name: str, content: Dict[str, Any]):
        """Add a virtual module that can be imported."""
        self.modules[name] = content
    
    def find_spec(self, name, path, target=None):
        """Find spec for virtual modules."""
        if name in self.modules:
            return importlib.machinery.ModuleSpec(name, self)
        return None
    
    def create_module(self, spec):
        """Create the virtual module."""
        return None  # Use default module creation
    
    def exec_module(self, module):
        """Execute the virtual module."""
        name = module.__name__
        if name in self.modules:
            for key, value in self.modules[name].items():
                setattr(module, key, value)

# Plugin system example
class PluginManager:
    """Simple plugin management system."""
    
    def __init__(self):
        self.plugins = {}
        self.plugin_hooks = {}
    
    def register_plugin(self, name: str, plugin_module):
        """Register a plugin."""
        self.plugins[name] = plugin_module
        
        # Look for hook functions
        for attr_name in dir(plugin_module):
            if attr_name.startswith('hook_'):
                hook_name = attr_name[5:]  # Remove 'hook_' prefix
                if hook_name not in self.plugin_hooks:
                    self.plugin_hooks[hook_name] = []
                
                hook_func = getattr(plugin_module, attr_name)
                self.plugin_hooks[hook_name].append((name, hook_func))
    
    def call_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Call all plugins that implement a specific hook."""
        results = []
        
        if hook_name in self.plugin_hooks:
            for plugin_name, hook_func in self.plugin_hooks[hook_name]:
                try:
                    result = hook_func(*args, **kwargs)
                    results.append((plugin_name, result))
                except Exception as e:
                    print(f"Error in plugin {plugin_name}: {e}")
        
        return results
    
    def list_plugins(self) -> List[str]:
        """List all registered plugins."""
        return list(self.plugins.keys())
    
    def get_plugin_info(self, name: str) -> Dict[str, Any]:
        """Get information about a specific plugin."""
        if name in self.plugins:
            plugin = self.plugins[name]
            return ModuleInspector.get_module_info(plugin)
        return {}

# Configuration module example
class ConfigManager:
    """Configuration management system."""
    
    def __init__(self):
        self.config = {}
        self.config_files = []
    
    def load_from_module(self, module_name: str):
        """Load configuration from a Python module."""
        try:
            config_module = importlib.import_module(module_name)
            
            for attr_name in dir(config_module):
                if not attr_name.startswith('_'):
                    self.config[attr_name] = getattr(config_module, attr_name)
            
            self.config_files.append(module_name)
            print(f"Loaded configuration from {module_name}")
            
        except ImportError as e:
            print(f"Failed to load configuration module {module_name}: {e}")
    
    def load_from_dict(self, config_dict: Dict[str, Any], source: str = "dict"):
        """Load configuration from a dictionary."""
        self.config.update(config_dict)
        self.config_files.append(source)
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self.config.copy()

# Lazy loading module
class LazyModule:
    """Lazy loading module wrapper."""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self._module = None
    
    def __getattr__(self, name: str):
        if self._module is None:
            print(f"Lazy loading module: {self.module_name}")
            self._module = importlib.import_module(self.module_name)
        
        return getattr(self._module, name)

# Module factory
class ModuleFactory:
    """Factory for creating modules with common patterns."""
    
    @staticmethod
    def create_singleton_module(name: str, class_def: type) -> ModuleType:
        """Create a module with a singleton instance."""
        code = f"""
class {class_def.__name__}:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Add methods from original class
"""
        
        # Add methods from the original class
        for method_name in dir(class_def):
            if not method_name.startswith('_'):
                method = getattr(class_def, method_name)
                if callable(method):
                    code += f"    {method_name} = {method}\n"
        
        code += f"\ninstance = {class_def.__name__}()\n"
        
        return create_dynamic_module(name, code)
    
    @staticmethod
    def create_namespace_module(name: str, namespace: Dict[str, Any]) -> ModuleType:
        """Create a module from a namespace dictionary."""
        module = ModuleType(name)
        module.__dict__.update(namespace)
        sys.modules[name] = module
        return module

# Example plugins for demonstration
def create_example_plugins():
    """Create example plugins for demonstration."""
    
    # Math plugin
    math_plugin_code = """
def hook_calculate(operation, a, b):
    if operation == 'add':
        return a + b
    elif operation == 'multiply':
        return a * b
    return None

def hook_validate(value):
    return isinstance(value, (int, float))

def get_plugin_info():
    return {
        'name': 'Math Plugin',
        'version': '1.0',
        'description': 'Basic math operations'
    }
"""
    
    # String plugin
    string_plugin_code = """
def hook_process_text(text):
    return text.upper()

def hook_validate(value):
    return isinstance(value, str)

def get_plugin_info():
    return {
        'name': 'String Plugin',
        'version': '1.0',
        'description': 'String processing operations'
    }
"""
    
    math_plugin = create_dynamic_module('math_plugin', math_plugin_code)
    string_plugin = create_dynamic_module('string_plugin', string_plugin_code)
    
    return math_plugin, string_plugin

def demonstrate_dynamic_modules():
    """Demonstrate dynamic module creation."""
    print("=== Dynamic Module Creation Demo ===")
    
    # Create a dynamic module
    module_code = """
def greet(name):
    return f"Hello, {name}!"

def calculate_area(radius):
    import math
    return math.pi * radius ** 2

VERSION = "1.0.0"
AUTHOR = "Dynamic Module Creator"
"""
    
    dynamic_module = create_dynamic_module('my_dynamic_module', module_code)
    
    print(f"Created module: {dynamic_module.__name__}")
    print(f"Greeting: {dynamic_module.greet('World')}")
    print(f"Circle area (r=5): {dynamic_module.calculate_area(5):.2f}")
    print(f"Module version: {dynamic_module.VERSION}")

def demonstrate_module_inspection():
    """Demonstrate module inspection capabilities."""
    print("\n=== Module Inspection Demo ===")
    
    # Inspect the math module
    import math
    math_info = ModuleInspector.get_module_info(math)
    
    print(f"Module: {math_info['name']}")
    print(f"File: {math_info['file']}")
    print(f"Functions: {list(math_info['functions'].keys())[:5]}...")  # Show first 5
    print(f"Total attributes: {len(math_info['attributes'])}")

def demonstrate_plugin_system():
    """Demonstrate the plugin system."""
    print("\n=== Plugin System Demo ===")
    
    # Create plugin manager
    plugin_manager = PluginManager()
    
    # Create and register plugins
    math_plugin, string_plugin = create_example_plugins()
    plugin_manager.register_plugin('math', math_plugin)
    plugin_manager.register_plugin('string', string_plugin)
    
    print(f"Registered plugins: {plugin_manager.list_plugins()}")
    
    # Call hooks
    print("\n--- Calling calculate hook ---")
    results = plugin_manager.call_hook('calculate', 'add', 5, 3)
    for plugin_name, result in results:
        print(f"{plugin_name}: {result}")
    
    print("\n--- Calling process_text hook ---")
    results = plugin_manager.call_hook('process_text', 'hello world')
    for plugin_name, result in results:
        print(f"{plugin_name}: {result}")
    
    print("\n--- Calling validate hook ---")
    results = plugin_manager.call_hook('validate', 42)
    for plugin_name, result in results:
        print(f"{plugin_name}: {result}")

def demonstrate_config_management():
    """Demonstrate configuration management."""
    print("\n=== Configuration Management Demo ===")
    
    # Create config manager
    config_manager = ConfigManager()
    
    # Load configuration from dictionary
    config_data = {
        'DATABASE_URL': 'sqlite:///app.db',
        'DEBUG': True,
        'SECRET_KEY': 'your-secret-key',
        'API_TIMEOUT': 30,
        'ALLOWED_HOSTS': ['localhost', '127.0.0.1']
    }
    
    config_manager.load_from_dict(config_data, 'app_config')
    
    print(f"Database URL: {config_manager.get('DATABASE_URL')}")
    print(f"Debug mode: {config_manager.get('DEBUG')}")
    print(f"API timeout: {config_manager.get('API_TIMEOUT')} seconds")
    print(f"Unknown setting: {config_manager.get('UNKNOWN_SETTING', 'default_value')}")

def demonstrate_lazy_loading():
    """Demonstrate lazy module loading."""
    print("\n=== Lazy Loading Demo ===")
    
    # Create lazy module (using json as example)
    lazy_json = LazyModule('json')
    
    print("Lazy module created (not loaded yet)")
    
    # First access will trigger loading
    data = {'name': 'John', 'age': 30}
    json_string = lazy_json.dumps(data)
    print(f"JSON string: {json_string}")
    
    # Subsequent access uses already loaded module
    parsed_data = lazy_json.loads(json_string)
    print(f"Parsed data: {parsed_data}")

def demonstrate_module_factory():
    """Demonstrate module factory patterns."""
    print("\n=== Module Factory Demo ===")
    
    # Create a simple class for singleton demonstration
    class Counter:
        def __init__(self):
            self.count = 0
        
        def increment(self):
            self.count += 1
            return self.count
        
        def get_count(self):
            return self.count
    
    # Create singleton module
    singleton_module = ModuleFactory.create_singleton_module('counter_singleton', Counter)
    
    print("Created singleton module")
    print(f"Count 1: {singleton_module.instance.increment()}")
    print(f"Count 2: {singleton_module.instance.increment()}")
    
    # Create namespace module
    namespace = {
        'PI': 3.14159,
        'E': 2.71828,
        'GOLDEN_RATIO': 1.618,
        'calculate_circle_area': lambda r: namespace['PI'] * r ** 2
    }
    
    constants_module = ModuleFactory.create_namespace_module('constants', namespace)
    print(f"Pi: {constants_module.PI}")
    print(f"Circle area (r=3): {constants_module.calculate_circle_area(3):.2f}")

def demonstrate_import_hooks():
    """Demonstrate custom import hooks."""
    print("\n=== Custom Import Hooks Demo ===")
    
    # Create custom import hook
    import_hook = CustomImportHook()
    
    # Add virtual module
    virtual_module_content = {
        'VERSION': '2.0.0',
        'greet': lambda name: f"Greetings, {name}!",
        'add': lambda a, b: a + b,
        'CONSTANTS': {'MAX_SIZE': 1000, 'DEFAULT_TIMEOUT': 30}
    }
    
    import_hook.add_virtual_module('virtual_module', virtual_module_content)
    
    # Install the hook
    sys.meta_path.insert(0, import_hook)
    
    try:
        # Import the virtual module
        import virtual_module
        
        print(f"Virtual module version: {virtual_module.VERSION}")
        print(f"Greeting: {virtual_module.greet('Python Developer')}")
        print(f"Addition: {virtual_module.add(10, 20)}")
        print(f"Constants: {virtual_module.CONSTANTS}")
        
    finally:
        # Remove the hook
        if import_hook in sys.meta_path:
            sys.meta_path.remove(import_hook)

def main():
    """Main demonstration function."""
    print("=== Python Advanced Modules and Packages Demo ===")
    
    demonstrate_dynamic_modules()
    demonstrate_module_inspection()
    demonstrate_plugin_system()
    demonstrate_config_management()
    demonstrate_lazy_loading()
    demonstrate_module_factory()
    demonstrate_import_hooks()
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()
