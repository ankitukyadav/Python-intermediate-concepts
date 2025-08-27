"""
Python Data Structures and Algorithms Demo
This script demonstrates custom data structures and common algorithms.
"""

from typing import Any, List, Optional, Tuple, Iterator
from collections import deque
import heapq
import bisect

# Custom Data Structures
class Node:
    """Node class for linked data structures."""
    
    def __init__(self, data: Any):
        self.data = data
        self.next: Optional['Node'] = None
    
    def __repr__(self):
        return f"Node({self.data})"

class LinkedList:
    """Implementation of a singly linked list."""
    
    def __init__(self):
        self.head: Optional[Node] = None
        self.size = 0
    
    def append(self, data: Any) -> None:
        """Add element to the end of the list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self.size += 1
    
    def prepend(self, data: Any) -> None:
        """Add element to the beginning of the list."""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
        self.size += 1
    
    def delete(self, data: Any) -> bool:
        """Delete first occurrence of data."""
        if not self.head:
            return False
        
        if self.head.data == data:
            self.head = self.head.next
            self.size -= 1
            return True
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                self.size -= 1
                return True
            current = current.next
        
        return False
    
    def find(self, data: Any) -> Optional[Node]:
        """Find node with given data."""
        current = self.head
        while current:
            if current.data == data:
                return current
            current = current.next
        return None
    
    def __len__(self) -> int:
        return self.size
    
    def __iter__(self) -> Iterator[Any]:
        current = self.head
        while current:
            yield current.data
            current = current.next
    
    def __repr__(self) -> str:
        return f"LinkedList({list(self)})"

class Stack:
    """Implementation of a stack using list."""
    
    def __init__(self):
        self._items: List[Any] = []
    
    def push(self, item: Any) -> None:
        """Push item onto stack."""
        self._items.append(item)
    
    def pop(self) -> Any:
        """Pop item from stack."""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()
    
    def peek(self) -> Any:
        """Peek at top item without removing."""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._items[-1]
    
    def is_empty(self) -> bool:
        """Check if stack is empty."""
        return len(self._items) == 0
    
    def size(self) -> int:
        """Get stack size."""
        return len(self._items)
    
    def __repr__(self) -> str:
        return f"Stack({self._items})"

class Queue:
    """Implementation of a queue using deque."""
    
    def __init__(self):
        self._items = deque()
    
    def enqueue(self, item: Any) -> None:
        """Add item to rear of queue."""
        self._items.append(item)
    
    def dequeue(self) -> Any:
        """Remove item from front of queue."""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._items.popleft()
    
    def front(self) -> Any:
        """Peek at front item."""
        if self.is_empty():
            raise IndexError("front from empty queue")
        return self._items[0]
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._items) == 0
    
    def size(self) -> int:
        """Get queue size."""
        return len(self._items)
    
    def __repr__(self) -> str:
        return f"Queue({list(self._items)})"

class TreeNode:
    """Node for binary tree."""
    
    def __init__(self, data: Any):
        self.data = data
        self.left: Optional['TreeNode'] = None
        self.right: Optional['TreeNode'] = None
    
    def __repr__(self):
        return f"TreeNode({self.data})"

class BinarySearchTree:
    """Implementation of a binary search tree."""
    
    def __init__(self):
        self.root: Optional[TreeNode] = None
    
    def insert(self, data: Any) -> None:
        """Insert data into BST."""
        if not self.root:
            self.root = TreeNode(data)
        else:
            self._insert_recursive(self.root, data)
    
    def _insert_recursive(self, node: TreeNode, data: Any) -> None:
        """Recursive helper for insert."""
        if data < node.data:
            if node.left is None:
                node.left = TreeNode(data)
            else:
                self._insert_recursive(node.left, data)
        elif data > node.data:
            if node.right is None:
                node.right = TreeNode(data)
            else:
                self._insert_recursive(node.right, data)
    
    def search(self, data: Any) -> bool:
        """Search for data in BST."""
        return self._search_recursive(self.root, data)
    
    def _search_recursive(self, node: Optional[TreeNode], data: Any) -> bool:
        """Recursive helper for search."""
        if node is None:
            return False
        if data == node.data:
            return True
        elif data < node.data:
            return self._search_recursive(node.left, data)
        else:
            return self._search_recursive(node.right, data)
    
    def inorder_traversal(self) -> List[Any]:
        """Inorder traversal of BST."""
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node: Optional[TreeNode], result: List[Any]) -> None:
        """Recursive helper for inorder traversal."""
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.data)
            self._inorder_recursive(node.right, result)

# Graph implementation
class Graph:
    """Implementation of an undirected graph using adjacency list."""
    
    def __init__(self):
        self.vertices = {}
    
    def add_vertex(self, vertex: Any) -> None:
        """Add a vertex to the graph."""
        if vertex not in self.vertices:
            self.vertices[vertex] = []
    
    def add_edge(self, vertex1: Any, vertex2: Any) -> None:
        """Add an edge between two vertices."""
        self.add_vertex(vertex1)
        self.add_vertex(vertex2)
        
        self.vertices[vertex1].append(vertex2)
        self.vertices[vertex2].append(vertex1)
    
    def get_neighbors(self, vertex: Any) -> List[Any]:
        """Get neighbors of a vertex."""
        return self.vertices.get(vertex, [])
    
    def bfs(self, start_vertex: Any) -> List[Any]:
        """Breadth-first search traversal."""
        if start_vertex not in self.vertices:
            return []
        
        visited = set()
        queue = deque([start_vertex])
        result = []
        
        while queue:
            vertex = queue.popleft()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                
                for neighbor in self.vertices[vertex]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        
        return result
    
    def dfs(self, start_vertex: Any) -> List[Any]:
        """Depth-first search traversal."""
        if start_vertex not in self.vertices:
            return []
        
        visited = set()
        result = []
        
        def dfs_recursive(vertex):
            visited.add(vertex)
            result.append(vertex)
            
            for neighbor in self.vertices[vertex]:
                if neighbor not in visited:
                    dfs_recursive(neighbor)
        
        dfs_recursive(start_vertex)
        return result

# Sorting Algorithms
class SortingAlgorithms:
    """Collection of sorting algorithms."""
    
    @staticmethod
    def bubble_sort(arr: List[Any]) -> List[Any]:
        """Bubble sort implementation."""
        arr = arr.copy()
        n = len(arr)
        
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True
            
            if not swapped:
                break
        
        return arr
    
    @staticmethod
    def quick_sort(arr: List[Any]) -> List[Any]:
        """Quick sort implementation."""
        if len(arr) <= 1:
            return arr
        
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        
        return (SortingAlgorithms.quick_sort(left) + 
                middle + 
                SortingAlgorithms.quick_sort(right))
    
    @staticmethod
    def merge_sort(arr: List[Any]) -> List[Any]:
        """Merge sort implementation."""
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = SortingAlgorithms.merge_sort(arr[:mid])
        right = SortingAlgorithms.merge_sort(arr[mid:])
        
        return SortingAlgorithms._merge(left, right)
    
    @staticmethod
    def _merge(left: List[Any], right: List[Any]) -> List[Any]:
        """Helper function for merge sort."""
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result

# Search Algorithms
class SearchAlgorithms:
    """Collection of search algorithms."""
    
    @staticmethod
    def linear_search(arr: List[Any], target: Any) -> int:
        """Linear search implementation."""
        for i, element in enumerate(arr):
            if element == target:
                return i
        return -1
    
    @staticmethod
    def binary_search(arr: List[Any], target: Any) -> int:
        """Binary search implementation (requires sorted array)."""
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1

# Dynamic Programming Examples
class DynamicProgramming:
    """Collection of dynamic programming solutions."""
    
    @staticmethod
    def fibonacci_dp(n: int) -> int:
        """Fibonacci using dynamic programming."""
        if n <= 1:
            return n
        
        dp = [0] * (n + 1)
        dp[1] = 1
        
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        
        return dp[n]
    
    @staticmethod
    def longest_common_subsequence(str1: str, str2: str) -> int:
        """Find length of longest common subsequence."""
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    @staticmethod
    def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
        """0/1 Knapsack problem using dynamic programming."""
        n = len(weights)
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(
                        values[i - 1] + dp[i - 1][w - weights[i - 1]],
                        dp[i - 1][w]
                    )
                else:
                    dp[i][w] = dp[i - 1][w]
        
        return dp[n][capacity]

def demonstrate_data_structures():
    """Demonstrate custom data structures."""
    print("=== Data Structures Demo ===")
    
    # Linked List
    print("\n--- Linked List ---")
    ll = LinkedList()
    for item in [1, 2, 3, 4, 5]:
        ll.append(item)
    
    print(f"Linked List: {ll}")
    print(f"Length: {len(ll)}")
    print(f"Find 3: {ll.find(3)}")
    ll.delete(3)
    print(f"After deleting 3: {ll}")
    
    # Stack
    print("\n--- Stack ---")
    stack = Stack()
    for item in [1, 2, 3, 4, 5]:
        stack.push(item)
    
    print(f"Stack: {stack}")
    print(f"Pop: {stack.pop()}")
    print(f"Peek: {stack.peek()}")
    print(f"Stack after pop: {stack}")
    
    # Queue
    print("\n--- Queue ---")
    queue = Queue()
    for item in [1, 2, 3, 4, 5]:
        queue.enqueue(item)
    
    print(f"Queue: {queue}")
    print(f"Dequeue: {queue.dequeue()}")
    print(f"Front: {queue.front()}")
    print(f"Queue after dequeue: {queue}")
    
    # Binary Search Tree
    print("\n--- Binary Search Tree ---")
    bst = BinarySearchTree()
    for item in [5, 3, 7, 2, 4, 6, 8]:
        bst.insert(item)
    
    print(f"Inorder traversal: {bst.inorder_traversal()}")
    print(f"Search 4: {bst.search(4)}")
    print(f"Search 9: {bst.search(9)}")
    
    # Graph
    print("\n--- Graph ---")
    graph = Graph()
    edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D'), ('D', 'E')]
    for v1, v2 in edges:
        graph.add_edge(v1, v2)
    
    print(f"BFS from A: {graph.bfs('A')}")
    print(f"DFS from A: {graph.dfs('A')}")

def demonstrate_algorithms():
    """Demonstrate various algorithms."""
    print("\n=== Algorithms Demo ===")
    
    # Sorting
    print("\n--- Sorting Algorithms ---")
    unsorted_data = [64, 34, 25, 12, 22, 11, 90]
    
    print(f"Original: {unsorted_data}")
    print(f"Bubble Sort: {SortingAlgorithms.bubble_sort(unsorted_data)}")
    print(f"Quick Sort: {SortingAlgorithms.quick_sort(unsorted_data)}")
    print(f"Merge Sort: {SortingAlgorithms.merge_sort(unsorted_data)}")
    
    # Searching
    print("\n--- Search Algorithms ---")
    sorted_data = [11, 12, 22, 25, 34, 64, 90]
    target = 25
    
    print(f"Array: {sorted_data}")
    print(f"Linear search for {target}: {SearchAlgorithms.linear_search(sorted_data, target)}")
    print(f"Binary search for {target}: {SearchAlgorithms.binary_search(sorted_data, target)}")
    
    # Dynamic Programming
    print("\n--- Dynamic Programming ---")
    print(f"Fibonacci(10): {DynamicProgramming.fibonacci_dp(10)}")
    
    str1, str2 = "ABCDGH", "AEDFHR"
    lcs_length = DynamicProgramming.longest_common_subsequence(str1, str2)
    print(f"LCS of '{str1}' and '{str2}': {lcs_length}")
    
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    max_value = DynamicProgramming.knapsack_01(weights, values, capacity)
    print(f"0/1 Knapsack (capacity {capacity}): {max_value}")

def main():
    """Main demonstration function."""
    print("=== Python Data Structures and Algorithms Demo ===")
    
    demonstrate_data_structures()
    demonstrate_algorithms()

if __name__ == "__main__":
    main()
