import math


class RangeSplitSort:
    """
    RangeSplitSort: A Hierarchical Segmented Sorting and Search Structure with Bitmap Optimization
    
    This class implements a hierarchical range-based sorting and search structure that efficiently organizes 
    numerical values into segments using a tree-like structure. It supports fast searching, insertion, and 
    neighbor-finding operations while leveraging bitwise operations for efficiency.
    
    Key Features:
    - **Hierarchical Segmentation:** Organizes values into multiple layers of segments for fast lookup.
    - **Bitmap Optimization:** Uses bitmasking (`bitmask`) to efficiently check for existing child nodes.
    - **Efficient Search (`search`):** Locates the node containing a value or its closest parent.
    - **Fast Next/Previous Lookup (`find_next`, `find_prev`):** Finds the next or previous available value.
    - **Traversal (`traverse_forward`, `traverse_backward`):** Allows sequential iteration over sorted values.
    - **Insertion (`insert`):** Supports dynamic insertion while maintaining structure.
    
    Usage Example:
    ```python
    values = [0.05, 0.2, 0.6, 1.5, 10, 50, 100, 500, 1000]
    num_segments = 64
    tree = RangeSplitSort(values, num_segments, use_bitwise=True)
    
    # Searching for a value
    found, parent, index = tree.search(50)
    
    # Finding next and previous values
    status, next_value, _ = tree.find_next(50)
    status, prev_value, _ = tree.find_prev(50)
    
    # Traversing forward and backward
    sorted_values = tree.traverse_forward()
    reverse_sorted_values = tree.traverse_backward()
"""


    def __init__(self, values=None, num_segments=64, parent=None, layer=0, use_bitwise=False):
        if use_bitwise and num_segments not in [16, 32, 64, 128, 256]:
            raise ValueError("Bitwise mode only supports num_segments = 32 or 64")
        
        self.values = values  # List of values in this node
        self.child = [None] * num_segments  # Array of child nodes
        self.parent = parent  # Pointer to the parent node
        self.num_segments = num_segments  # Number of segments
        self.parent_index = 0
        self.layer = layer
        self.org_values = []
        self.use_bitwise = use_bitwise
        self.bitmask = 0  # Bitmask for tracking child nodes

        children = []
        if parent is None:
            self.max_layer = 0
            self.base = self
            
            self.org_values = None
            max_value = max(values)
            self.divider = num_segments ** int(math.log(max_value, num_segments) + 1)
            # print(f"Root node created with divider: {self.divider}")
            for value in self.values:
                index = int(value / self.divider)
                
                if self.child[index] is None:
                    self.child[index] = RangeSplitSort([], self.num_segments, self, self.layer + 1, use_bitwise=self.use_bitwise)
                    # children.append(self.child[index])
                    children.append(index)
                    self.bitmask |= (1 << index)  # Set bit in bitmask
                    # print(f"Created child node at index {index} for value {value}")

                self.child[index].values.append(value - int(value / self.divider) * self.divider)
                self.child[index].org_values.append(value)
                self.child[index].parent_index = index

            # for c in children:
            #     c.populate()
            for i in children:
                self.child[i].populate()
            del self.values
            self.values = None
        else:
            self.base=self.parent.base
            if self.layer > self.base.max_layer:
                self.base.max_layer = self.layer
            self.divider = parent.divider / num_segments
            # print(f"Child node created with divider: {self.divider} and Layer: {self.layer}")
    
    def populate(self):
        if len(set(self.values)) <= 1:
            # print(f"Stopping recursion at node with values: {self.values}")
            return
            
        # print(f"Distributing values in node with divider {self.divider}: {self.values}")
        children = []
        for value in self.values:
            index = int(value / self.divider)
            
            if self.child[index] is None:
                self.child[index] = RangeSplitSort([], self.num_segments, self, self.layer + 1, use_bitwise=self.use_bitwise)
                self.bitmask |= (1 << index)  # Set bit in bitmask
                # print(f"Created child node at index {index} for value {value}")
                children.append(index)
                
            self.child[index].values.append(value - int(value / self.divider) * self.divider)
            self.child[index].org_values.append(value)
            self.child[index].parent_index = index
            # print(f"Added value {value} to child at index {index}")
        
        # for i, child in enumerate(self.child):
        for i in children:
            child = self.child[i]
            child.populate()
            # del self.org_values
            self.org_values = None
            # del self.values
            self.values = None
            
    def search(self, value):
        target = self
        
        vv = value
        while True:
        # for i in range(1000):
            index = int(vv / target.divider)
            if target.org_values:
                if target.org_values[0] == value:
                    return True, target.parent, target.parent_index
                else:
                    return False, target.parent, target.parent_index
        
            if not target.child[index]:
                return False, target, index
                
            vv = vv - int(vv / target.divider) * target.divider
            target = target.child[index]
        return False, None
    
    def insert(self, value):
        target = self
        vv = value
        target_index = 0
        parent_index = 0
        previous_target = None
        
        # for i in range(1000):
        while True:
            index = int(vv / target.divider)
            if target.org_values:
                if target.org_values[0] == value:
                    target = previous_target
                    target_index = parent_index
                    return False
                break
            
            if not target.child[index]:
                break
            
            previous_target = target
            parent_index = index
            
            vv = vv - int(vv / target.divider) * target.divider

            target = target.child[index]
            
        target.child[index] = RangeSplitSort([], target.num_segments, target, target.layer + 1, use_bitwise=self.use_bitwise)
        target.bitmask |= (1 << index)  # Set bit in bitmask
        target.child[index].values.append(value - int(value / target.divider) * target.divider)
        target.child[index].org_values.append(value)
        target.child[index].parent_index = index
        target = target.child[index]
        target.populate()    
        return True
        
    def _find_next_bm(b, pos):
        # Step 1: Offset input by -1 for 0-based indexing
        pos -= 1
        
        # Step 2: Mask out bits up to and including `pos`
        masked_b = b & ~((1 << (pos + 1)) - 1)
        
        # Step 3: If no bits are set after `pos`, return -1
        if masked_b == 0:
            return -1
            # return 0
        
        # Step 4: Find the least significant set bit (LSB)
        lsb = masked_b & -masked_b
        
        # Step 5: Convert the LSB to the correct bit position (1-based indexing)
        return lsb.bit_length()
    
    def _find_previous_bm(b, pos):
        # Step 1: Mask out bits strictly above `pos`
        x1 = b & ((1 << pos) - 1)
        
        # Step 2: If no bits are set, return -1
        if x1 == 0:
            # return 0
            return -1
        
        # Step 3: Isolate the most significant set bit using bitwise operations
        return x1.bit_length()

    def find_next_side(self, start_point, node):
        if self.use_bitwise:
            ret = RangeSplitSort._find_next_bm(node.bitmask, start_point+1)
            if ret == -1:
                return -1
            else: 
                return ret-1
        else:
            index = start_point
            found = False
            for i in range(index, node.num_segments):
                if node.child[i] is not None:
                    # return True, i
                    return i
            # return False, 0
        return -1

    def find_next(self, current_value):
        # Returns status, next value and pointer if there is
        # REturns status, 0, pointer of the parent otherwise
        
        found, node, index = self.search(current_value)
        target = node
        
        index = index + 1
        found = False
        # for iii in range(1000000):
        while True:
            ind2 = index
            # for ii in range(10000):
            while True:

                ind2 = self.find_next_side(ind2, target)
                if ind2 == -1:
                    break
                if target.child[ind2].org_values:
                    return True, target.child[ind2].org_values[0], target
                target = target.child[ind2]
                ind2 = 0
            index = target.parent_index+1
            target = target.parent
            if not target:
                return False, 0, None
            

    def find_prev_side(self, start_point, node):
        
        if self.use_bitwise:
            ret = RangeSplitSort._find_previous_bm(node.bitmask, start_point+1)
            if ret == -1:
                return -1
            else: 
                return ret-1
        else:
            index = start_point
            found = False
            for i in range(index, -1, -1):
                if node.child[i] is not None:
                    # return True, i
                    return i
            # return False, 0
            return -1

    def find_prev(self, current_value):
        # Returns status, next value and pointer if there is
        # REturns status, 0, pointer of the parent otherwise
        
        found, node, index = self.search(current_value)

        target = node
        
        index = index - 1
        found = False
        # for iii in range(400):
        while True:
            ind2 = index
            # for ii in range(400):
            while True:

                # found, ind2 = self.find_prev_side(ind2, target)
                ind2 = self.find_prev_side(ind2, target)
                if ind2 == -1:
                    break
                if target.child[ind2].org_values:
                    return True, target.child[ind2].org_values[0], target
                target = target.child[ind2]
                ind2 = self.num_segments-1

            index = target.parent_index-1
            target = target.parent
            if not target:
                return False, 0, None


    def traverse_forward(self, current_value=0):
        outv = []
        next_node = self
        next_val = current_value

        found, node, index = self.search(current_value)
        if found:
            outv.append(current_value)
        
        while True:
            status, next_val, next_node = next_node.find_next(next_val)
            if not status:
                break
            outv.append(next_val)
            # print("outv", outv)
            next_node = next_node
        return outv

    def traverse_backward(self, current_value=0):
        outv = []
        next_node = self
        if current_value == 0:
            next_val = self.divider
        else:
            next_val = current_value
            
        found, node, index = self.search(current_value)
        if found:
            outv.append(current_value)
        
        # for i in range(10000):
        while True:
            status, next_val, next_node = next_node.find_prev(next_val)
            if not status:
                break
            outv.append(next_val)
            next_node = next_node
        return outv

    def sort(numbers, reverse=False, num_segments=64, use_bitwise=True):

        # Split numbers into positive and negative lists
        positive_list = [num for num in numbers if num >= 0]
        negative_list = [-num for num in numbers if num < 0]
        pos_max = max(positive_list)
        neg_max = max(negative_list)
        if positive_list:
            pos_sort = RangeSplitSort(positive_list, num_segments=num_segments, use_bitwise=use_bitwise)
        if negative_list:
            neg_sort = RangeSplitSort(negative_list, num_segments=num_segments, use_bitwise=use_bitwise)
        
        if reverse:
            if positive_list:
                positive_list = pos_sort.traverse_backward(pos_max)
            if negative_list:
                negative_list = neg_sort.traverse_forward()
                negative_list = [-num for num in negative_list if num < 0]
            return positive_list + negative_list
        else:
            if positive_list:
                positive_list = pos_sort.traverse_forward()
            if negative_list:
                negative_list = neg_sort.traverse_backward(neg_max)
                negative_list = [-num for num in negative_list if num < 0]
            return negative_list + positive_list
# Test script
def print_tree(node, level=0, index=1):
    print("  " * level + f"Node (Layer: {node.layer}, index: {index}, parent_index: {node.parent_index}, Values: {node.values}, Org Values: {node.org_values}, Bitmask: {bin(node.bitmask)})")
    # print("  " * level + f"Node (Layer: {node.layer}, index: {index}, Values: {node.values}, Org Values: {node.org_values}")
    for i, child in enumerate(node.child):
        if child is not None:
            print_tree(child, level + 1, i)

test_values = [0.05, 0.2, 0.6, 1.5, 10, 50, 100, 500, 1000]
num_segments = 64
# num_segments = 100
# num_segments = 10
root = RangeSplitSort(test_values, num_segments, use_bitwise=True)

# print("\nTree Structure:")
# print_tree(root)

# Test search
print("\nTesting search...")
new_values = [10, 2003]
for val in new_values:
    res = root.search(val)
    print("search", val, res)
    
# Test insertion
print("\nTesting insertion...")
new_values = [2003, 0.23, 71]
for val in new_values:
    root.insert(val)

print("\nTree Structure:")
print_tree(root)

# Test find_next
print("\nTesting find_next...")
test_values = [0.01, 3, 10, 50, 72, 100, 500, 1000]
# test_values = [0.01]
# test_values = [0.01]
for val in test_values:
    # next_val, next_node = root.find_next(val)
    # print(f"Next value after {val}: {next_val}")
    # status, next_val, next_node = root.find_next_side(val)
    status, next_val, next_node = root.find_next(val)
    print(f"Next value after  {status}: {val}: {next_val}")

# root.find_next(0)

print(root.search(71))

print(root.traverse_forward())
# print(root.traverse_forward(72))
# print(root.traverse_forward(0))
print(root.traverse_forward(71))
# print(root.traverse_forward(70))
# print(root.find_next(70)[1])
print(root.find_prev(72))
# print(root.find_next(70)[1])
print(root.traverse_backward(70))
print(root.traverse_backward(1000))
print(root.find_prev(100))




import time
import pandas as pd
import random

# Example usage and testing
def run_test(size, element_step, use_integer=True):

    # Insert test
    if use_integer:
        vals = [i for i in range(0, size, element_step)]
    else:
        vals = [random.uniform(0, int(size/10)) for _ in range(size)]

    start_time = time.time()
    # test_bitmap.insert(vals)
    # test_bitmap = RangeSplitSort(vals, 100, use_bitwise=False)
    test_bitmap = RangeSplitSort(vals, 64, use_bitwise=True)
    insertion_time = time.time() - start_time

    # Contains test (middle element)
    contains_value = (size // 2) - ((size // 2) % element_step)  # Closest inserted value to the middle
    start_time = time.time()
    # contains_result = test_bitmap.get(contains_value)
    contains_result = test_bitmap.search(contains_value)
    contains_time = time.time() - start_time

    # Set test (middle element)
    start_time = time.time()
    # contains_result = test_bitmap.set(contains_value)
    contains_result = test_bitmap.insert(contains_value)
    set_time = time.time() - start_time
    
    # Find next test
    start_time = time.time()
    next_result = test_bitmap.find_next(contains_value)
    next_time = time.time() - start_time

    # Find previous test
    start_time = time.time()
    # next_result = test_bitmap.find_previous(contains_value)
    next_result = test_bitmap.find_prev(contains_value)
    previous_time = time.time() - start_time

    # Traverse sorted test
    start_time = time.time()
    sorted_traversal = test_bitmap.traverse_forward()
    traversal_time = time.time() - start_time

    num_layers = test_bitmap.max_layer+1
    
    # Return the results for this test
    return {
        "Test Data Size": int(len(vals)),
        "Insertion Time (s)": set_time,
        "Contains Time": contains_time,
        "Bulk Insert Time (sorting)": insertion_time,
        "Next Time (s)": next_time,
        "Previous Time (s)": previous_time,
        "Traversal Time": traversal_time,
        "Number of Layers": num_layers
    }


# Test configurations
sizes = [100, 10000, 1_000_000]  # Corrected sizes to satisfy multi-layer design requirements
# sizes = [100, 10000, 100000]  # Corrected sizes to satisfy multi-layer design requirements
element_step = 10  # Sparse test with step size

# Run and collect results
use_integer = False
results = [run_test(size, element_step, use_integer=use_integer) for size in sizes]

# Create DataFrame and reorganize metrics as rows and sizes as columns
df = pd.DataFrame(results).T
df.columns = [f"Size {sizes[i]}" for i in range(len(results))]

# Calculate change rates manually
if len(df.columns) > 1:
    change_rate_1_2 = ((df.iloc[:, 1] - df.iloc[:, 0]) / df.iloc[:, 0]).replace([float('inf'), -float('inf')], 0).fillna(0)
    df["Change Rate 1-2"] = change_rate_1_2.values

if len(df.columns) > 2:
    change_rate_2_3 = ((df.iloc[:, 2] - df.iloc[:, 1]) / df.iloc[:, 1]).replace([float('inf'), -float('inf')], 0).fillna(0)
    df["Change Rate 2-3"] = change_rate_2_3.values

print(df)