# Refactored Version

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

    def __init__(self, values=None, num_segments=64, is_base=True, layer=0, use_bitwise=True, use_dictionary=False, use_child_generator=False):
        # use_bitwise - use the use of bitmap or array, but it is usually faster when False
        # use_dictionary - use dictionary or array but it is usually faster when False
        # use_child_generator - created to test bulk creation but it is faster when False
        
        self.values = values  # List of values in this node
        self.parent = None  # Pointer to the parent node
        self.parent_index = 0
        self.layer = layer
        self.org_values = []
        self.bitmask = 0  # Bitmask for tracking child nodes
        self.child = None

        children = []
        if is_base:
            if use_bitwise and num_segments not in [32, 64]:
                raise ValueError("Bitwise mode only supports num_segments = 32 or 64")
            self.use_bitwise = use_bitwise
            self.use_dictionary = use_dictionary
            self.child = {} if use_dictionary else [None] * num_segments  # Array of child nodes
            
            self.num_segments = num_segments  # Number of segments
            self.max_layer = 0
            self.base = self
            
            self.org_values = None
            max_value = max(values)
            self.divider = num_segments ** int(math.log(max_value, num_segments) + 1)

            if use_child_generator:
                self.child_repository = [RangeSplitSort([], is_base=False) for _ in range(len(values)*5)]
            else:
                self.child_repository = []
            self.child_rep_inc = 0
            
            
            # print(f"Root node created with divider: {self.divider}")
            for value in self.values:
                index = int(value / self.divider)
                
                # if self.child[index] is None:
                if (use_dictionary and index not in self.child) or (not use_dictionary and self.child[index] is None):
                    # self.child[index] = RangeSplitSort([], layer=self.layer + 1, is_base=False)
                    self.child[index] = self.get_child(layer=self.layer + 1)
                    children.append(index)
                    self.bitmask |= (1 << index)  # Set bit in bitmask
                    # print(f"Created child node at index {index} for value {value}")

                self.child[index].values.append(value - int(value / self.divider) * self.divider)
                self.child[index].org_values.append(value)
                self.child[index].parent_index = index

            for i in children:
                self.child[i].populate(self)
            del self.values
            self.values = None

    
    def get_child(self, layer):
        if self.base.child_rep_inc < len(self.base.child_repository):
            self.base.child_repository[self.base.child_rep_inc].layer = layer
            ret =  self.base.child_repository[self.base.child_rep_inc]
            self.base.child_rep_inc += 1
            return ret
        else:
            return RangeSplitSort([], layer=self.layer, is_base=False)
    
    def populate(self, parent):

            
        self.parent = parent
        self.base=self.parent.base
        if self.layer > self.base.max_layer:
            self.base.max_layer = self.layer
        self.divider = parent.divider / self.base.num_segments
        
        if len(set(self.values)) <= 1:
            # print(f"Stopping recursion at node with values: {self.values}")
            return
            
        self.child = {} if self.base.use_dictionary else [None] * num_segments  # Array of child nodes
            
        # print(f"Distributing values in node with divider {self.divider}: {self.values}")
        children = []
        # for value in self.values:
        for i, value in enumerate(self.values):
            index = int(value / self.divider)
            
            if (self.base.use_dictionary and index not in self.child) or (not self.base.use_dictionary and self.child[index] is None):

                # self.child[index] = RangeSplitSort([], layer=self.layer + 1, is_base=False)
                self.child[index] = self.get_child(layer=self.layer + 1)
                self.bitmask |= (1 << index)  # Set bit in bitmask
                # print(f"Created child node at index {index} for value {value}")
                children.append(index)
                
            self.child[index].values.append(value - int(value / self.divider) * self.divider)
            self.child[index].org_values.append(self.org_values[i])
            self.child[index].parent_index = index
            # print(f"Added value {value} to child at index {index}")
        
        for i in children:
            child = self.child[i]
            child.populate(self)
            del self.org_values
            self.org_values = None
            del self.values
            self.values = None
            
    def search(self, value):
        target = self
        if value > self.divider:
            return False, self, -1

        vvv = []
        vv = value
        for i in range(1000):
            if i == 999:
                print("INFINITE LOOP")
            vvv.append(vv)
            index = int(vv / target.divider)
            if target.org_values:
                if target.org_values[0] == value:
                    return True, target, target.parent_index
                else:
                    return False, target, target.parent_index
            try:
                if (self.base.use_dictionary and index not in target.child) or (not self.base.use_dictionary and target.child[index] is None):

                    return False, target, index
            except Exception as e:
                print(f"Caught an error: {e}")
                raise  # Re-throws the same exception

            prev_vv = vv
            divided = int(vv / target.divider) * target.divider
            vv = vv - int(vv / target.divider) * target.divider
                
            target = target.child[index]
            
            next_index = int(vv / target.divider)
        return False, None
    
    def insert(self, value):
        if value > self.divider:
            print("Too big")
            return True
        target = self
        vv = value
        target_index = 0
        parent_index = 0
        previous_target = None
        
        # while True:
        for i in range(1000):
            if i == 999:
                print("INFINITE LOOP")
            if target.org_values:
                if target.org_values[0] == value:
                    target = previous_target
                    target_index = parent_index
                    return False
                break
            
            index = int(vv / target.divider)
            if (self.base.use_dictionary and index not in target.child) or (not self.base.use_dictionary and target.child[index] is None):
                break
            
            previous_target = target
            parent_index = index
            
            vv = vv - int(vv / target.divider) * target.divider

            target = target.child[index]
            
        # target.child[index] = RangeSplitSort([], layer=target.layer + 1, is_base=False)

        if not target.child:
            target.child = {} if self.base.use_dictionary else [None] * num_segments  # Array of child nodes
        target.child[index] = self.get_child(layer=self.layer + 1)
        
        target.bitmask |= (1 << index)  # Set bit in bitmask
        divider = target.parent.divider / self.base.num_segments
        target.child[index].values.append(value - int(value / divider) * divider)
        target.child[index].org_values.append(value)
        target.child[index].parent_index = index
        org_target = target
        target = target.child[index]
        # target.populate(self)    
        target.populate(org_target)    
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
        # x1 = b & ((1 << pos) - 1)
        x1 = b & ((1 << pos+1) - 1)
        
        # Step 2: If no bits are set, return -1
        if x1 == 0:
            # return 0
            return -1
        
        # Step 3: Isolate the most significant set bit using bitwise operations
        return x1.bit_length()

        
    def find_next_prev_side(self, start_point, node, forward=True):

        if start_point < 0:
            print("find_next_prev_side got -1")
        # if self.use_bitwise:
        if self.base.use_bitwise:
            if forward:
                ret = RangeSplitSort._find_next_bm(node.bitmask, start_point)
            else:
                ret = RangeSplitSort._find_previous_bm(node.bitmask, start_point)
            if ret == -1:
                return -1
            else: 
                return ret-1
        else:
            index = start_point
            found = False
            if forward:
                # for i in range(index, node.num_segments):
                for i in range(index, node.base.num_segments):
                    # if node.child[i] is not None:
                    if (self.base.use_dictionary and i in node.child) or \
                        (not self.base.use_dictionary and node.child[i] is not None):
                        # return True, i
                        return i
            else:
                for i in range(index, -1, -1):
                    # if node.child[i] is not None:
                    if (self.base.use_dictionary and i in node.child) or \
                        (not self.base.use_dictionary and node.child[i] is not None):
                        return i
        return -1

    def find_next_prev(self, current_value, index=-1, forward=True):
 
        if index == -1:
            found, node, index = self.search(current_value)


            if node.org_values:
                if found:
                    node = node.parent
                    pass
                else:
                    if (forward and node.org_values[0] > current_value) or (not forward and node.org_values[0] < current_value):
                        return True, node.org_values[0], node.parent, index
                    # else:
                        index = node.parent_index 
                        node = node.parent
            else:
                if not found and index == -1:
                    index = 0
                    if node == node.base:
                        return False, 0, None, -1
                    node = node.parent
                
            target = node
        else:
            target = self

        # out of bounds
        # if (forward and index >= self.num_segments) or (not forward and index <= 0):
        if (forward and index >= self.base.num_segments) or (not forward and index <= 0):
            return False, 0, None, -1
            
        # index = index + 1
        index = index + 1 if forward else index - 1
            
        found = False
        # while True:
        for iii in range(1000):
            if iii == 999:
                print("INFINITE LOOP 1")
                return 
            ind2 = index
            # while True:
            for ii in range(1000):
                if ii == 999:
                    print("INFINITE LOOP 2")
                    return 
                ind2 = self.find_next_prev_side(ind2, target, forward=forward)
                if ind2 == -1:
                    break
                if target.child[ind2].org_values:
                    return True, target.child[ind2].org_values[0], target, ind2
                target = target.child[ind2]
                # ind2 = 0 if forward else self.num_segments - 1
                ind2 = 0 if forward else self.base.num_segments - 1
            
            # index = target.parent_index+1
            # print("target.parent_index", target.parent_index)
            for k in range(100):
                index = target.parent_index+1 if forward else target.parent_index-1
                target = target.parent
                if not target:
                    # print("NOT FOUND 2!!!")
                    return False, 0, None, -1
                # if -1 < index and index < self.num_segments:
                if -1 < index and index < self.base.num_segments:
                    break
                    
        print("END LOOP !!!")

    
    def find_next(self, current_value, index=-1):
        return self.find_next_prev(current_value=current_value, index=index, forward=True)
        
    def find_prev(self, current_value, index=-1):
        return self.find_next_prev(current_value=current_value, index=index, forward=False)

    def traverse(self, current_value=0, forward=True):
        if not forward:
            current_value = self.divider - 1
            
        outv = []
        next_node = self
        next_val = current_value

        found, node, index = self.search(current_value)
        
        if found:
            outv.append(current_value)

        index = -1
        while True:
            if forward:
                status, next_val, next_node, index = next_node.find_next(next_val, index)
            else:
                status, next_val, next_node, index = next_node.find_prev(next_val, index)
            if not status:
                break
            outv.append(next_val)
        return outv
        
    def traverse_forward(self, current_value=0):
        return self.traverse(current_value=current_value, forward=True)
        
    def traverse_backward(self, current_value=0):
        return self.traverse(current_value=current_value, forward=False)


    def sort(numbers, reverse=False, num_segments=64, use_bitwise=True):
        positive_list = [num for num in numbers if num >= 0]
        negative_list = [-num for num in numbers if num < 0]

        pos_max = max(positive_list) if positive_list else 0
        neg_max = max(negative_list) if negative_list else 0
        if positive_list:
            pos_sort = RangeSplitSort(positive_list, num_segments=num_segments, use_bitwise=use_bitwise)
        if negative_list:
            neg_sort = RangeSplitSort(negative_list, num_segments=num_segments, use_bitwise=use_bitwise)
        
        if reverse:
            if positive_list:
                positive_list = pos_sort.traverse_backward(pos_max)
            if negative_list:
                negative_list = neg_sort.traverse_forward()
                negative_list = [-num for num in negative_list]
            return positive_list + negative_list
        else:
            if positive_list:
                positive_list = pos_sort.traverse_forward()
            if negative_list:
                negative_list = neg_sort.traverse_backward(neg_max)
                negative_list = [-num for num in negative_list]
            return negative_list + positive_list

# Test script
def print_tree(node, level=0, index=1):
    print("  " * level + f"Node (Layer: {node.layer}, index: {index}, parent_index: {node.parent_index}, Values: {node.values}, Org Values: {node.org_values}, Bitmask: {bin(node.bitmask)})")
    if node.child:
        for i, child in enumerate(node.child):
            if child is not None:
                print_tree(child, level + 1, i)

import random
import time

# test_values = [0.05, 0.2, 0.6, 1.5, 10, 50, 100, 500, 1000, 4999, 9998]
random.seed(42)
# siz = 1000
# siz = 1_000_000
siz = 100_000
# siz = 100
test_values = [random.uniform(0, siz*100) for _ in range(siz)]
# print("test_values", test_values[:20], test_values[-20:])

# num_segments = 100
# num_segments = 10
num_segments = 64
use_bitwise = True
use_dictionary = False


start_time = time.time()
root = RangeSplitSort(test_values, num_segments, use_bitwise=use_bitwise, use_dictionary=use_dictionary)
insertion_time = time.time() - start_time
print("insertion_time", insertion_time)

# print("\nTree Structure:")
# print_tree(root)

# # Test search
# print("\nTesting search...")
# new_values = [5000, 10, 2003]
# for val in new_values:
#     res = root.search(val)
#     print("search", val, res)
    
# Test insertion
print("\nTesting insertion...")
new_values = [2003, 0.23, 71]
for val in new_values:
    print("inserting", val)
    root.insert(val)

# print("\nTree Structure:")
# print_tree(root) 

# # Test find_next
# print("\nTesting find_next...")
# # test_values2 = [0.01, 3, 10, 50, 72, 100, 500, 1000]
# for val in test_values2:
#     # next_val, next_node = root.find_next(val)
#     # print(f"Next value after {val}: {next_val}")
#     # status, next_val, next_node = root.find_next_side(val)
#     print("START find_next", val)
#     status, next_val, next_node, index = root.find_next(val)
#     print(f"Next value after  {status}: {val}: {next_val}: {index}")
#     # print("START find_prev", val)
#     # status, next_val, next_node, index = root.find_prev(val)
#     # print(f"Previous value before  {status}: {val}: {next_val}: {index}")




print("EXECUTE root.traverse_forward()")
res = root.traverse_forward()
# print("res", res)
print(res[:20], res[-20:])
# print("EXECUTE root.traverse_forward(72)")
# res = root.traverse_forward(72)
# print(res[:20], res[-20:])

print("EXECUTE root.traverse_backward()")
res = root.traverse_backward()
print(res[:20], res[-20:])


# test_values = [v - siz/2 for v in test_values]
# print("EXECUTE RangeSplitSort.sort()")
# res = RangeSplitSort.sort(test_values)
# print(res[:20], res[-20:])


import time
import pandas as pd
import random

# Example usage and testing
def run_test(size, num_segments=64, range_scale=1, use_integer=True, use_bitwise=True):

    print("run_test", "size", size, "num_segments", num_segments, "range_scale", range_scale, "use_integer", use_integer, "use_bitwise", use_bitwise)
    print("range", size*range_scale)
    # Insert test
    if use_integer:
        vals = [random.randint(0, size*range_scale) for _ in range(size)]
    else:
        vals = [random.uniform(0, size*range_scale) for _ in range(size)]

    start_time = time.time()
    test_bitmap = RangeSplitSort(vals, num_segments, use_bitwise=use_bitwise, use_dictionary=True)
    insertion_time = time.time() - start_time

    # Contains test (middle element)
    # contains_value = (size // 2) - ((size // 2) % element_step)  # Closest inserted value to the middle
    contains_value = size*range_scale // 2  # Closest inserted value to the middle
    start_time = time.time()
    contains_result = test_bitmap.search(contains_value)
    contains_time = time.time() - start_time

    # Set test (middle element)
    start_time = time.time()
    contains_result = test_bitmap.insert(contains_value)
    set_time = time.time() - start_time
    
    # Find next test
    start_time = time.time()
    next_result = test_bitmap.find_next(contains_value)
    next_time = time.time() - start_time

    # Find previous test
    start_time = time.time()
    next_result = test_bitmap.find_prev(contains_value)
    previous_time = time.time() - start_time

    # Traverse sorted test
    start_time = time.time()
    sorted_traversal = test_bitmap.traverse_forward()
    traversal_time = time.time() - start_time
    
    vals = [v - int(size/2) for v in vals]
    
    # sort
    start_time = time.time()
    sorted_data = RangeSplitSort.sort(vals)
    sort_time = time.time() - start_time
    
    num_layers = test_bitmap.max_layer+1
    
    # Return the results for this test
    return {
        "Test Data Size": int(len(vals)),
        "Insertion Time (s)": round(set_time, 6),
        "Contains Time": round(contains_time, 6),
        "Bulk Insert Time (sorting)": round(insertion_time, 6),
        "Next Time (s)": round(next_time, 6),
        "Previous Time (s)": round(previous_time, 6),
        "Traversal Time": round(traversal_time, 6),
        "Sort Time": round(sort_time, 6),
        "Number of Layers": round(num_layers, 6)
    }


# Test configurations
sizes = [100, 10000, 1_000_000]  # Corrected sizes to satisfy multi-layer design requirements
# sizes = [100, 10000, 100_000]  # Corrected sizes to satisfy multi-layer design requirements
num_segments = 64

# num_segments, range_scale, use_integer, use_bitwise
# for p in [(64, 1, True, True), (64, 100, True, True), (64, 1, False, True), (64, 100, False, True)]:
for p in [(64, 100, True, False), (64, 100, True, True)]:
    # Run and collect results
    # use_integer = False
    print("        ----------------")
    print("use_integer", p[0], "num_segments", p[1])
    results = [run_test(size, num_segments=p[0], range_scale=p[1], use_integer=p[2], use_bitwise=p[3]) for size in sizes]
    
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
