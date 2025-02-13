# RangeSplitSort: A Practical and Efficient Sorting and Search Structure with Bitmap Optimization

RangeSplitSort is an efficient and space-saving sorting and searching structure that improves performance over traditional sorting algorithms and data structures. Unlike binary search trees and balanced trees that operate in $O(\log_2 N)$, RangeSplitSort uses a **divide-and-conquer recursive approach** with a user-defined base (e.g., 64), reducing search depth and enhancing lookup speeds. Additionally, **bitwise operations** help efficiently find next values, minimizing redundant computations. Unlike **LayeredBitmap**, which was previously introduced and which stores data directly in bitmaps, RangeSplitSort can handle **floating-point numbers natively** and optimizes space usage by only storing bitmaps for range indication. This results in an average complexity of **$O(\log_B N)$**, where B is the segment base, offering significant performance improvements.

---

### **1. Introduction**

Sorting and searching are essential operations in computing, used in databases, file systems, and many applications requiring quick lookups. Standard techniques like **merge sort ( $O(N \log_2 N)$ )**, **quick sort ( $O(N \log_2 N)$ )**, and **binary search trees ( $O(\log_2 N)$ )** have been widely adopted.

RangeSplitSort offers a practical solution by **dividing the dataset into multiple segments (B instead of 2)**, significantly reducing tree depth. It also introduces **bitwise operations** for near-instantaneous next-value lookups, saving computational effort. Unlike **LayeredBitmap**, which explicitly stores all data points as bitmaps, RangeSplitSort efficiently tracks ranges while supporting **floating-point numbers** natively.

When analyzing the growth of a number using logarithms, the choice of base significantly affects the output scale. For instance, given $N = 1,000,000,000$, we find that:

$$
\log_2(1,000,000,000) \approx 29.90
$$

$$
\log_{64}(1,000,000,000) \approx 4.98
$$

This demonstrates that using a higher base, such as 64, drastically reduces the logarithmic value compared to base 2. The reason is that $\log_B N$ is inversely proportional to the base, meaning a larger base leads to a smaller result. In computational and algorithmic analysis, choosing a higher base can simplify expressions and improve efficiency when dealing with large-scale problems, as it effectively compresses the growth rate representation. 

Note: in Big O notation convention, both $O(\log_2 N)$ and $O(\log_{64} N)$ are expressed as $O(\log)$, but it doen't represent the performance accuracately in the today's upper range as $O(\log_{64} N)$ behaves practically as constant as $O(\log_{64} N) < O(8)$ even in extreme condition as N<280 TB whereas in $O(\log_{2} N)$ goes up to $O(48)$.  The difference should get quite large in nested queries as the difference exponentiates.


---

### **2. Algorithm Design**

#### **2.1 Divide-and-Conquer Recursive Segmentation**

- Data is recursively divided into **B segments**, instead of just two as in binary trees.
- A dynamically calculated **divider** ensures balanced data distribution.
- A **bitmap (`bitmask`)** keeps track of child nodes, enabling rapid traversal while using minimal memory.

#### **2.2 Efficient Search Mechanism**

- Instead of traditional **$O(\log_2 N)$** search complexity, RangeSplitSort operates in **$O(\log_B N)$**.
- **Bitwise operations** enable constant-time checks for child nodes.
- If an exact match isn't found, the closest higher or lower value is retrieved efficiently.

#### **2.3 Fast Next and Previous Value Lookup**

- Unlike traditional sorted structures requiring **$O(\log_2 N)$** or more operations, **RangeSplitSort** uses **bitwise operations** to locate the next set bit in the bitmap.
- This results in **$O(1)$ if bitwise operations can be done in constant time, otherwise $O(\log_B N)$**, significantly improving search efficiency.

#### **2.4 Handling Negative Values**

- This structure does not natively support negative values, but it can be implemented by splitting the dataset into separate positive and negative lists and converting negative values to positive equivalents for processing, which an $O(1)$ process.

---

### **3. Performance Comparison**

#### **3.1 Time Complexity**

| Operation         | RangeSplitSort                                                   | Binary Search Tree | AVL/Red-Black Tree | Merge Sort | Quick Sort |
| ----------------- | ---------------------------------------------------------------- | ------------------ | ------------------ | ---------- | ---------- |
| Insertion         | $O(\log_B N)$                                                      | $O(\log_2 N)$        | $O(\log_2 N)$        | $O(N \log_2 N)$ | $O(N \log_2 N)$ |
| Search            | $O(\log_B N)$                                                      | $O(\log_2 N)$        | $O(\log_2 N)$        | N/A        | N/A        |
| Next Value Lookup | $O(\log_B N)$                                                            | $O(\log_2 N)$          | $O(\log_2 N)$          | N/A        | N/A        |
| Sorting           | $O(N \log_B N)$                                                    | $O(N \log_2 N)$      | $O(N \log_2 N)$      | $O(N \log_2 N)$ | $O(N \log_2 N)$ |

#### **3.2 How It Compares to Other Sorting Algorithms**

- **Merge Sort & Quick Sort:** Both require **$O(N \log_B N)$** time to sort, making them fast but unsuitable for dynamic insertions and searches.
- **Binary Search Trees & AVL Trees:** While these allow **$O(\log_2 N)$** insertions and searches, they rely on binary divisions, increasing tree depth and lookup time.
- **RangeSplitSort:** By segmenting data into B groups (e.g., 64), the tree depth is significantly smaller, improving search performance to **$O(\log_B N)$**. Additionally, **bitwise operations** enable next-value lookups in near **constant time**, making it ideal for ordered data processing.

#### **3.3 Space Complexity**

- Similar to other hierarchical structures, **$O(N \log_2 N)$** space is required to cover the whole tree. To reach 10 layer, two or more items need to share the significant digits of 60 digits.  So, it is very unlikely to go further unless the dataset is pathologically structured. 
- Unlike **LayeredBitmap**, which explicitly stores every data point as a bitmap, RangeSplitSort **only tracks ranges**, leading to significant memory savings.
- **Floating-point numbers are fully supported**, unlike certain bitmap-based approaches requiring conversion.

#### **3.4 Experimental Results**

- The time complexities claimed in this paper are based on results from a demonstration-purpose program, validating the theoretical expectations.
- Next-value lookups achieve near **constant-time performance** thanks to bitmap-based optimization.
- Memory consumption is notably lower than **LayeredBitmap**, since only range tracking is stored instead of individual values.

---

### **4. Practical Applications**

- **Database Indexing:** Faster range queries and ordered retrieval.
- **Real-Time Systems:** Quick insertions and searches with minimal latency.
- **Big Data Processing:** Efficient sorting and querying with reduced computational overhead.

---

### **5. Conclusion**

RangeSplitSort presents a **divide-and-conquer recursive sorting approach** that significantly reduces tree depth and search time. Unlike binary trees that rely on **$\log_2 N$** complexity, RangeSplitSort achieves **$\log_B N$** (with B being configurable, e.g., 64). Compared to **LayeredBitmap**, it supports **floating-point values** while using far less memory by tracking only range indications in bitmaps.

This makes it an excellent choice for applications requiring **fast dynamic inserts, efficient sorted data retrieval, and memory-conscious storage**. Future work will explore **parallelized processing and GPU acceleration** to further improve performance.
