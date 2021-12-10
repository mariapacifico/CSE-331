Project 3: AVL Trees
====================

**Due: Thursday, November 4th @ 8:00 pm EST**

_This is not a team project. Do not copy someone else’s work._

Assignment Overview
-------------------

[AVL trees](https://en.wikipedia.org/wiki/AVL_tree) are a self-balancing [binary search tree (BST)](https://en.wikipedia.org/wiki/Binary_search_tree) optimized to maintain logarithmic-time operations regardless of the order in which data is inserted and deleted. First introduced by Soviet computer scientists Georgy Adelson-Velsky and Evgenii Landis in their 1962 paper "[An algorithm for the organization of information](https://zhjwpku.com/assets/pdf/AED2-10-avl-paper.pdf)," AVL trees have stood the test of time and remain a popular choice when a space-efficient data structure supporting fast insertion/search/deletion is necessary.

![](img/avl.gif)![avl.gif](https://s3.amazonaws.com/mimirplatform.production/files/67ed7546-bc85-439f-b45b-f2555ac94d18/avl.gif)

To motivate AVL trees, it is worth considering a common problem that arises in traditional BSTs. BSTs are designed to perform logarithmic-time insertion/search/deletion but may operate at linear time if data is inserted or deleted according to certain patterns which cause the BST to become _unbalanced_. For example, when data is inserted into a traditional BST in sorted (or reverse-sorted) order, the BST will grow leaves in a single direction and effectively turn into a linked list.

![](img/balance.png)![balance.png](https://s3.amazonaws.com/mimirplatform.production/files/b942a423-e8cb-4ba8-b777-10f47c616831/balance.png)

If our dataset is small, this may not be a problem—but when we're working with thousands or millions of records in a database, the difference between logarithmic and linear is astounding!

![bigogrowth.png](https://s3.amazonaws.com/mimirplatform.production/files/3e0227a1-6c58-4141-8129-ebabe788049d/bigogrowth.png)

AVL trees improve upon traditional BSTs by _self-balancing_ in order to guarantee logarithmic-time operations. In this project, you will be implementing an AVL tree from scratch in Python.

Assignment Notes, Tips, Tricks:
-------------------------------

1.  In this project, you'll be using Python `Generator` objects to traverse a tree in a space-efficient manner. Unlike a traversal returned in the form `List[Node]` using _O(n)_ space, a traversal returning a generator will use _O(1)_ space by _yielding_ each `Node` in a sequential, on-demand manner. See [this link](https://realpython.com/introduction-to-python-generators/) for a nice introduction to `Generator`s in Python!
2.  One of the most common errors in this project is forgetting or incorrectly updating the height and rebalancing within functions that change the tree structure (insert). Read the notes we put under the function description in the specs carefully and think about how you can use recursion/call stack to help you rebalance the tree. What is the call stack's relationship to the node which you removed/inserted?
3.  AVL Trees are more complicated structures than what you have worked with before but if you boil each function down to the different cases within them, then it begins to look a lot simpler. Try to decompose each function into what checks/cases you need to look for before an operation. Checks like is the node I'm operating on is the origin? Does this node have a parent pointer? Is there a right node before I make a call on node.right? Am I updating the correct pointers?
4.  The Debugger is your friend!! Do not be scared to use it, it is worth the extra time to learn its functionality if you haven't yet. Use it to determine if what you think your code is doing, is actually what it's doing!! It's the most helpful tool when trying to figure out why your more complex functions aren't working.
5.  The rotation functions are extremely important to implementing AVL trees. That's why we give you left\_rotate. The right\_rotate function is very similar, with only a few modifications needed. **However, Onsay will be going over right\_rotate in class. It would be in your best interest to attend that lecture to get the solution and conceptual understanding required for right\_rotate.**

Assignment Specifications
-------------------------

#### class Node:

_DO NOT MODIFY the following attributes/functions_

*   **Attributes**
    
    *   **value: T:** Value held by the `Node`. Note that this may be any type, such as a `str`, `int`, `float`, `dict`, or a more complex object.
    *   **parent: Node:** Reference to this `Node`'s parent `Node` (may be `None`).
    *   **left: Node:** Reference to this `Node`'s left child `Node` (may be `None`).
    *   **right: Node:** Reference to this `Node`'s right child `Node` (may be `None`).
    *   **height: int:** Number of levels of `Node`s below (the height of a leaf `Node` is 0).
*   **\_\_init\_\_(self, value: T, parent: Node = None, left: Node = None, right: Node = None) -> None**
    
    *   Constructs an AVL Tree node.
    *   **value: T:** Value held by the `Node`.
    *   **parent: Node:** Reference to this `Node`'s parent `Node` (may be `None`).
    *   **left: Node:** Reference to this `Node`'s left child `Node` (may be `None`).
    *   **right: Node:** Reference to this `Node`'s right child `Node` (may be `None`).
    *   **Returns:** `None`.
*   **\_\_str\_\_(self) -> str** and **\_\_repr\_\_(self) -> str**
    
    *   Represents the `Node` as a string in the form `<value_held_by_node>`. Thus, `<7>` indicates a `Node` object holding an `int` value of 7, whereas `<None>` indicates a `Node` object holding a value of `None`.
    *   Note that Python will automatically invoke this function when using printing a `Node` to the console, and PyCharm will automatically invoke this function when displaying a `Node` in the debugger.
    *   Call this with `str(node)` (rather than `node.__str__()`).
    *   **Returns:** `str`.

#### class AVLTree:

_DO NOT MODIFY the following attributes/functions_

*   **Attributes**
    
    *   **origin: Node:** Root node of the entire `AVLTree` (may be `None`). This naming convention helps us disambiguate between when we are referring to the root of the entire `AVLTree` and the root of a subtree within the `AVLTree`. In fact, any given `Node` object within an `AVLTree` can be thought of as being the root of the subtree of all `Node`s below—and `origin` is the uppermost such root in our tree.
    *   **size: int:** Number of nodes in the `AVLTree`.
*   **\_\_init\_\_(self) -> None**
    
    *   Construct an empty `AVLTree`. Initialize the `origin` to `None` and set the size to zero.
    *   **Returns:** `None`.
*   **\_\_str\_\_(self) -> str** and **\_\_repr\_\_(self) -> str**
    
    *   Represents the `AVLTree` as a string in level-order form, with each `Node` displayed as `<value> (<value_of_parent>, height)` in the resulting string.
    *   Note that Python will automatically invoke this function when using printing a `Node` to the console, and PyCharm will automatically invoke this function when displaying a `Node` in the debugger.
    *   Call this with `str(node)` (rather than `node.__str__()`).
    *   **Returns:** `str`.
*   **height(self, root: Node) -> int**
    
    *   Return the height of a subtree in the AVL tree, properly handling the case of `root = None`. Recall that the height of an empty subtree is -1.
    *   _Time / Space: O(log n) / O(1)_.
    *   **root: Node:** The root `Node` of the subtree being measured.
    *   **Returns:** Height of the subtree at `root`, i.e., the number of levels of `Node`s below this `Node`. The height of a leaf `Node` is 0, and the height of a `None`\-type is -1.
*   **left\_rotate(self, root: Node) -> Node**
    
    *   Perform a left rotation on the subtree rooted at `root`. Return root of new subtree after rotation.
    *   _Time / Space: O(1) / O(1)_.
    *   **root: Node:** The root `Node` of the subtree being rotated.
    *   **Returns:** Root of new subtree after rotation.
    
*   **remove(self, root: Node, val: T) -> Node**
    
    *   Remove the node with the value `val` from the subtree rooted at `root`, and return the root of the balanced subtree following removal.
    *   If `val` does not exist in the AVL tree, do nothing.
    *   If the node being removed has two children, swap the value of this node with its **predecessor** and recursively remove this predecessor node (which contains the value to be removed after swapping and is guaranteed to be a leaf).
        *   Although one technically _could_ swap values with the successor node in a two-child removal, our testcases assume you will swap with the predecessor.
    *   _Time / Space: O(log n) / O(1)_.
    *   **root: Node:** The root `Node` of the subtree from which to delete `val`.
    *   **val: T:** The value to be deleted from the subtree rooted at `root`.
    *   **Returns:** Root of new subtree after removal and rebalancing (could be the original root).

_IMPLEMENT the following functions_

*   **right\_rotate(self, root: Node) -> Node**
    
    *   Perform a right rotation on the subtree rooted at `root`. Return root of new subtree after rotation.
    *   In class note
    *   This **should be nearly identical to `left_rotate`**, with only a few lines differing. Team 331 agreed that giving one rotation helps ease the burden of this project — but writing the other rotation will be a good learning experience!
    *   _Time / Space: O(1) / O(1)_.
    *   **root: Node:** The root `Node` of the subtree being rotated.
    *   **Returns:** Root of new subtree after rotation.

*   **balance\_factor(self, root: Node) -> int**
    
    *   Compute the balance factor of the subtree rooted at `root`.
    *   Recall that the balance factor is defined to be `h_L - h_R` where `h_L` is the height of the left subtree beneath this `Node` and `h_R` is the height of the right subtree beneath this `Node`.
    *   Note that in a properly-balanced AVL tree, the balance factor of all nodes in the tree will be in the set {-1, 0, +1}, as rebalancing will be triggered when a node's balance factor becomes -2 or +2.
    *   The balance factor of an empty subtree (`None`\-type `root`) is 0.
    *   To stay within time complexity, keep the `height` attribute of each `Node` object updated on all insertions/deletions/rebalances, then use `h_L = left.height` and `h_R = right.height`.
    *   _Time / Space: O(1) / O(1)_.
    *   **root: Node:** The root `Node` of the subtree on which to compute the balance factor.
    *   **Returns:** `int` representing the balance factor of `root`.
*   **rebalance(self, root: Node) -> Node**
    
    *   Rebalance the subtree rooted at `root` (if necessary) and return the new root of the resulting subtree.
    *   Recall that rebalancing is only necessary at this `root` if the balance factor `b` of this `root` satisfies `b >= 2 or b <= -2`.
    *   Recall that there are [four types of imbalances possible in an AVL tree](https://en.wikipedia.org/wiki/AVL_tree#Rebalancing), and that each requires a different sequence of rotation(s) to be called.
    *   _Time / Space: O(1) / O(1)_.
    *   **root: Node:** The root `Node` of the subtree to be rebalanced.
    *   **Returns:** Root of new subtree after rebalancing (could be the original root).
*   **insert(self, root: Node, val: T) -> Node**
    
    *   Insert a node with `val` into the subtree rooted at `root`, returning the root node of the balanced subtree after insertion.
    *   If `val` already exists in the AVL tree, do nothing.
    *   Should update `size` and `origin` attributes of `AVLTree` if necessary and correctly set parent/child pointers when inserting a new `Node`
    *   Should update the `height` attribute and call `rebalance` on all `Node` objects affected by the insertion (ancestor nodes directly above on path to origin).
    *   Easiest to implement recursively.
    *   _Time / Space: O(log n) / O(1)_.
    *   **root: Node:** The root `Node` of the subtree in which to insert `val`.
    *   **val: T:** The value to be inserted in the subtree rooted at `root`.
    *   **Returns:** Root of new subtree after insertion and rebalancing (could be the original root).
*   **min(self, root: Node) -> Node**
    
    *   Find and return the `Node` with the smallest value in the subtree rooted at `root`.
    *   Easiest to implement recursively.
    *   _Time / Space: O(log n) / O(1)_.
    *   **root: Node:** The root `Node` of the subtree in which to search for a minimum.
    *   **Returns:** `Node` object containing the smallest value in the subtree rooted at `root`.
*   **max(self, root: Node) -> Node**
    
    *   Find and return the `Node` with the largest value in the subtree rooted at `root`.
    *   Easiest to implement recursively.
    *   _Time / Space: O(log n) / O(1)_.
    *   **root: Node:** The root `Node` of the subtree in which to search for a maximum.
    *   **Returns:** `Node` object containing the largest value in the subtree rooted at `root`.
*   **search(self, root: Node, val: T) -> Node**
    
    *   Find and return the `Node` with the value `val` in the subtree rooted at `root`.
    *   If `val` does not exist in the subtree rooted at `root`, return the `Node` below which `val` would be inserted as a child. For example, on a balanced 1-2-3 tree (with 2 on top and 1, 3 as children), `search(node_2, 0)` would return `node_1` since the value of 0 would be inserted as a left child of `node_1`.
    *   Easiest to implement recursively.
    *   _Time / Space: O(log n) / O(1)_.
    *   **root: Node:** The root `Node` of the subtree in which to search for `val`.
    *   **val: T:** The value being searched in the subtree rooted at `root`.
    *   **Returns:** `Node` object containing `val` if it exists, else the `Node` object below which `val` would be inserted as a child.
*   **inorder(self, root: Node) -> Generator\[Node, None, None\]**
    
    *   Perform an inorder (left, current, right) traversal of the subtree rooted at `root` using a [Python generator](https://realpython.com/introduction-to-python-generators/).
    *   Use `yield` to immediately generate an element in your function, and `yield from` to generate an element from a recursive function call.
    *   Do not yield (generate) `None`\-types.
    *   _Time / Space: O(n) / O(1)_.
        *   Although we will traverse the entire tree and hence incur O(n) time, our use of a generator will keep us at constant space complexity since elements are yielded one at a time! This is a key advantage of returning a generator instead of a list.
    *   **root: Node:** The root `Node` of the subtree currently being traversed.
    *   **Returns:** `Generator` object which yields `Node` objects only (no `None`\-type yields). Once all nodes of the tree have been yielded, a `StopIteration` exception is raised.
*   **preorder(self, root: Node) -> Generator\[Node, None, None\]**
    
    *   Perform a preorder (current, left, right) traversal of the subtree rooted at `root` using a [Python generator](https://realpython.com/introduction-to-python-generators/).
    *   Use `yield` to immediately generate an element in your function, and `yield from` to generate an element from a recursive function call.
    *   Do not yield (generate) `None`\-types.
    *   _Time / Space: O(n) / O(1)_.
        *   Although we will traverse the entire tree and hence incur O(n) time, our use of a generator will keep us at constant space complexity since elements are yielded one at a time! This is a key advantage of returning a generator instead of a list.
    *   **root: Node:** The root `Node` of the subtree currently being traversed.
    *   **Returns:** `Generator` object which yields `Node` objects only (no `None`\-type yields). Once all nodes of the tree have been yielded, a `StopIteration` exception is raised.
*   **postorder(self, root: Node) -> Generator\[Node, None, None\]**
    
    *   Perform a postorder (left, right, current) traversal of the subtree rooted at `root` using a [Python generator](https://realpython.com/introduction-to-python-generators/).
    *   Use `yield` to immediately generate an element in your function, and `yield from` to generate an element from a recursive function call.
    *   Do not yield (generate) `None`\-types.
    *   _Time / Space: O(n) / O(1)_.
        *   Although we will traverse the entire tree and hence incur O(n) time, our use of a generator will keep us at constant space complexity since elements are yielded one at a time! This is a key advantage of returning a generator instead of a list.
    *   **root: Node:** The root `Node` of the subtree currently being traversed.
    *   **Returns:** `Generator` object which yields `Node` objects only (no `None`\-type yields). Once all nodes of the tree have been yielded, a `StopIteration` exception is raised.
*   **levelorder(self, root: Node) -> Generator\[Node, None, None\]**
    
    *   Perform a level-order (breadth-first) traversal of the subtree rooted at `root` using a [Python generator](https://realpython.com/introduction-to-python-generators/).
    *   Use the builtin `queue.SimpleQueue` class to maintain your queue of children throughout the course of the traversal—[see the official documentation here.](https://docs.python.org/3/library/queue.html#queue.SimpleQueue)
    *   Use `yield` to immediately generate an element in your function, and `yield from` to generate an element from a recursive function call.
    *   Do not yield (generate) `None`\-types.
    *   _Time / Space: O(n) / O(n)_.
        *   We will traverse the entire tree and incur O(n) time. In addition, the queue we must use for an inorder traversal will grow to size n/2 = O(n) just before beginning the final level of leaf nodes in the case of a [perfect binary tree.](https://www.programiz.com/dsa/perfect-binary-tree)
    *   **root: Node:** The root `Node` of the subtree currently being traversed.
    *   **Returns:** `Generator` object which yields `Node` objects only (no `None`\-type yields). Once all nodes of the tree have been yielded, a `StopIteration` exception is raised.

Application
-----------

Your CEO at Solr was very impressed by your work on finding the maximum areas of histograms a few weeks ago. Now, she would like your help picking the employees of the month at your branch. Solr likes to recognize multiple people each month for their outstanding achievements, **so there is one employee of the month at each level of the hierarchy.** Each employee submits nominations for employee of the month for each level of the organization, starting with the junior employees, all the way to the CEO. Your task is to write a function to determine who has the **maximum number of nominations at each level of the hierarchy.** Conveniently, Solr's organization is set up so that every two employees report to one manager, creating a hierarchy that looks like a binary tree!

**_class Employee(Node) :_**

*   **Attributes:**  
    *   **Value, Parent, Left, Right, Height** inherited from Node Class
        *   **NOTE:** Value will represent the employee's name
    *   **Nominations: int:** The total amount of nominations this employee got through the month
    *   **def \_\_init\_\_(self, value: T, parent: Node = None,**  
         **left: Node = None, right: Node = None, nominations: int = 0, )**:
        
    *   **\_\_str\_\_(self) -> str** and **\_\_repr\_\_(self) -> str:**  
        *   represent employees as strings, for debugging reasons
    *   **\_\_eq\_\_(self, other) -> Bool:**  
        *   overrides '=' operator for employee class

**class Company():**

*   This class is purely for testing purposes, you shouldn't worry about understanding it. It represents the company made up of employee nodes with a simple BST implementation that uses the employee's name for comparisons.

_Implement the following function:_

**_findBestEmployees( CEO: Employee) -> List\[Employee\]_**

*   This function takes in a CEO, the top employee of the company hierarchy, and returns a list of the best employee at each level of the company. Specifically a list of employee objects.
*   If two employees tie in nominations the leftmost employee will win Employee of the month.
*   If the company is empty, this should return an empty list
*   **CEO: Employee:**  An instance of the Employee class which has pointers to their 2 subordinates.
*   **Returns:** A list of employees who had the most nominations among all employees at their level (tree depth). 
*   Time Complexity: O(n) where n is the total amount of employees in the company (size of the tree)
*   Space complexity: O(n) 

**Guarantees**

*   No duplicate names
*   Size of the company 0 > s >= 40
*   Every employee has at most 2 people reporting to him (2 children) and reports to 1 employee (1 parent)

**Examples:**

Ex. 1

![ex1.png](https://s3.amazonaws.com/mimirplatform.production/files/570fd605-5d3a-4488-a290-6c1d4f7d738c/ex1.png)

_Input:_ Employee(nominations = 9)

_Output:_ \[Employee(9), Employee(17), Employee(12)\]

In this case, there are three levels of hierarchy (this would just be a subset of the company). The highest level is automatically the employee with the most nominations since there are no competitors. The second level has two employees, one with 17 nominations, and another with 7, so the employee with 17 wins. Lastly, in the final level, the employee with 12 nominations wins, since they have the maximum number of nominations for that level.

Ex. 2

![ex2.png](https://s3.amazonaws.com/mimirplatform.production/files/bcf62233-9a0c-47aa-b0df-3764472ee434/ex2.png)

_Input:_ Employee(nominations = 13)

_Output:_ \[Employee(13), Employee(11), Employee(12)\]

Again, in this case, there are three levels of hierarchy. The highest level automatically wins, and the employee with 11 nominations at the second level wins. Lastly, in the final level, there are two employees with 12 nominations. In this case, the leftmost employee should win employee of the month. Is this fair? Probably not, but this isn't an ethics course.

**Tips and Tricks:**

*   Think about a traversal that could be useful in this situation. Remember, we are picking the maximum of each **level.**
*   You may find that a queue is useful for this problem. If you take a look at the imports section, we have imported the python queue module. Python's [queue.SimpleQueue](https://docs.python.org/3/library/queue.html#queue.SimpleQueue) should be pretty useful.
*   The application can be done in O(d) space where d is the depth of the tree. This should be another hint towards what traversal should be used here.

**Extra Credit Application**

After the employee of the month was anounced, you are not the one of the list. However, you want to impress your boss by implementing function to finding the sum of all values in the tree from given range. You realy desire to make your boss impressive. So, your function will work with binary search tree that can be updated. 

**_class NodeWithSum(Node) :_**

*   **Attributes:**  
    *   **Value, Parent, Left, Right, Height** inherited from Node Class
    *   **Sum: int:** The total sum of all values in subtree that node is root
    *   **\_\_init\_\_(self, val) -> None:**  
        *   Initialize the node with given value
    *   **\_\_str\_\_(self) -> str** and **\_\_repr\_\_(self) -> str:**  
        *   represent employees as strings, for debugging reasons

**class _TreeWithSum(AVLTree)_**

*   This class is purely for testing purposes, you shouldn't worry about understanding it. It represents the simple BST implementation

_Implement the following function:_

**_findSum( tree: BSTWithSum, range\_values: Tuple\[int, int\] ) -> int_**

*   Find the sum of all values in the tree which is in the given range
*   **tree: _BSTWithSum_:** An instance of TreeWithSum class to find sum
*   **rangeValues:** A tuple of int and int represented the start and end of interesting values
*   **Returns:** Sum of all values in the given tree with inside the given range
*   Time Complexity: O(log(n)) where n is the total amount of number in the tree
*   Space complexity: O(1) 
*   **To get the extra credit, you need to complete the function within required time and space complexity**

**Guarantees**

*   No duplicate value in the tree
*   Given range is correct mean if given range is \[l , r\], l <= r and both l and r more than 0
*   You can implement function on TreeWithSum class if it necessary

**Examples:**

Ex. 1

![Screen Shot 2021-10-20 at 6.35.49 PM.png](https://s3.amazonaws.com/mimirplatform.production/files/5ed7b0a8-00c0-4506-b09f-f877bf66b9f2/Screen%20Shot%202021-10-20%20at%206.35.49%20PM.png)

_Tree given aboved_

Input: Range Value = \[0, 10\]

Output: 24 since all values within \[0, 10\] are 1, 3, 5, 6, and 9. Sum of all values is 1 + 3 + 5 + 6 + 9 = 24

Input: Range Value = \[3, 9\]

Output: 23 since all values within \[3, 9\] are 3, 5, 6, and 9. Sum of all values is 3 + 5 + 6 + 9 = 23

Input: Range Value = \[6, 15\]

Output: 27 since all values within \[6, 15\] are 6, 9, and 12. Sum of all values is 6 + 9 + 12 = 27

Submission
----------

#### Deliverables

Be sure to upload the following deliverables in a .zip folder to Mimir by 8:00p EST on Thursday, November 4th.

    Project3.zip
        |— Project3/
            |— README.xml      (for project feedback)
            |— __init__.py     (for proper Mimir testcase loading)
            |— AVLTree.py      (contains your solution source code)
    

#### Grading

*   Tests (70)
    *   Point distribution is specified in Mimir test cases
*   Manual (30)
    *   Time and space complexity points are **all-or-nothing** for each function. If you fail to meet time **or** space complexity in a given function, you do not receive manual points for that function.
    *   You can lose up to 3 points for missing docstrings across the full project
    *   `AVLTree` time & space: \_\_/20
        *   `right_rotate`: \_\_/2
        *   `balance_factor`: \_\_/2
        *   `rebalance`: \_\_/2
        *   `insert`: \_\_/2
        *   `min`: \_\_/1
        *   `max`: \_\_/1
        *   `search`: \_\_/2
        *   `inorder`: \_\_/2
        *   `preorder`: \_\_/2
        *   `postorder`: \_\_/2
        *   `levelorder`: \_\_/2
    *   `Application`: \_\_/10
*   Extra Credit\[5\]
    *   Extra Credit: \_\_/5
    *   You must meet time complexity O(log(n)) and space complexity O(1) to earn extra credit

Appendix
--------

#### Authors

Project authored by Jacob Caurdy, Lukas Richters, Bank Premsri.

![](img/bestworst.png)