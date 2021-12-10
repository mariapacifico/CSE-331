Introduction
------------

Humanity has just initiated radio contact with intelligent extraterrestrial life for the first time in history. Technically speaking, contact was initiated with us... The technological development of the alien civilization makes it so that in their eyes, we are essentially amoebae. How do we know this? For one, they were able to send a message to us containing text in \*our languages\* despite the fact that we made no effort to send a message anywhere. Furthermore, they also sent us a worldwide weather forecast which was verified to be accurate to the _millisecond_. The message was as follows: \`Atmospheric Noise. お手並み拝見\`.

The current theory about the goal of that message is that the aliens have some way to control the static noise present in the earth's atmosphere and have hidden a message inside to simultaneously demonstrate their technological prowess and determine whether humanity is worth their time. The whole world has mobilized teams of people to attempt this challenge that will no doubt go down in history. You are a member of a team that believes that the answer lies in the use of atmospheric noise as a 'true' random number generator.

If this hypothesis is indeed true and the aliens can quite literally control what has long been considered completely unpredictable, the implications are chilling, to say the least.  

As such, the team has been collecting random number observations based on the atmospheric noise and storing them in a Binary Search Tree since they wanted the data to be sorted but also to be efficient when it comes to insertions/deletions. As a new member to the team eager to prove yourself, you have been assigned the task of performing some operations on this tree and attempting to discover a hidden message in the numbers. Specifically, your current assignment is to calculate the smaller product for all the nodes. What might that be you ask? Read on to find out...

P.S This story was partly inspired by the book _The Three Body Problem, by Cixin Liu_. We recommend everyone to give it a read! Maybe it contains a hint to the contents of the upcoming assignments?! 😮

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Challenge
---------

### Overview

You are given the root node to a binary search tree of size **n**. Aside from the `value`, `left` and `right` variables, nodes in this tree also have an `index` variable. The basic premise is to calculate the **smaller product** of a node.

The smaller product of a node is the product of all nodes less than the given node. You have to calculate the smaller product for **all** the nodes in the tree in **linear time and space** and return the results in a list. The position of the result in the output list for a given node is determined by the index variable of the node.

To be more concise, `output_list[node.index] = product of all values < node.val`

**Note:**

*   **If the node is the smallest in the BST, the smaller product would be calculated as `None`.**
*   **If the node is the second smallest in the BST, the smaller product would just be the smallest node in the tree**

### Code

Modify the following function:

    def smaller_product(root: Node) -> List[int]:
        """
        Time Complexity: O(n) where n is number of nodes in the BST
        Space Complexity: O(n) where n is number of nodes in the BST

        :param root: Root of the BST
        :return: List of ints where for each node in tree,
                 list[node's index] = product of all values < node.val
        """


The root node supplied will be of type Node which has the following class definition:

    class Node:
        """Node that contains value, index and left and right pointers."""
        def __init__(self, val: int, left: Node = None, right: Node = None, index: int = 0):
            self.val = val
            self.left = left
            self.right = right
            self.index = index


Example 1
---------

### Input:
```
            ________________________________(value = 4, index = 0)___________
           /                                                                 \
(value = 1, index = 2)___________                                 (value = 6, index = 1)___________
                                 \                                                                 \
                      (value = 2, index = 4)                                            (value = 9, index = 3)
```

### **Output: \[2, 8, None, 48, 1\]**

### Explanation

*   `output[0]` = product of all nodes less than `4`: so 1 \* 2 = 2.

*   `output[1]` = product of all nodes less than 6: so 1 \* 2 \* 4 = 8.
*   `output[2]` = product of all nodes less than 1 which is **`None`** since 1 is the smallest value in the tree.

*   `output[3]` = product of all nodes less than 9: so 1 \* 2 \* 4 \* 6 = 48.
*   `output[4]` = product of all nodes less than 2: product of all nodes less than `2`: which is 1 since `2` is the _second_ smallest value in the tree and 1 is the smallest.


Example 2
---------

### Input:

                ________________________________(value = -1, index = 0)___________
               /                                                                 \
    (value = -3, index = 1)___________                                 (value = 0, index = 2)___________
                                     \                                                                 \
                          (value = -2, index = 3)                                            (value = 4, index = 4)


### **Output: \[6, None, -6, -3, 0\]**

### Explanation

*   `output[0]` = product of all nodes less than `(-1)`: so `(-3)` \* `(-2)` = `6`.

*   `output[1]` = product of all nodes less than `(-3)` which is **`None`** since `(-3)` is the smallest value in the tree.

*   `output[2]` = product of all nodes less than `0`: so `(-1)` \* `(-3)` \* `(-2)` = `(-6)`.

*   `output[3]` = product of all nodes less than `(-2)`: which is `-3` since `(-2)` is the _second_ smallest value in the tree and `(-3)` is the smallest.

*   `output[4]` = product of all nodes less than `4`: so `(0) *` `(-1) *` `(-3)` \* `(-2)` = `0`.


Guarantees
----------

*   The tree will not contain any 2 nodes with the same value or index.
*   `node.index` < **n**

----------------------------------------------------------------------------------------------------------------------

Submission
----------

Be sure to upload the following deliverables in a .zip folder to Mimir by the deadline.

Your .zip folder can contain other files (for example, `description.md`), but must include (at least) the following:

    CC4.zip
        |— CC4/
            |— README.xml       (for coding challenge feedback)
            |— __init__.py      (for proper Mimir testcase loading)
            |— solution.py      (contains your solution source code)
            
#### Grading

The following 100-point rubric will be used to determine your grade on CC4:

*   Tests (70)
    *   Coding Standard: \_\_/5
    *   Test empty\_and\_single: \_\_/5
    *   Test basic: \_\_/5
    *   Test medium: \_\_/10
    *   Test all negative: \_\_/10
    *   Test comprehensive\_small: \_\_/15
    *   Test comprehensive\_large: \_\_/15
    *   Test README.xml Validity: \_\_/5
*   Manual (30)
    *   M1 - Time and Space Complexity `O(n)`: \_\_/30
    *   If you do not meet `O(n)` time and space complexity, you are not eligible to earn any of the 30 manual points. In other words, these manual complexity points are "all-or-nothing." If you do not pass ALL of the automated test cases, you are not eligible to earn any of the 30 manual points.
    * If your program is missing the doctring -3 points is deducted from your coding standard.
Tips, Tricks & Notes
--------------------


* Think of how you can get the nodes in the BST _in_ the _order_ that makes the calculations very easy. Read the bolded words again.
* It may be helpful to think of decomposing this CC into 2 parts: (i) computing smaller products, and (ii) reordering the smaller products according to index in the return list.
* Check out D2L and Zybooks for more information on BST. You can use any algorithms from Onsay's lecture notes and also on Zybooks.  
* You are only required to have O(n) space complexity but this problem is doable in O(1) extra space (Excluding the recursion stack and the output list). You are encouraged to think about the lower space solution if you finish early!
* Remember that _all challenges are opportunities_, in this course and beyond. The journey to your solution is the true reward, so make the most of it. Enjoy!

Created by Caroline Gormely, Elizabeth DeBack, Joseph Pallipadan