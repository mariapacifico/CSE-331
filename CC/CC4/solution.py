from __future__ import annotations
from typing import List


class Node:
    """Node that contains value, index and left and right pointers."""
    def __init__(self, val: int, left: Node = None, right: Node = None, index: int = 0):
        self.val = val
        self.index = index
        self.left = left
        self.right = right

def inorder(root: Node, final: List[(int, int)]) -> None:
    """
    Takes a BST and puts in a list with the order of the values
    :param root: A Node within the BST that needs to be in order
    :param final: A list which is being edited throughout the function
    :return: A list that contains the order of the values. Also includes the index with each value.
    """
    if root is None:
        return final

    inorder(root.left, final)
    # reached the most left value in BST, add to list
    final.append((root.val, root.index))
    inorder(root.right, final)

def smaller_product(root: Node) -> List[int]:
    """
    Calculates the smallest product values for each node in a BST
    :param root: The root of the BST that contains the values and index values
    :return: A list containing the smallest product values based on each node's index
    """
    # Put in order
    root_inorder = []
    inorder(root, root_inorder)

    # Create new list to add final values
    final_list = [None] * len(root_inorder)

    # Keep track of multiplied
    multiply = 1

    # iterate through inorder list to get values and index
    for i in range(len(root_inorder)):
        value = root_inorder[i][0]
        index = root_inorder[i][1]

        # first value, set to None
        if i == 0:
            final_list[index] = None

        # otherwise, set to multiply
        else:
            final_list[index] = multiply

        # update multiply
        multiply *= value

    return final_list

