"""
Project 1
CSE 331 S21 (Onsay)
Maria Pacifico
DLL.py
"""

from typing import TypeVar, List

# for more information on typehinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")            # represents generic type
Node = TypeVar("Node")      # represents a Node object (forward-declare to use in Node __init__)

# pro tip: PyCharm auto-renders docstrings (the multiline strings under each function definition)
# in its "Documentation" view when written in the format we use here. Open the "Documentation"
# view to quickly see what a function does by placing your cursor on it and using CTRL + Q.
# https://www.jetbrains.com/help/pycharm/documentation-tool-window.html


class Node:
    """
    Implementation of a doubly linked list node.
    Do not modify.
    """
    __slots__ = ["value", "next", "prev"]

    def __init__(self, value: T, next: Node = None, prev: Node = None) -> None:
        """
        Construct a doubly linked list node.

        :param value: value held by the Node.
        :param next: reference to the next Node in the linked list.
        :param prev: reference to the previous Node in the linked list.
        :return: None.
        """
        self.next = next
        self.prev = prev
        self.value = value

    def __repr__(self) -> str:
        """
        Represents the Node as a string.

        :return: string representation of the Node.
        """
        return f"Node({str(self.value)})"

    __str__ = __repr__


class DLL:
    """
    Implementation of a doubly linked list without padding nodes.
    Modify only below indicated line.
    """
    __slots__ = ["head", "tail", "size"]

    def __init__(self) -> None:
        """
        Construct an empty doubly linked list.

        :return: None.
        """
        self.head = self.tail = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the DLL as a string.

        :return: string representation of the DLL.
        """
        result = []
        node = self.head
        while node is not None:
            result.append(str(node))
            node = node.next
        return " <-> ".join(result)

    def __str__(self) -> str:
        """
        Represent the DLL as a string.

        :return: string representation of the DLL.
        """
        return repr(self)

    # MODIFY BELOW #

    def empty(self) -> bool:
        """
        Return boolean indicating whether DLL is empty.

        Required time & space complexity (respectively): O(1) & O(1).

        :return: True if DLL is empty, else False.
        """
        if self.tail is None and self.head is None: #Nothing in list
            return True

    def push(self, val: T, back: bool = True) -> None:
        """
        Create Node containing `val` and add to back (or front) of DLL. Increment size by one.

        Required time & space complexity (respectively): O(1) & O(1).

        Note: You might find it easier to implement this as a push_back and
            push_front function first.

        :param val: value to be added to the DLL.
        :param back: if True, add Node containing value to back (tail-end) of DLL;
            if False, add to front (head-end).
        :return: None.
        """
        new_node = Node(val) # allocate node
        self.size += 1 #increment size

        # when the list is empty
        if self.empty():
            self.head = new_node
            self.tail = new_node
            return

        # add value to the back
        if back:
            # change tail to point to value
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node

        # add value to the front
        else:
            # change self head to point to value
            self.head.prev = new_node
            new_node.next = self.head
            self.head = new_node

    def pop(self, back: bool = True) -> None:
        """
        Remove Node from back (or front) of DLL. Decrement size by 1. If DLL is empty, do nothing.

        Required time & space complexity (respectively): O(1) & O(1).

        :param back: if True, remove Node from (tail-end) of DLL;
            if False, remove from front (head-end).
        :return: None.
        """
        # empty DLL
        if self.empty():
            return

        # one element in list
        if self.size == 1:
            self.head = None
            self.tail = None
            self.size = 0
            return

        # remove from back
        if back:
            prev_node = self.tail.prev
            prev_node.next = None #set next to nothing
            self.tail = prev_node

        # remove from front
        else:
            next_head = self.head.next
            next_head.prev = None
            self.head = next_head

        # remove 1 from size
        self.size -= 1


    def from_list(self, source: List[T]) -> None:
        """
        Construct DLL from a standard Python list.

        Required time & space complexity (respectively): O(n) & O(n).

        :param source: standard Python list from which to construct DLL.
        :return: None.
        """
        # iterate through list, push to self
        for index in source:
            self.push(index)

    def to_list(self) -> List[T]:
        """
        Construct standard Python list from DLL.

        Required time & space complexity (respectively): O(n) & O(n).

        :return: standard Python list containing values stored in DLL.
        """
        final_list = []

        first = self.head
        size = 0

        # iterate through DLL, add all values to list
        while size != self.size:
            final_list.append(first.value)
            first = first.next
            size += 1

        return final_list

    def find(self, val: T) -> Node:
        """
        Find first instance of `val` in the DLL and return associated Node object.

        Required time & space complexity (respectively): O(n) & O(1).

        :param val: value to be found in DLL.
        :return: first Node object in DLL containing `val`.
            If `val` does not exist in DLL, return None.
        """

        # iterate through DLL, check value
        first = self.head
        size = 0

        while size != self.size:

            # return node when matches value
            if first.value == val:
                return first

            first = first.next
            size += 1


    def find_all(self, val: T) -> List[Node]:
        """
        Find all instances of `val` in DLL and return Node objects in standard Python list.

        Required time & space complexity (respectively): O(n) & O(n).

        :param val: value to be searched for in DLL.
        :return: Python list of all Node objects in DLL containing `val`.
            If `val` does not exist in DLL, return empty list.
        """
        final_list = []

        # iterate through DLL, check value
        first = self.head
        size = 0

        while size != self.size:

            # add to list when matches value
            if first.value == val:
                final_list.append(first)

            first = first.next
            size += 1

        return final_list

    def _remove_node(self, to_remove: Node) -> None:
        """
        Given a node in the linked list, remove it.
        Should only be called from within the DLL class.

        Required time & space complexity (respectively): O(1) & O(1).

        :param to_remove: node to be removed from the list
        :return: None
        """
        prev_to_remove = to_remove.prev
        after_to_remove = to_remove.next

        # node is the head
        if prev_to_remove is None:
            self.pop(False)
            return

        # node is the tail
        if after_to_remove is None:
            self.pop()
            return

        # delete node in the middle
        prev_to_remove.next = after_to_remove
        after_to_remove.prev = prev_to_remove
        self.size -= 1

    def delete(self, val: T) -> bool:
        """
        Delete first instance of `val` in the DLL. Must call _remove_node.

        Required time & space complexity (respectively): O(n) & O(1).

        :param val: value to be deleted from DLL.
        :return: True if Node containing `val` was deleted from DLL; else, False.
        """
        node_to_delete = self.find(val)

        # not in list
        if node_to_delete is None:
            return False

        self._remove_node(node_to_delete)
        return True

    def delete_all(self, val: T) -> int:
        """
        Delete all instances of `val` in the DLL. Must call _remove_node.

        Required time & space complexity (respectively): O(n) & O(1).

        :param val: value to be deleted from DLL.
        :return: integer indicating the number of Nodes containing `val` deleted from DLL;
                 if no Node containing `val` exists in DLL, return 0.
        """
        # find all nodes
        nodes_to_delete = self.find_all(val)

        # remove all nodes within list
        for i in nodes_to_delete:
            self._remove_node(i)

        # number of nodes deleted within list
        return len(nodes_to_delete)

    def reverse(self) -> None:
        """
        Reverse DLL in-place by modifying all `next` and `prev` references of Nodes in the
        DLL and resetting the `head` and `tail` references.
        Must be implemented in-place for full credit. May not create new Node objects.

        Required time & space complexity (respectively): O(n) & O(1).

        :return: None.
        """
        #switch head and tails
        head = self.head
        tail = self.tail

        self.head = tail
        self.tail = head

        start = self.head

        size = 0
        stop = self.size

        # iterate through self backwards
        while size != stop:

            # store prev
            start_prev = start.prev

            #switch
            start.prev = start.next
            start.next = start_prev

            start = start.next
            size += 1


def flurricane(dll: DLL, delta: float) -> DLL:
    """
    Applies a moving average filter of width `delta` to the time-series data in `dll`.

    Required time & space complexity (respectively): O(n) & O(N).

    :param dll: A `DLL` where each `Node` holds a `value` of `Tuple(float, float)` representing
                the pair `(t, x)`, where `t` represents the time of some measurement `x`.
    :param delta: A `float` representing the width of the moving average filter to apply.
    :return: A `DLL` holding `Tuple(float, float)` representing `(t, filtered_x)`, where `t` is
             exactly the `t` in the input list, and `filtered_x` is the avg of all measurements
    `x` recorded from `t-delta` to `t` (including endpoints).
    """
    # empty DLL, return empty DLL
    if dll.size == 0:
        return DLL()

    # final DLL
    final_dll = DLL()

    # start at the end of DLL
    tail = dll.tail

    # iterate through DLL
    while tail is not None:

        # get t and delta t
        t_value = tail.value[0]
        delta_t = t_value - delta

        # to calculate average
        divisor = 1
        sum = tail.value[1]

        # track node
        current = tail.prev

        # while loop to calculate the values required for the average that are
        # within range
        # make sure there is a node to the left and within [delta_t, t]
        while current is not None and current.value[0] >= delta_t:
            # update sum and divisor
            sum += current.value[1]
            divisor += 1
            #move to the next node to the left
            current = current.prev

        # final avg
        avg = sum / divisor
        # add to DLL
        final_dll.push((t_value, avg), False)

        tail = tail.prev


    return final_dll
