"""
Project 1
CSE 331 S21 (Onsay)
Maria Pacifico
tests.py
"""

from DLL import DLL, Node, flurricane
from typing import TypeVar, List
from random import seed
import copy
import unittest
import cProfile
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.cm import tab10
# from xml.dom import min
# for more information on typehinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")            # represents generic type


class DLLTests(unittest.TestCase):

    def check_dll(self, expected: List[T], dll: DLL):
        """
        Assert structure of dll is proper and contains the values of result.
        Used as helper function throughout testcases. Not an actual testcase itself.
        Collapse/hide this by clicking the minus arrow on the left sidebar.

        :param expected: list of expected values in dll
        :param dll: DLL to be validated
        :return: None. Raises exception and fails testcase if structure is DLL is not properly structured
                 or contains values different from those in result.
        """
        # check size
        self.assertEqual(len(expected), dll.size)

        # short-circuit if empty list
        if dll.size == 0:
            self.assertIsNone(dll.head)
            self.assertIsNone(dll.tail)
            return

        # check head and tail
        self.assertIsNone(dll.head.prev)
        self.assertIsNone(dll.tail.next)
        if isinstance(expected[0], tuple):
            for j, element in enumerate(expected[0]):
                self.assertAlmostEqual(element, dll.head.value[j], places=2)
            for j, element in enumerate(expected[-1]):
                self.assertAlmostEqual(element, dll.tail.value[j], places=2)
        else:
            self.assertAlmostEqual(expected[0], dll.head.value, places=2)
            self.assertAlmostEqual(expected[-1], dll.tail.value, places=2)

        # check all intermediate connections and values
        left, right = dll.head, dll.head.next
        i = 0
        while right is not None:
            self.assertIs(left.next, right)
            self.assertIs(left, right.prev)
            if isinstance(expected[i], tuple):
                for j, element in enumerate(expected[i]):
                    self.assertAlmostEqual(element, left.value[j], places=2)
                for j, element in enumerate(expected[i + 1]):
                    self.assertAlmostEqual(element, right.value[j], places=2)
            else:
                self.assertAlmostEqual(expected[i], left.value, places=2)
                self.assertAlmostEqual(expected[i + 1], right.value, places=2)
            left, right = left.next, right.next
            i += 1

    def test_empty(self):

        # (1) empty DLL
        dll = DLL()
        self.assertTrue(dll.empty())

        # (2) DLL with one node
        dll.head = dll.tail = Node(1)
        dll.size += 1
        self.assertFalse(dll.empty())

        # (3) DLL with multiple nodes
        for i in range(0, 50):
            dll.tail.next = Node(i, None, dll.tail)
            dll.tail = dll.tail.next
            dll.size += 1
            self.assertFalse(dll.empty())

        # (4) DLL after removing all nodes
        dll.head = dll.tail = None
        dll.size = 0
        self.assertTrue(dll.empty())

    def test_push(self):

        # (1) push single node on back
        dll = DLL()
        dll.push(0)
        self.assertIs(dll.head, dll.tail)   # see note 8 in specs for `is` vs `==`
        self.check_dll([0], dll)            # if failure here, see (1).
        # pro tip: use CTRL + B with your cursor on check_dll to jump to its definition at the top of the file.
        # then, use CTRL + Alt + RightArrow to jump back here!
        # https://www.jetbrains.com/help/pycharm/navigating-through-the-source-code.html

        # (2) push single node on front
        dll = DLL()
        dll.push(0, back=False)
        self.assertIs(dll.head, dll.tail)
        self.check_dll([0], dll)            # if failure here, see (2)

        # (3) push multiple nodes on back
        dll = DLL()
        lst = []
        for i in range(5):
            dll.push(i)
            lst.append(i)
            self.check_dll(lst, dll)        # if failure here, see (3)

        # (4) push multiple nodes on front
        dll = DLL()
        lst = []
        for i in range(5):
            dll.push(i, back=False)
            lst.insert(0, i)
            self.check_dll(lst, dll)        # if failure here, see (4)

        # (5) alternate pushing onto front and back
        dll = DLL()
        lst = []
        for i in range(50):
            dll.push(i, i % 2 == 0)         # push back if i is even, else push front
            if i % 2 == 0:                  # pushed back, new tail
                lst.append(i)
                self.check_dll(lst, dll)    # if failure here, see (5)
            else:                           # pushed front, new head
                lst.insert(0, i)
                self.check_dll(lst, dll)    # if failure here, see (5)

    def test_pop(self):

        # (1) pop back on empty list (should do nothing)
        dll = DLL()
        try:
            dll.pop()
        except Exception as e:
            self.fail(msg=f"Raised {type(e)} when popping from back of empty list.")

        # (2) pop front on empty list (should do nothing)
        dll = DLL()
        try:
            dll.pop(back=False)
        except Exception as e:
            self.fail(msg=f"Raised {type(e)} when popping from front of empty list.")

        # (3) pop back on multiple node list
        dll = DLL()
        lst = []
        for i in range(5):          # construct list
            dll.push(i)
            lst.append(i)
        for i in range(5):          # destruct list
            dll.pop()
            lst.pop()
            self.check_dll(lst, dll)     # if failure here, see (3)

        # (4) pop front on multiple node list
        dll = DLL()
        lst = []
        for i in range(5):          # construct list
            dll.push(i)
            lst.append(i)
        for i in range(5):          # destruct list
            dll.pop(back=False)
            lst.pop(0)
            self.check_dll(lst, dll)     # if failure here, see (4)

        # (5) alternate popping from front, back
        dll = DLL()
        lst = []
        for i in range(50):
            dll.push(i)
            lst.append(i)
        for end in range(49):           # remove all but one node
            dll.pop(end % 2 == 0)       # pop back if even, front if odd
            if end % 2 == 0:            # removed tail
                lst.pop()
                self.check_dll(lst, dll)     # if failure here, see (5)
            else:                       # removed head
                lst.pop(0)
                self.check_dll(lst, dll)     # if failure here, see (5)

        # (6) check there is exactly one node left in DLL (middle of original), then remove
        self.check_dll([24], dll)        # if failure here, see (6)
        dll_copy = copy.deepcopy(dll)
        dll.pop()                   # remove tail
        dll_copy.pop(back=False)    # remove head
        self.check_dll([], dll)          # if failure here, see (6)
        self.check_dll([], dll_copy)     # if failure here, see (6)

    def test_from_list(self):

        # (1) create DLL from empty list
        dll = DLL()
        dll.from_list([])
        self.check_dll([], dll)              # if failure here, see (1)

        # (2) create DLL from longer lists
        for i in range(50):
            source = list(range(i))
            dll = DLL()
            dll.from_list(source)
            self.check_dll(source, dll)      # if failure here, see (2)

    def test_to_list(self):

        # (1) create list from empty DLL
        dll = DLL()
        output = dll.to_list()
        self.check_dll(output, dll)          # if failure here, see (1)

        # (2) create list from longer DLLs
        for i in range(50):
            dll = DLL()
            for j in range(i):
                dll.push(j)
            output = dll.to_list()
            self.check_dll(output, dll)      # if failure here, see (2)

    def test_find(self):

        # (1) find in empty DLL
        dll = DLL()
        node = dll.find(331)
        self.assertIsNone(node)

        # (2) find existing value in single-node DLL
        dll = DLL()
        dll.push(0)
        node = dll.find(0)
        self.assertIsInstance(node, Node)
        self.assertEqual(0, node.value)
        self.assertIsNone(node.next)
        self.assertIsNone(node.prev)

        # (3) find non-existing value in single-node DLL
        node = dll.find(331)
        self.assertIsNone(node)

        # (4) find in longer DLL with all unique values
        dll = DLL()
        for i in range(10):
            dll.push(i)

        node = dll.find(0)
        self.assertIsInstance(node, Node)
        self.assertIs(dll.head, node)
        self.assertIsNone(node.prev)
        self.assertEqual(0, node.value)
        self.assertEqual(1, node.next.value)

        node = dll.find(9)
        self.assertIsInstance(node, Node)
        self.assertIs(dll.tail, node)
        self.assertIsNone(node.next)
        self.assertEqual(9, node.value)
        self.assertEqual(8, node.prev.value)

        node = dll.find(4)
        self.assertIsInstance(node, Node)
        self.assertEqual(4, node.value)
        self.assertEqual(3, node.prev.value)
        self.assertEqual(5, node.next.value)

        node = dll.find(331)
        self.assertIsNone(node)

        # (5) find first instance in longer DLL with duplicated values
        for i in range(9, 0, -1):     # DLL will be 0, 1, ..., 9, 9, 8, ..., 0
            dll.push(i)

        node = dll.find(0)      # should find head 0, not tail 0
        self.assertIsInstance(node, Node)
        self.assertIs(dll.head, node)
        self.assertIsNone(node.prev)
        self.assertEqual(0, node.value)
        self.assertEqual(1, node.next.value)

        node = dll.find(9)      # should find first 9
        self.assertIsInstance(node, Node)
        self.assertEqual(9, node.value)
        self.assertEqual(8, node.prev.value)
        self.assertEqual(9, node.next.value)

        node = dll.find(4)      # should find first 4
        self.assertIsInstance(node, Node)
        self.assertEqual(4, node.value)
        self.assertEqual(3, node.prev.value)
        self.assertEqual(5, node.next.value)

        node = dll.find(331)
        self.assertIsNone(node)

    def test_find_all(self):
        # (1) find_all in empty DLL
        dll = DLL()
        nodes = dll.find_all(331)
        self.assertEqual([], nodes)

        # (2) find_all existing value in single-node DLL
        dll = DLL()
        dll.push(0)
        nodes = dll.find_all(0)
        self.assertIsInstance(nodes, List)
        self.assertEqual(1, len(nodes))
        self.assertEqual(0, nodes[0].value)
        self.assertIsNone(nodes[0].next)
        self.assertIsNone(nodes[0].prev)

        # (3) find non-existing value in single-node DLL
        nodes = dll.find_all(331)
        self.assertEqual([], nodes)

        # (4) find_all in longer DLL with all unique values
        dll = DLL()
        for i in range(10):
            dll.push(i)

        nodes = dll.find_all(0)
        self.assertIsInstance(nodes, List)
        self.assertEqual(1, len(nodes))
        self.assertIs(dll.head, nodes[0])
        self.assertIsNone(nodes[0].prev)
        self.assertEqual(0, nodes[0].value)
        self.assertEqual(1, nodes[0].next.value)

        nodes = dll.find_all(9)
        self.assertIsInstance(nodes, List)
        self.assertEqual(1, len(nodes))
        self.assertIs(dll.tail, nodes[0])
        self.assertIsNone(nodes[0].next)
        self.assertEqual(9, nodes[0].value)
        self.assertEqual(8, nodes[0].prev.value)

        nodes = dll.find_all(4)
        self.assertIsInstance(nodes, List)
        self.assertEqual(1, len(nodes))
        self.assertEqual(4, nodes[0].value)
        self.assertEqual(3, nodes[0].prev.value)
        self.assertEqual(5, nodes[0].next.value)

        nodes = dll.find_all(331)
        self.assertEqual([], nodes)

        # (5) find all instances in longer DLL with duplicated values
        for i in range(9, -1, -1):     # DLL will be 0, 1, ..., 9, 9, 8, ..., 0
            dll.push(i)

        nodes = dll.find_all(0)
        self.assertIsInstance(nodes, List)
        self.assertEqual(2, len(nodes))
        self.assertIs(dll.head, nodes[0])
        self.assertIsNone(nodes[0].prev)
        self.assertEqual(0, nodes[0].value)
        self.assertEqual(1, nodes[0].next.value)
        self.assertIs(dll.tail, nodes[1])
        self.assertIsNone(nodes[1].next)
        self.assertEqual(0, nodes[1].value)
        self.assertEqual(1, nodes[1].prev.value)

        nodes = dll.find_all(9)
        self.assertIsInstance(nodes, List)
        self.assertEqual(2, len(nodes))
        self.assertEqual(9, nodes[0].value)
        self.assertEqual(8, nodes[0].prev.value)
        self.assertEqual(9, nodes[0].next.value)
        self.assertEqual(9, nodes[1].value)
        self.assertEqual(9, nodes[1].prev.value)
        self.assertEqual(8, nodes[1].next.value)

        nodes = dll.find_all(4)
        self.assertIsInstance(nodes, List)
        self.assertEqual(2, len(nodes))
        self.assertEqual(4, nodes[0].value)
        self.assertEqual(3, nodes[0].prev.value)
        self.assertEqual(5, nodes[0].next.value)
        self.assertEqual(4, nodes[1].value)
        self.assertEqual(5, nodes[1].prev.value)
        self.assertEqual(3, nodes[1].next.value)

        nodes = dll.find_all(331)
        self.assertEqual([], nodes)

    def test_delete(self):

        # (1) delete from empty DLL
        dll = DLL()
        result = dll.delete(331)
        self.assertFalse(result)

        # (2) delete existing value in single-node DLL
        dll = DLL()
        dll.push(0)
        result = dll.delete(0)
        self.assertTrue(result)
        self.check_dll([], dll)              # if failure here, see (2)

        # (3) delete non-existing value in single-node DLL
        dll = DLL()
        dll.push(0)
        result = dll.delete(331)
        self.assertFalse(result)
        self.check_dll([0], dll)             # if failure here, see (3)

        # (4) delete from longer DLL with all unique values
        dll = DLL()
        lst = []
        for i in range(10):
            dll.push(i)
            lst.append(i)

        to_delete = [1, 4, 7, 5, 6, 3, 2, 9, 0, 8]
        for i in range(10):
            result = dll.delete(to_delete[i])
            self.assertTrue(result)
            result = dll.delete(331)
            self.assertFalse(result)

            lst.remove(to_delete[i])
            #self.check_dll(lst, dll)     # if failure here, see (4)

        # (5) delete first instance in longer DLL with duplicated values
        dll = DLL()
        lst = []
        for i in range(10):
            dll.push(i)
            lst.append(i)
        for i in range(9, -1, -1):      # DLL will be 0, 1, ..., 9, 9, 8, ..., 0
            dll.push(i)
            lst.append(i)

        to_delete = [1, 4, 7, 5, 6, 3, 2, 9, 0, 8]
        for i in range(10):
            result = dll.delete(to_delete[i])
            self.assertTrue(result)
            result = dll.delete(331)
            self.assertFalse(result)

            lst.remove(to_delete[i])
            self.check_dll(lst, dll)     # if failure here, see (5)

        # (6) sanity check after deletions
        lst = list(range(9, -1, -1))
        self.check_dll(lst, dll)         # if failure here, see (6)

    def test_delete_all(self):

        # (1) delete all from empty DLL
        dll = DLL()
        count = dll.delete_all(331)
        self.assertEqual(0, count)

        # (2) delete existing value in single-node DLL
        dll = DLL()
        dll.push(0)
        count = dll.delete_all(0)
        self.assertEqual(1, count)
        self.check_dll([], dll)              # if failure here, see (2)

        # (3) delete non-existing value in single-node DLL
        dll = DLL()
        dll.push(0)
        count = dll.delete_all(331)
        self.assertEqual(0, count)
        self.check_dll([0], dll)             # if failure here, see (3)

        # (4) delete from longer DLL with all unique values
        dll = DLL()
        lst = []
        for i in range(10):
            dll.push(i)
            lst.append(i)

        to_delete = [1, 4, 7, 5, 6, 3, 2, 9, 0, 8]
        for i in range(10):
            count = dll.delete_all(to_delete[i])
            self.assertEqual(1, count)
            count = dll.delete_all(331)
            self.assertEqual(0, count)

            lst.remove(to_delete[i])
            self.check_dll(lst, dll)         # if failure here, see (4)

        # (5) delete all in longer DLL with duplicated values
        dll = DLL()
        lst = []
        for i in range(10):
            dll.push(i)
            lst.append(i)
        for i in range(9, -1, -1):      # DLL will be 0, 1, ..., 9, 9, 8, ..., 0
            dll.push(i)
            lst.append(i)

        to_delete = [1, 4, 7, 5, 6, 3, 2, 9, 0, 8]
        for i in range(10):
            count = dll.delete_all(to_delete[i])
            self.assertEqual(2, count)
            count = dll.delete_all(331)
            self.assertEqual(0, count)

            lst.remove(to_delete[i])
            lst.remove(to_delete[i])    # delete both instances
            self.check_dll(lst, dll)         # if failure here, see (5)

        # (6) sanity check empty list after all deletions
        self.check_dll([], dll)              # if failure here, see (6)

    def test_reverse(self):

        # (1) reverse empty DLL
        dll = DLL()
        dll.reverse()
        self.check_dll([], dll)      # if failure here, see (1)

        # (2) reverse single-node DLL
        dll = DLL()
        dll.push(0)
        dll.reverse()
        self.check_dll([0], dll)      # if failure here, see (2)

        # (3) reverse longer DLL
        dll = DLL()
        lst = []
        for i in range(10):
            dll.push(i)
            lst.append(i)
        old_head, old_tail = dll.head, dll.tail
        dll.reverse()
        new_head, new_tail = dll.head, dll.tail
        lst.reverse()

        self.check_dll(lst, dll)
        self.assertIs(new_head, old_tail)
        self.assertIs(new_tail, old_head)

        # (4) reverse palindrome DLL
        dll = DLL()
        lst = []
        for i in range(10):
            dll.push(i)
            lst.append(i)
        for i in range(9, -1, -1):
            dll.push(i)
            lst.append(i)
        old_head, old_tail = dll.head, dll.tail
        dll.reverse()
        new_head, new_tail = dll.head, dll.tail
        lst.reverse()

        self.check_dll(lst, dll)
        self.assertIs(new_head, old_tail)
        self.assertIs(new_tail, old_head)

    def test_DLL_comprehensive(self):

        # test all functions: empty, push, pop, from_list, to_list, find, find_all, delete, delete_all, reverse

        # (1) test functions on empty DLL
        dll = DLL()
        self.assertTrue(dll.empty())

        dll.pop()       # should not raise exception

        lst = dll.to_list()
        self.assertEqual([], lst)

        node = dll.find(331)
        self.assertIsNone(node)
        nodes = dll.find_all(331)
        self.assertEqual([], nodes)

        result = dll.delete(331)
        self.assertFalse(result)
        count = dll.delete_all(331)
        self.assertEqual(0, count)

        dll.reverse()   # should not raise exception

        # (2) test functions on single-node DLL
        dll = DLL()
        dll.push(331)
        self.assertFalse(dll.empty())
        self.assertIs(dll.head, dll.tail)
        self.assertEqual(331, dll.head.value)
        self.assertEqual(331, dll.tail.value)

        dll.pop()
        self.assertTrue(dll.empty())
        self.assertIsNone(dll.head)
        self.assertIsNone(dll.tail)

        dll.from_list([331])
        lst = dll.to_list()
        self.assertEqual([331], lst)

        node = dll.find(331)
        self.assertIsInstance(node, Node)
        self.assertEqual(331, node.value)
        self.assertIsNone(node.prev)
        self.assertIsNone(node.next)

        nodes = dll.find_all(331)
        self.assertIsInstance(nodes, List)
        self.assertEqual(1, len(nodes))
        self.assertEqual(331, nodes[0].value)
        self.assertIsNone(nodes[0].next)
        self.assertIsNone(nodes[0].prev)

        result = dll.delete(331)
        self.assertTrue(result)
        self.assertTrue(dll.empty())

        dll.push(331)
        count = dll.delete_all(331)
        self.assertEqual(1, count)
        self.assertTrue(dll.empty())

        dll.push(331)
        dll.reverse()
        self.assertFalse(dll.empty())
        self.assertEqual(331, dll.head.value)
        self.assertEqual(331, dll.tail.value)
        self.assertIs(dll.head, dll.tail)

        # (3) test functions on large, randomly-populated DLL containing 0-999 and 10 duplicates of 0-9
        seed(331)
        source = list(range(1000)) + [i for i in range(10) for _ in range(10)]
        #shuffle(source)
        source_rev = source[::-1]

        # (3.1) empty, push, pop
        dll, dll_rev = DLL(), DLL()
        self.assertTrue(dll.empty())
        self.assertTrue(dll_rev.empty())

        for num in source:
            dll.push(num)
            dll_rev.push(num, back=False)

        self.assertFalse(dll.empty())
        self.assertFalse(dll_rev.empty())
        self.check_dll(source, dll)
        self.check_dll(source_rev, dll_rev)

        for i in range(10):
            dll.pop()
            dll_rev.pop(back=False)
        self.check_dll(source[:-10], dll)
        self.check_dll(source_rev[10:], dll_rev)

        # (3.2) from_list, to_list
        dll, dll_rev = DLL(), DLL()
        dll.from_list(source)
        dll_rev.from_list(source_rev)
        lst = dll.to_list()
        lst_rev = dll_rev.to_list()
        self.assertEqual(source, lst)
        self.assertEqual(source_rev, lst_rev)
        self.check_dll(source, dll)
        self.check_dll(source_rev, dll_rev)

        # (3.3) find, find_all, delete, delete_all
        for i in range(1000):
            # search existing values
            result, result_rev = dll.find(i), dll_rev.find(i)
            self.assertEqual(i, result.value)
            self.assertEqual(i, result_rev.value)
            result, result_rev = dll.find_all(i), dll.find_all(i)
            if i < 10:
                self.assertEqual(11, len(result))
                self.assertEqual(11, len(result_rev))
            else:
                self.assertEqual(1, len(result))
                self.assertEqual(1, len(result_rev))
        # search non-existing values
        result, result_rev = dll.find(1000), dll_rev.find(1000)
        self.assertIsNone(result)
        self.assertIsNone(result_rev)
        result, result_rev = dll.find_all(1000), dll.find_all(1000)
        self.assertEqual([], result)
        self.assertEqual([], result_rev)

        # delete existing values
        for i in range(1000):
            result, result_rev = dll.delete(i), dll_rev.delete(i)
            self.assertTrue(result)
            self.assertTrue(result_rev)
            result, result_rev = dll.find_all(i), dll.find_all(i)
            if i < 10:
                self.assertEqual(10, len(result))
                self.assertEqual(10, len(result_rev))
            else:
                self.assertEqual([], result)
                self.assertEqual([], result_rev)

        # delete non-existing values
        result, result_rev = dll.delete(1000), dll_rev.delete(1000)
        self.assertFalse(result)
        self.assertFalse(result_rev)

        # delete all remaining existing values
        for i in range(1000):
            count, count_rev = dll.delete_all(i), dll_rev.delete_all(i)
            if i < 10:
                self.assertEqual(10, count)
                self.assertEqual(10, count_rev)
            else:
                self.assertEqual(0, count)
                self.assertEqual(0, count_rev)
            result, result_rev = dll.find_all(i), dll.find_all(i)
            self.assertEqual([], result)
            self.assertEqual([], result_rev)
        self.assertTrue(dll.empty())

        # (3.4) reverse
        dll.from_list(source)
        dll_rev.from_list(source_rev)
        dll.reverse()
        dll_rev.reverse()
        self.check_dll(source_rev, dll)
        self.check_dll(source, dll_rev)

    def test_flurricane(self):

        # (1) test empty DLL returns an empty DLL
        dll = DLL()
        filtered = flurricane(dll, delta=0)
        self.assertIsInstance(filtered, DLL)
        self.assertTrue(filtered.empty())

        # # (2) test single-node DLL returns a single-node DLL with exact same entry
        dll = DLL()
        dll.push((0, 0))
        filtered = flurricane(dll, delta=0)
        self.assertIsInstance(filtered, DLL)
        self.assertEqual(filtered.head.value, (0, 0))
        self.assertEqual(filtered.head, filtered.tail)
        self.assertIsNone(filtered.head.next)

        # (3) test two-node DLL with averaging (large delta)
        dll = DLL()
        dll.from_list([(0, 0), (1, 1)])
        filtered = flurricane(dll, delta=2)
        self.assertIsInstance(filtered, DLL)
        self.assertEqual(filtered.head.value, (0, 0))
        self.assertEqual(filtered.tail.value, (1, 0.5))

        # (4) test two-node DLL without averaging (small delta)
        dll = DLL()
        dll.from_list([(0, 0), (1, 1)])
        filtered = flurricane(dll, delta=0)
        self.assertIsInstance(filtered, DLL)
        self.assertEqual(filtered.head.value, (0, 0))
        self.assertEqual(filtered.tail.value, (1, 1))

        # (3) test regularly-spaced four-node DLL with averaging (large delta)
        dll = DLL()
        lst = [(0, 0), (1, 1), (2, 2), (3, 3)]
        dll.from_list(lst)
        filtered = flurricane(dll, delta=1)
        self.assertIsInstance(filtered, DLL)
        expected = [(0, 0), (1, 0.5), (2, 1.5), (3, 2.5)]
        self.check_dll(expected, filtered)      # if failure here, see (3)

        # (4) test regularly-spaced four-node DLL without averaging (small delta)
        dll = DLL()
        lst = [(0, 0), (1, 1), (2, 2), (3, 3)]
        dll.from_list(lst)
        filtered = flurricane(dll, delta=0)
        self.assertIsInstance(filtered, DLL)
        expected = lst
        self.check_dll(expected, filtered)      # if failure here, see (4)

        # (5) test irregularly-spaced four-node DLL with averaging (large delta)
        dll = DLL()
        lst = [(0, 0), (1, 1), (5, 2), (6, 3)]
        dll.from_list(lst)
        filtered_1 = flurricane(dll, delta=1)
        filtered_2 = flurricane(dll, delta=2)
        self.assertIsInstance(filtered, DLL)
        expected = [(0, 0), (1, 0.5), (5, 2), (6, 2.5)]
        self.check_dll(expected, filtered_1)      # if failure here, see (5)
        self.check_dll(expected, filtered_2)      # if failure here, see (5)

        # (6) test irregularly-spaced four-node DLL without averaging (small delta)
        dll = DLL()
        lst = [(0, 0), (1, 1), (5, 2), (6, 3)]
        dll.from_list(lst)
        filtered = flurricane(dll, delta=0)
        self.assertIsInstance(filtered, DLL)
        expected = lst
        self.check_dll(expected, filtered)      # if failure here, see (6)


    def test_flurricane_comprehensive(self):

        def get_katrina_data():
            """
            Read NOAA NHC HURDAT2 data on Katrina from CSV.
            More information about HURDAT2: https://www.nhc.noaa.gov/data/#hurdat
            Raw HURDAT2 data: https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2020-052921.txt

            :return: DLL of time-series of interest (wind, pressure)
            """
            with open("katrina.csv", "r") as f:
                lines = f.readlines()
                lines = [line.split(",") for line in lines]
            header = lines[0]
            date_idx = header.index("date")
            time_idx = header.index("time")
            wind_idx = header.index("max_wind")
            pressure_idx = header.index("min_pressure")
            datetimes = [float(line[date_idx]) + float(line[time_idx]) / 2400 for line in lines[1:]]
            winds = [float(line[wind_idx]) for line in lines[1:]]
            pressures = [float(line[pressure_idx]) for line in lines[1:]]
            dll_winds = DLL()
            dll_winds.from_list(list(zip(datetimes, winds)))
            dll_pressures = DLL()
            dll_pressures.from_list(list(zip(datetimes, pressures)))
            return dll_winds, dll_pressures

        def plot(title, data, deltas, expecteds):
            """
            Visualize raw timeseries data and associated smoothed timeseries.

            :param title: String title to apply to plot.
            :param data: Raw time-series data in the form [(t_0, x_0), (t_1, x_1),... ]
            :param deltas: Filter widths associated with each filtered time series in `expecteds`.
            :param expecteds: List of filtered time series, each in the form [(t_0, x_filtered_0),... ]
            :return: None (shows plot)
            """
            fig, ax = plt.subplots()
            for i in range(len(deltas)):
                delta, expected = deltas[i], np.array(expecteds[i])
                ax.plot(expected[:, 0], expected[:, 1], label=f"delta={delta}", color=tab10(i))
            data = np.array(data)
            ax.plot(data[:, 0], data[:, 1], label=f"data", color="k", linestyle=":")
            ax.scatter(data[:, 0], data[:, 1], label=f"data", color="k")
            ax.set_title(title)
            ax.legend()
            plt.tight_layout()
            plt.show()

        # set this to true to visualize raw / filtered data
        show_plots = False

        # (1) test increasing and decreasing trend with regularly-spaced samples
        dll = DLL()
        lst = [(0, 0), (1, 5), (2, 10), (3, 5), (4, 10), (5, 5), (6, 0), (7, 0)]
        dll.from_list(lst)
        filtered_0 = flurricane(dll, delta=0)
        filtered_1 = flurricane(dll, delta=1)
        filtered_2 = flurricane(dll, delta=2)
        filtered_3 = flurricane(dll, delta=3)
        filtered_all = flurricane(dll, delta=float("inf"))
        expected_0 = lst
        expected_1 = [(0, 0), (1, 2.5), (2, 7.5), (3, 7.5), (4, 7.5), (5, 7.5), (6, 2.5), (7, 0)]
        expected_2 = [(0, 0), (1, 2.5), (2, 5), (3, 20/3), (4, 25/3), (5, 20/3), (6, 5), (7, 5/3)]
        expected_3 = [(0, 0), (1, 2.5), (2, 5), (3, 5), (4, 7.5), (5, 7.5), (6, 5), (7, 15/4)]
        expected_all = [(0, 0), (1, 5/2), (2, 15/3), (3, 20/4), (4, 30/5), (5, 35/6), (6, 35/7), (7, 35/8)]
        if show_plots:
            # preview before assertions
            plot("Comprehensive 1: Regularly-Spaced", lst, deltas=[0, 1, 2, 3, float("inf")],
                 expecteds=[expected_0, expected_1, expected_2, expected_3, expected_all])
        self.check_dll(expected_0, filtered_0)      # if failure here, see (1)
        self.check_dll(expected_1, filtered_1)      # if failure here, see (1)
        self.check_dll(expected_2, filtered_2)      # if failure here, see (1)
        self.check_dll(expected_3, filtered_3)      # if failure here, see (1)
        self.check_dll(expected_all, filtered_all)  # if failure here, see (1)

        # clean up for easier debugging
        del filtered_0, filtered_1, filtered_2, filtered_3, filtered_all
        del expected_0, expected_1, expected_2, expected_3, expected_all

        # (2) test increasing and decreasing trend with irregularly-spaced samples
        dll = DLL()
        lst = [(0, 0), (1, 5), (10, 10), (11, 5), (20, 10), (21, 5), (30, 0), (31, 0)]
        dll.from_list(lst)
        filtered_1 = flurricane(dll, delta=1)
        filtered_5 = flurricane(dll, delta=5)
        filtered_11 = flurricane(dll, delta=11)
        filtered_15 = flurricane(dll, delta=15)
        filtered_all = flurricane(dll, delta=float("inf"))
        expected_1 = expected_5 = [(0, 0), (1, 2.5), (10, 10), (11, 7.5), (20, 10), (21, 7.5), (30, 0), (31, 0)]
        expected_11 = expected_15 = [(0, 0), (1, 2.5), (10, 5), (11, 5), (20, 25/3), (21, 7.5), (30, 5), (31, 15/4)]
        expected_all = [(0, 0), (1, 5/2), (10, 15/3), (11, 20/4), (20, 30/5), (21, 35/6), (30, 35/7), (31, 35/8)]
        if show_plots:
            # preview before assertions
            plot("Comprehensive 2: Irregularly-Spaced", lst, deltas=[1, 5, 11, 15, float("inf")],
                 expecteds=[expected_1, expected_5, expected_11, expected_15, expected_all])
        self.check_dll(expected_1, filtered_1)      # if failure here, see (2)
        self.check_dll(expected_5, filtered_5)      # if failure here, see (2)
        self.check_dll(expected_11, filtered_11)    # if failure here, see (2)
        self.check_dll(expected_15, filtered_15)    # if failure here, see (2)
        self.check_dll(expected_all, filtered_all)  # if failure here, see (2)

        # clean up for easier debugging
        del filtered_1, filtered_5, filtered_11, filtered_15, filtered_all
        del expected_1, expected_5, expected_11, expected_15, expected_all

        # read in real-world hurricane data
        dll_winds, dll_pressures = get_katrina_data()

        # (3) test on real-world wind data
        # note that one day corresponds to one unit of time
        filtered_12 = flurricane(dll_winds, delta=0.5)    # 12-hour filtering
        filtered_24 = flurricane(dll_winds, delta=1)      # 1-day filtering
        filtered_all = flurricane(dll_winds, delta=float("inf"))      # cumulative moving average
        expected_12 = [(23.75, 30.0), (24.0, 30.0), (24.25, 30.0), (24.5, 31.666666666666668), (24.75, 35.0), (25.0, 40.0), (25.25, 45.0), (25.5, 50.0), (25.75, 55.0), (25.929166666666667, 61.666666666666664), (26.0, 63.75), (26.25, 66.25), (26.5, 70.0), (26.75, 75.0), (27.0, 83.33333333333333), (27.25, 90.0), (27.5, 95.0), (27.75, 98.33333333333333), (28.0, 100.0), (28.25, 108.33333333333333), (28.5, 123.33333333333333), (28.75, 140.0), (29.0, 145.0), (29.25, 138.33333333333334), (29.4625, 125.0), (29.5, 121.25), (29.602083333333333, 112.5), (29.75, 106.0), (30.0, 86.25), (30.25, 56.666666666666664), (30.5, 40.0), (30.75, 33.333333333333336), (31.0, 30.0), (31.25, 28.333333333333332)]
        expected_24 = [(23.75, 30.0), (24.0, 30.0), (24.25, 30.0), (24.5, 31.25), (24.75, 33.0), (25.0, 36.0), (25.25, 40.0), (25.5, 45.0), (25.75, 50.0), (25.929166666666667, 56.0), (26.0, 58.333333333333336), (26.25, 61.666666666666664), (26.5, 65.83333333333333), (26.75, 70.83333333333333), (27.0, 77.0), (27.25, 82.0), (27.5, 89.0), (27.75, 94.0), (28.0, 97.0), (28.25, 104.0), (28.5, 114.0), (28.75, 124.0), (29.0, 132.0), (29.25, 137.0), (29.4625, 134.0), (29.5, 130.0), (29.602083333333333, 123.33333333333333), (29.75, 117.14285714285714), (30.0, 102.85714285714286), (30.25, 88.57142857142857), (30.5, 69.16666666666667), (30.75, 46.0), (31.0, 36.0), (31.25, 31.0)]
        expected_all = [(23.75, 30.0), (24.0, 30.0), (24.25, 30.0), (24.5, 31.25), (24.75, 33.0), (25.0, 35.0), (25.25, 37.142857142857146), (25.5, 39.375), (25.75, 41.666666666666664), (25.929166666666667, 44.5), (26.0, 46.81818181818182), (26.25, 48.333333333333336), (26.5, 50.38461538461539), (26.75, 52.857142857142854), (27.0, 55.333333333333336), (27.25, 57.8125), (27.5, 60.294117647058826), (27.75, 62.5), (28.0, 64.47368421052632), (28.25, 67.5), (28.5, 71.19047619047619), (28.75, 74.77272727272727), (29.0, 77.6086956521739), (29.25, 79.58333333333333), (29.4625, 80.8), (29.5, 81.92307692307692), (29.602083333333333, 82.77777777777777), (29.75, 82.67857142857143), (30.0, 81.55172413793103), (30.25, 80.16666666666667), (30.5, 78.54838709677419), (30.75, 77.03125), (31.0, 75.60606060606061), (31.25, 74.11764705882354)]
        if show_plots:
            # preview before assertions
            plot("Comprehensive 3: Katrina Wind", dll_winds.to_list(), deltas=[0.5, 1, float("inf")],
                 expecteds=[expected_12, expected_24, expected_all])
        self.check_dll(expected_12, filtered_12)      # if failure here, see (3)
        self.check_dll(expected_24, filtered_24)      # if failure here, see (3)
        self.check_dll(expected_all, filtered_all)    # if failure here, see (3)

        # clean up for easier debugging
        del filtered_12, filtered_24, filtered_all
        del expected_12, expected_24, expected_all

        # (4) test on real-world pressure data
        # note that one day corresponds to one unit of time
        filtered_12 = flurricane(dll_pressures, delta=0.5)    # 12-hour filtering
        filtered_24 = flurricane(dll_pressures, delta=1)      # 1-day filtering
        filtered_all = flurricane(dll_pressures, delta=float("inf"))      # cumulative moving average
        expected_12 = [(23.75, 1008.0), (24.0, 1007.5), (24.25, 1007.3333333333334), (24.5, 1006.6666666666666), (24.75, 1005.3333333333334), (25.0, 1003.0), (25.25, 1000.0), (25.5, 997.0), (25.75, 993.0), (25.929166666666667, 988.6666666666666), (26.0, 987.25), (26.25, 985.5), (26.5, 983.0), (26.75, 978.0), (27.0, 968.6666666666666), (27.25, 959.0), (27.5, 950.3333333333334), (27.75, 946.6666666666666), (28.0, 943.6666666666666), (28.25, 939.6666666666666), (28.5, 926.6666666666666), (28.75, 913.6666666666666), (29.0, 905.3333333333334), (29.25, 906.6666666666666), (29.4625, 912.6666666666666), (29.5, 915.25), (29.602083333333333, 921.0), (29.75, 926.4), (30.0, 940.0), (30.25, 962.3333333333334), (30.5, 974.6666666666666), (30.75, 984.3333333333334), (31.0, 989.6666666666666), (31.25, 993.3333333333334)]
        expected_24 = [(23.75, 1008.0), (24.0, 1007.5), (24.25, 1007.3333333333334), (24.5, 1007.0), (24.75, 1006.2), (25.0, 1004.6), (25.25, 1002.6), (25.5, 1000.0), (25.75, 996.4), (25.929166666666667, 992.6), (26.0, 991.0), (26.25, 988.8333333333334), (26.5, 985.8333333333334), (26.75, 981.5), (27.0, 975.2), (27.25, 968.6), (27.5, 959.6), (27.75, 953.4), (28.0, 948.0), (28.25, 942.2), (28.5, 934.0), (28.75, 926.0), (29.0, 917.4), (29.25, 911.8), (29.4625, 909.8), (29.5, 912.0), (29.602083333333333, 915.1666666666666), (29.75, 919.8571428571429), (30.0, 928.2857142857143), (30.25, 938.7142857142857), (30.5, 953.8333333333334), (30.75, 972.4), (31.0, 981.6), (31.25, 988.6)]
        expected_all = [(23.75, 1008.0), (24.0, 1007.5), (24.25, 1007.3333333333334), (24.5, 1007.0), (24.75, 1006.2), (25.0, 1005.1666666666666), (25.25, 1004.0), (25.5, 1002.75), (25.75, 1001.1111111111111), (25.929166666666667, 999.4), (26.0, 997.9090909090909), (26.25, 997.0), (26.5, 995.6153846153846), (26.75, 993.6428571428571), (27.0, 991.3333333333334), (27.25, 988.75), (27.5, 986.0), (27.75, 983.8888888888889), (28.0, 981.6315789473684), (28.25, 979.05), (28.5, 975.7142857142857), (28.75, 972.3636363636364), (29.0, 969.4347826086956), (29.25, 967.0833333333334), (29.4625, 965.2), (29.5, 963.5769230769231), (29.602083333333333, 962.2592592592592), (29.75, 961.75), (30.0, 961.7241379310345), (30.25, 962.2666666666667), (30.5, 963.0), (30.75, 963.84375), (31.0, 964.7575757575758), (31.25, 965.6764705882352)]
        if show_plots:
            # preview before assertions
            plot("Comprehensive 4: Katrina Pressure", dll_pressures.to_list(), deltas=[0.5, 1, float("inf")],
                 expecteds=[expected_12, expected_24, expected_all])
        self.check_dll(expected_12, filtered_12)      # if failure here, see (4)
        self.check_dll(expected_24, filtered_24)      # if failure here, see (4)
        self.check_dll(expected_all, filtered_all)    # if failure here, see (4)

        # clean up for easier debugging
        del filtered_12, filtered_24, filtered_all
        del expected_12, expected_24, expected_all


    def test_readme_xml_validity(self):

        path = "README.xml"
        xml_doc = minidom.parse(path)
        response = {}
        tags = ["netid", "feedback", "difficulty", "time", "citations", "type", "number"]

        # assert that we can access all tags
        for tag in tags:
            raw = xml_doc.getElementsByTagName(tag)[0].firstChild.nodeValue
            lines = [s.strip() for s in raw.split("\n")]  # if multiple lines, strip each line
            clean = " ".join(lines).strip()  # rejoin lines with spaces and strip leading space
            self.assertNotEqual("REPLACE", clean)  # make sure entry was edited
            response[tag] = clean  # save each entry

        # assert that difficulty is a float between 0-10
        difficulty_float = float(response["difficulty"])
        self.assertGreaterEqual(difficulty_float, 0.0)
        self.assertLessEqual(difficulty_float, 10.0)

        # assert that hours is a float between 0-100 (hopefully it didn't take 100 hours!)
        time_float = float(response["time"])
        self.assertGreaterEqual(time_float, 0.0)
        self.assertLessEqual(time_float, 100.0)

        # assert assignment type and number was not changed
        self.assertEqual("Project", response["type"])
        self.assertEqual("1", response["number"])

    def test_cProfile(self):

        # Use cProfile to gain insight into your code's runtime complexity!
        # It is a very powerful tool for use in CSE 331 and beyond.
        #  - Change the first argument to runctx to profile different testcases
        #  - Change the sort argument to one of "calls", "cumulative", "tottime", "name", line", "module"
        # More information is available at https://docs.python.org/3/library/profile.html

        # Comment out `self.assertEqual(True, False)` and uncomment the following line to profile one of your testcases.
        # Comment out both `cProfile` and `self.assertEqual(True, False)` lines to pass this testcase.
        # cProfile.runctx("DLLTests.test_DLL_comprehensive(self)", globals(), locals(), sort="calls")
        # self.assertEqual(True, False)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
