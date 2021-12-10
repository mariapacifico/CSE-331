"""
Project 2 - Hybrid Sorting - Tests
CSE 331 Fall 2021
Professor Sebnem Onsay
"""

import unittest
from HybridSort import insertion_sort, merge_sort, hybrid_sort, inversions_count, reverse_sort, Ship, navigation_test
# from HybridSort import insertion_sort, merge_sort, hybrid_sort, inversions_count, reverse_sort, password_sort, Ship, navigation_test
from random import seed, sample, randint, shuffle
from itertools import permutations
# import matplotlib.pyplot as plt
from typing import Dict, List


def plot_ships(ships: Dict[Ship, List[Ship]]):
    xCords = []
    yCords = []
    for ship in ships.keys():
        xCords.append(ship.x)
        yCords.append(ship.y)
    plt.scatter(xCords, yCords, marker="*")
    plt.show()


seed(331)


class Project2Tests(unittest.TestCase):

    def test_insertion_sort_basic(self):
        # (1) test with basic list of integers - default comparator
        data = [7, 4, 1, 0, 8, 9, 3, 2, 12]
        expected = sorted(data)
        insertion_sort(data)
        self.assertEqual(expected, data)

        # (2) test with basic list of strings - default comparator
        data = ["dog", "banana", "orange", "tree", "clutter", "candy", "silence"]
        expected = sorted(data)
        insertion_sort(data)
        self.assertEqual(expected, data)

        # (3) test with already sorted data - default comparator
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        expected = sorted(data)
        insertion_sort(data)
        self.assertEqual(expected, data)

        # (4) test empty
        data = []
        insertion_sort(data)
        self.assertEqual([], data)

        # (5) check that function does not return anything
        data = [5, 6, 3, 2]
        self.assertIsNone(insertion_sort(data))

    def test_insertion_sort_comparator(self):
        # (1) sort powers of ten by number of digits, in reverse
        data = [10 ** i for i in range(15)]
        shuffle(data)
        expected = sorted(data, key=lambda x: -1 * len(str(x)))
        insertion_sort(data, comparator=lambda x, y: len(str(x)) > len(str(y)))
        self.assertEqual(expected, data)

        # (2) sort strings by length
        data = ['a' * i for i in range(15)]
        shuffle(data)
        expected = sorted(data, key=lambda x: len(x))
        insertion_sort(data, comparator=lambda x, y: len(x) <= len(y))
        self.assertEqual(expected, data)

    def test_insertion_sort_comprehensive(self):
        # (1) sort a lot of integers
        data = list(range(1500))
        shuffle(data)
        expected = sorted(data)
        insertion_sort(data)
        self.assertEqual(expected, data)

        # (2) sort a lot of integers with alternative comparator
        # this comparator (defined as a named lambda) compares values as follows:
        #   x <= y
        #   if and only if
        #   sum(digits(x)) <= sum(digits(y))
        # ex: 12 <= 15 since 1 + 2 = 3 <= 6 = 1 + 5
        comp = lambda x, y: sum([int(digit) for digit in str(x)]) <= sum([int(digit) for digit in str(y)])
        data = list(range(1500))
        expected = sorted(data, key=lambda x: sum([int(digit) for digit in str(x)]))
        insertion_sort(data, comparator=comp)
        # there are multiple possible orderings, thus we must compare via sums of digits
        for index, item in enumerate(expected):
            expected_sum = sum([int(digit) for digit in str(item)])
            actual_sum = sum([int(digit) for digit in str(data[index])])
            self.assertEqual(expected_sum, actual_sum)

    def test_merge_sort_basic(self):
        # (1) test with basic list of integers - default comparator and threshold
        data = [7, 4, 1, 0, 8, 9, 3, 2, 12]
        expected = sorted(data)
        merge_sort(data)
        self.assertEqual(expected, data)

        # (2) test with basic set of strings - default comparator and threshold
        data = ["dog", "banana", "orange", "tree", "clutter", "candy", "silence"]
        expected = sorted(data)
        merge_sort(data)
        self.assertEqual(expected, data)

        # (3) test with already sorted data - default comparator and threshold
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        expected = sorted(data)
        merge_sort(data)
        self.assertEqual(expected, data)

        # (4) test empty - default comparator and threshold
        data = []
        merge_sort(data)
        self.assertEqual([], data)

    def test_merge_sort_threshold(self):

        # first, all the tests from basic should work with higher thresholds

        # (1) test with basic list of integers - default comparator
        data = [7, 4, 1, 0, 8, 9, 3, 2, 12]
        expected = sorted(data)
        merge_sort(data, threshold=2)
        self.assertEqual(expected, data)

        # (2) test with basic set of strings - default comparator
        data = ["dog", "banana", "orange", "tree", "clutter", "candy", "silence"]
        expected = sorted(data)
        merge_sort(data, threshold=2)
        self.assertEqual(expected, data)

        # (3) test with already sorted data - default comparator
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        expected = sorted(data)
        merge_sort(data, threshold=2)
        self.assertEqual(expected, data)

        # (4) now, for a longer test - a bunch of thresholds
        data = list(range(25))
        expected = sorted(data)
        for t in range(11):
            shuffle(data )
            merge_sort(data, threshold=t)
            self.assertEqual(expected, data)

    def test_merge_sort_comparator(self):
        # (1) sort powers of ten by number of digits, in reverse
        data = [10 ** i for i in range(15)]
        shuffle(data)
        expected = sorted(data, key=lambda x: -1 * len(str(x)))
        merge_sort(data, comparator=lambda x, y: len(str(x)) > len(str(y)))
        self.assertEqual(expected, data)

        # (2) sort strings by length
        data = ['a' * i for i in range(15)]
        shuffle(data)
        expected = sorted(data, key=lambda x: len(x))
        merge_sort(data, comparator=lambda x, y: len(x) <= len(y))
        self.assertEqual(expected, data)

    def test_merge_sort_comprehensive(self):
        # (1) sort a lot of integers, with a lot of thresholds
        data = list(range(500))
        for t in range(100):
            shuffle(data)
            expected = sorted(data)
            merge_sort(data, threshold=t)
            self.assertEqual(expected, data)

        # (2) sort a lot of integers with alternative comparator, threshold of 8
        # this comparator (defined as a named lambda) compares values as follows:
        #   x <= y
        #   if and only if
        #   sum(digits(x)) <= sum(digits(y))
        # ex: 12 <= 15 since 1 + 2 = 3 <= 6 = 1 + 5
        comp = lambda x, y: sum([int(digit) for digit in str(x)]) <= sum([int(digit) for digit in str(y)])
        data = list(range(1500))
        expected = sorted(data, key=lambda x: sum([int(digit) for digit in str(x)]))
        merge_sort(data, threshold=8, comparator=comp)
        # there are multiple possible orderings, thus we must compare via sums of digits
        for index, item in enumerate(expected):
            expected_sum = sum([int(digit) for digit in str(item)])
            actual_sum = sum([int(digit) for digit in str(data[index])])
            self.assertEqual(expected_sum, actual_sum)

    def test_hybrid_sort_basic(self):
        # (1) test with basic list of integers - default comparator, threshold of zero
        data = [7, 4, 1, 0, 8, 9, 3, 2, 12]
        expected = sorted(data)
        hybrid_sort(data, threshold=0)
        self.assertEqual(expected, data)

        # (2) test with basic list of strings - default comparator, threshold
        data = ["dog", "banana", "orange", "tree", "clutter", "candy", "silence"]
        expected = sorted(data)
        hybrid_sort(data, threshold=0)
        self.assertEqual(expected, data)

        # (3) test with already sorted data - default comparator, threshold
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        expected = sorted(data)
        hybrid_sort(data, threshold=0)
        self.assertEqual(expected, data)

        # (4) test empty
        data = []
        hybrid_sort(data, threshold=0)
        self.assertEqual([], data)

        # (5) check that function does not return anything
        data = [5, 6, 3, 2]
        self.assertIsNone(hybrid_sort(data, threshold=0))

    def test_hybrid_sort_comprehensive(self):
        # this should be easy to pass if you've passed the comprehensive tests for merge and insertion

        # (1) sort a lot of integers with alternative comparator, thresholds in [1,...,49]
        # this comparator (defined as a named lambda) compares values as follows:
        #   x <= y
        #   if and only if
        #   sum(digits(x)) <= sum(digits(y))
        # ex: 12 <= 15 since 1 + 2 = 3 <= 6 = 1 + 5
        comp = lambda x, y: sum([int(digit) for digit in str(x)]) <= sum([int(digit) for digit in str(y)])
        data = list(range(1000))
        expected = sorted(data, key=lambda x: sum([int(digit) for digit in str(x)]))
        for t in range(50):
            shuffle(data)
            hybrid_sort(data, threshold=t, comparator=comp)
            for index, item in enumerate(expected):
                expected_sum = sum([int(digit) for digit in str(item)])
                actual_sum = sum([int(digit) for digit in str(data[index])])

    def test_hybrid_sort_speed(self):
        # ***WORTH NO POINTS, FOR PERSONAL TESTING PURPOSES ONLY***
        # the point of this sort is to be fast, right?
        # this (probably) won't pass if you're not careful with time complexity
        data = list(range(300000))
        expected = data[:]
        shuffle(data)
        hybrid_sort(data, threshold=10)
        self.assertEqual(expected, data)

    def test_reverse_sort_basic(self):
        # (1) test with basic list of integers - default comparator, threshold of zero
        data = [7, 4, 1, 0, 8, 9, 3, 2, 12]
        expected = sorted(data, reverse=True)
        reverse_sort(data, threshold=0)
        self.assertEqual(expected, data)

        # (2) test with basic list of strings - default comparator, threshold
        data = ["dog", "banana", "orange", "tree", "clutter", "candy", "silence"]
        expected = sorted(data, reverse=True)
        reverse_sort(data, threshold=0)
        self.assertEqual(expected, data)

        # (3) test with already sorted data - default comparator, threshold
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        expected = sorted(data, reverse=True)
        reverse_sort(data, threshold=0)
        self.assertEqual(expected, data)

        # (4) test empty
        data = []
        reverse_sort(data, threshold=0)
        self.assertEqual([], data)

        # (5) check that function does not return anything
        data = [5, 6, 3, 2]
        self.assertIsNone(reverse_sort(data, threshold=0))

        # (6) now let's test with multiple thresholds
        data = list(range(50))
        expected = sorted(data, reverse=True)
        for t in range(20):
            shuffle(data)
            reverse_sort(data, threshold=t)
            self.assertEqual(expected, data)

    def test_reverse_sort_speed(self):
        # ***WORTH NO POINTS, FOR PERSONAL TESTING PURPOSES ONLY***
        # should be just as fast as hybrid sort
        data = list(range(300000))
        expected = sorted(data, reverse=True)
        shuffle(data)
        reverse_sort(data, threshold=10)
        self.assertEqual(expected, data)

    def test_inversion_count(self):
        # (1)
        data = [2, 4, 3, 1]
        val = inversions_count(data)
        self.assertEqual(4, inversions_count(data))

        # (2)
        data = [4, 3, 2, 1]
        val = inversions_count(data)
        self.assertEqual(6, inversions_count(data))

        # (3)
        data = [1, 2, 3, 4]
        self.assertEqual(0, inversions_count(data))

        # (4)
        data = [2, 4, 1, 3, 5]
        self.assertEqual(3, inversions_count(data))

        # (5)
        data = [5, 4, 3, 2, 1]
        val = inversions_count(data)
        self.assertEqual(10, inversions_count(data))
        self.assertEqual([5, 4, 3, 2, 1], data)

        # (6)
        data = [1, 2, 3, 4, 5]
        self.assertEqual(0, inversions_count(data))

        # random tests
        # (7)
        seed(1130)
        data = [randint(0, 100) for _ in range(10)]
        self.assertEqual(30, inversions_count(data))

        # (8)
        in_order = sorted(data)
        self.assertEqual(0, inversions_count(in_order))

        # (9)
        reverse = sorted(data, reverse=True)
        self.assertEqual(45, inversions_count(reverse))

        # (10)
        data = [randint(0, 100) for _ in range(11)]
        self.assertEqual(27, inversions_count(data))

        # (11)
        in_order = sorted(data)
        self.assertEqual(0, inversions_count(in_order))

        # (12)
        reverse = sorted(data, reverse=True)
        self.assertEqual(55, inversions_count(reverse))

        seed(22)

        # large even length list
        # (13)
        data = [randint(0, 10000) for _ in range(100)]
        self.assertEqual(2430, inversions_count(data))

        # (14)
        reverse = sorted(data, reverse=True)
        self.assertEqual(4950, inversions_count(reverse))

        # (15)
        in_order = sorted(data)
        self.assertEqual(0, inversions_count(in_order))

        # large odd length list
        # (16)
        data = [randint(0, 10000) for _ in range(101)]
        self.assertEqual(2530, inversions_count(data))

        # (17)
        reverse = sorted(data, reverse=True)
        self.assertEqual(5050, inversions_count(reverse))

        # (18)
        in_order = sorted(data)
        self.assertEqual(0, inversions_count(in_order))

    def test_plotter(self):
        # ***NOT ON MIMIR: WORTH NO POINTS, FOR PERSONAL TESTING PURPOSES ONLY***
        seed(331)
        ships = [Ship('a' * i, randint(-10, 10), randint(-10, 10)) for i in range(1, 50)]
        ships_dict = {}
        for ship in ships:
            ships_dict[ship] = [s for s in ships if s is not ship]
        plot_ships(ships_dict)

    def test_application_small(self):
        seed(331)

        # #(1) No ships
        # ships = []
        # ships_dict = {}
        # actual = str(navigation_test(ships_dict))
        # expected = '[]'
        # self.assertEqual(expected, actual)

        #(2) One ship
        ships_dict = {Ship('1', 0, 0):[]}
        actual = str(navigation_test(ships_dict))
        expected = '[Ship: 1 x=0 y=0]'
        self.assertEqual(expected, actual)

        # (3)
        ships = [Ship(str(i), randint(-10, 10), randint(-10, 10)) for i in range(1, 5)]
        ships_dict = {}
        for ship in ships:
            ships_dict[ship] = [s for s in ships if s is not ship]
        actual = str(navigation_test(ships_dict))
        expected = '[Ship: 4 x=5 y=4, Ship: 1 x=0 y=-5, Ship: 2 x=-3 y=10, Ship: 3 x=-7 y=10]'
        self.assertEqual(expected, actual)

        # (4)
        ships = [Ship(str(i), randint(-10, 10), randint(-10, 10)) for i in range(1, 10)]
        ships_dict = {}
        for ship in ships:
            ships_dict[ship] = [s for s in ships if s is not ship]
        actual = str(navigation_test(ships_dict))
        expected = '[Ship: 7 x=9 y=0, Ship: 1 x=9 y=0, Ship: 2 x=8 y=3, Ship: 8 x=3 y=7, '\
                   'Ship: 9 x=-3 y=-6, Ship: 3 x=3 y=9, Ship: 5 x=3 y=7, Ship: 6 x=-1 y=6, Ship: 4 x=3 y=-9]'
        self.assertEqual(expected, actual)

        # (5) Test taxicab distance
        actual = str(navigation_test(ships_dict, False))
        expected = '[Ship: 1 x=9 y=0, Ship: 2 x=8 y=3, Ship: 7 x=9 y=0, Ship: 3 x=3 y=9, Ship: 5 x=3 y=7, ' \
                   'Ship: 8 x=3 y=7, Ship: 6 x=-1 y=6, Ship: 9 x=-3 y=-6, Ship: 4 x=3 y=-9]'
        self.assertEqual(expected, actual)

    def test_application_large(self):
        seed(331)

        # (1)
        ships = [Ship(str(i), randint(-15, 15), randint(-15, 15)) for i in range(1, 20)]
        ships_dict = {}
        for ship in ships:
            ships_dict[ship] = [s for s in ships if s is not ship]
        actual = str(navigation_test(ships_dict))
        expected = '[Ship: 17 x=-2 y=-14, Ship: 13 x=6 y=11, Ship: 15 x=12 y=7, Ship: 11 x=12 y=8, ' \
                   'Ship: 16 x=13 y=4, Ship: 8 x=15 y=14, Ship: 19 x=-6 y=13, Ship: 18 x=-2 y=2, ' \
                   'Ship: 4 x=12 y=11, Ship: 12 x=-2 y=12, Ship: 10 x=3 y=8, Ship: 14 x=-2 y=9, ' \
                   'Ship: 5 x=5 y=-12, Ship: 1 x=11 y=8, Ship: 6 x=5 y=0, Ship: 9 x=6 y=-5, ' \
                   'Ship: 3 x=-10 y=-8, Ship: 7 x=-1 y=4, Ship: 2 x=-5 y=12]'
        self.assertEqual(expected, actual)

        # (2)
        ships = [Ship(str(i), randint(-15, 15), randint(-15, 15)) for i in range(1, 50)]
        ships_dict = {}
        for ship in ships:
            ships_dict[ship] = [s for s in ships if s is not ship]
        actual = str(navigation_test(ships_dict))
        expected = '[Ship: 28 x=0 y=9, Ship: 33 x=4 y=6, Ship: 23 x=2 y=1, Ship: 1 x=1 y=8, ' \
                   'Ship: 22 x=0 y=1, Ship: 9 x=4 y=4, Ship: 19 x=5 y=7, Ship: 46 x=10 y=11, ' \
                   'Ship: 48 x=10 y=15, Ship: 27 x=6 y=14, Ship: 31 x=6 y=15, Ship: 39 x=9 y=1, ' \
                   'Ship: 11 x=7 y=3, Ship: 2 x=4 y=9, Ship: 34 x=12 y=7, Ship: 13 x=-4 y=8, ' \
                   'Ship: 26 x=-6 y=13, Ship: 4 x=-2 y=2, Ship: 12 x=5 y=15, Ship: 37 x=-12 y=15, ' \
                   'Ship: 8 x=-5 y=9, Ship: 3 x=-5 y=10, Ship: 16 x=14 y=7, Ship: 5 x=15 y=11, ' \
                   'Ship: 15 x=13 y=1, Ship: 17 x=14 y=1, Ship: 45 x=-14 y=9, Ship: 14 x=-7 y=2, ' \
                   'Ship: 35 x=12 y=-5, Ship: 44 x=7 y=-8, Ship: 43 x=11 y=-10, Ship: 30 x=-14 y=3, ' \
                   'Ship: 42 x=9 y=-11, Ship: 24 x=8 y=-9, Ship: 36 x=-4 y=-7, Ship: 47 x=-15 y=-6, ' \
                   'Ship: 25 x=10 y=-12, Ship: 6 x=6 y=-8, Ship: 41 x=-4 y=-9, Ship: 21 x=5 y=-11, ' \
                   'Ship: 29 x=-14 y=-5, Ship: 7 x=8 y=-11, Ship: 40 x=-14 y=-11, Ship: 49 x=-6 y=-10, ' \
                   'Ship: 38 x=-9 y=-10, Ship: 32 x=-14 y=-9, Ship: 18 x=-14 y=-15, Ship: 20 x=-1 y=-14, ' \
                   'Ship: 10 x=-10 y=-13]'
        self.assertEqual(expected, actual)

        # (3)
        ships = [Ship(str(i), randint(-15, 15), randint(-15, 15)) for i in range(1, 331)]
        ships_dict = {}
        for ship in ships:
            ships_dict[ship] = [s for s in ships if s is not ship]
        actual = str(navigation_test(ships_dict))
        expected = '[Ship: 310 x=-10 y=-4, Ship: 321 x=-7 y=-7, Ship: 180 x=-9 y=-2, Ship: 174 x=-9 y=-3, ' \
                   'Ship: 243 x=-6 y=-5, Ship: 273 x=-9 y=-6, Ship: 280 x=-12 y=-4, Ship: 166 x=-11 y=-4, ' \
                   'Ship: 119 x=-6 y=-3, Ship: 303 x=-13 y=-7, Ship: 290 x=-14 y=-3, Ship: 210 x=-9 y=-6, ' \
                   'Ship: 271 x=-7 y=0, Ship: 196 x=-12 y=-5, Ship: 286 x=-12 y=-7, Ship: 246 x=-15 y=-4, ' \
                   'Ship: 326 x=-7 y=2, Ship: 73 x=-7 y=-1, Ship: 276 x=-14 y=-8, Ship: 146 x=-12 y=-4, ' \
                   'Ship: 197 x=-10 y=0, Ship: 53 x=-7 y=-1, Ship: 75 x=-8 y=-5, Ship: 142 x=-13 y=-3, ' \
                   'Ship: 161 x=-12 y=-1, Ship: 39 x=-7 y=-5, Ship: 48 x=-9 y=-1, Ship: 128 x=-4 y=-5, ' \
                   'Ship: 22 x=-7 y=-1, Ship: 35 x=-10 y=-1, Ship: 42 x=-4 y=-3, Ship: 74 x=-4 y=-4, ' \
                   'Ship: 167 x=-13 y=-7, Ship: 237 x=-7 y=2, Ship: 187 x=-14 y=-2, Ship: 100 x=-8 y=0, ' \
                   'Ship: 232 x=-10 y=-9, Ship: 61 x=-13 y=-3, Ship: 263 x=-12 y=-11, Ship: 194 x=-12 y=0, ' \
                   'Ship: 231 x=-11 y=1, Ship: 69 x=-13 y=-2, Ship: 254 x=-15 y=-1, Ship: 275 x=-10 y=-13, ' \
                   'Ship: 144 x=-12 y=-10, Ship: 154 x=-2 y=0, Ship: 211 x=-2 y=1, Ship: 27 x=-13 y=-2, ' \
                   'Ship: 202 x=-11 y=-12, Ship: 109 x=-14 y=-12, Ship: 189 x=-10 y=2, Ship: 230 x=-15 y=0, ' \
                   'Ship: 267 x=-10 y=-14, Ship: 117 x=-10 y=1, Ship: 249 x=-2 y=-6, Ship: 185 x=-11 y=2, ' \
                   'Ship: 226 x=-9 y=3, Ship: 62 x=-13 y=-10, Ship: 24 x=-4 y=-7, Ship: 9 x=-15 y=-7, ' \
                   'Ship: 151 x=-15 y=-14, Ship: 235 x=-1 y=-1, Ship: 181 x=-10 y=-12, Ship: 277 x=-6 y=-12, ' \
                   'Ship: 101 x=-14 y=0, Ship: 80 x=-14 y=-13, Ship: 30 x=-2 y=-5, Ship: 102 x=-4 y=2, ' \
                   'Ship: 26 x=-2 y=-5, Ship: 110 x=-5 y=-10, Ship: 56 x=-12 y=-11, Ship: 320 x=-7 y=4, ' \
                   'Ship: 268 x=-10 y=4, Ship: 219 x=-10 y=-15, Ship: 312 x=-1 y=-6, Ship: 4 x=-14 y=-1, ' \
                   'Ship: 241 x=-5 y=4, Ship: 162 x=-5 y=-11, Ship: 89 x=-2 y=-6, Ship: 288 x=-6 y=-14, ' \
                   'Ship: 206 x=-8 y=4, Ship: 33 x=-7 y=-11, Ship: 124 x=-3 y=3, Ship: 257 x=-15 y=5, ' \
                   'Ship: 43 x=-10 y=3, Ship: 138 x=-6 y=-13, Ship: 245 x=-3 y=4, Ship: 258 x=0 y=-2, ' \
                   'Ship: 105 x=-10 y=4, Ship: 147 x=-7 y=-14, Ship: 190 x=0 y=0, Ship: 17 x=-14 y=2, ' \
                   'Ship: 173 x=-5 y=5, Ship: 250 x=-7 y=5, Ship: 38 x=-4 y=-11, Ship: 3 x=-9 y=-14, ' \
                   'Ship: 193 x=-5 y=-15, Ship: 57 x=-2 y=3, Ship: 65 x=-4 y=-12, Ship: 107 x=-1 y=3, ' \
                   'Ship: 177 x=-13 y=7, Ship: 140 x=-1 y=-8, Ship: 204 x=-4 y=6, Ship: 44 x=-15 y=4, ' \
                   'Ship: 213 x=0 y=-6, Ship: 126 x=-14 y=7, Ship: 240 x=-3 y=6, Ship: 18 x=-1 y=-7, ' \
                   'Ship: 51 x=-1 y=3, Ship: 201 x=1 y=-3, Ship: 259 x=0 y=-7, Ship: 291 x=1 y=-6, ' \
                   'Ship: 287 x=1 y=-6, Ship: 64 x=-5 y=-15, Ship: 298 x=-15 y=11, Ship: 132 x=-1 y=4, ' \
                   'Ship: 264 x=-12 y=9, Ship: 122 x=1 y=-2, Ship: 11 x=-5 y=-14, Ship: 269 x=-15 y=11, ' \
                   'Ship: 31 x=1 y=-1, Ship: 137 x=-7 y=6, Ship: 217 x=-3 y=-15, Ship: 104 x=-11 y=7, ' \
                   'Ship: 313 x=-6 y=8, Ship: 112 x=-3 y=6, Ship: 169 x=0 y=3, Ship: 266 x=-7 y=8, ' \
                   'Ship: 47 x=-14 y=8, Ship: 221 x=-10 y=9, Ship: 272 x=2 y=-5, Ship: 96 x=1 y=-5, ' \
                   'Ship: 304 x=0 y=-12, Ship: 205 x=0 y=5, Ship: 292 x=-11 y=12, Ship: 214 x=0 y=-11, ' \
                   'Ship: 284 x=0 y=-12, Ship: 136 x=-6 y=7, Ship: 113 x=1 y=-6, Ship: 330 x=-4 y=9, ' \
                   'Ship: 195 x=-2 y=7, Ship: 157 x=-8 y=8, Ship: 46 x=0 y=-8, Ship: 135 x=-7 y=8, ' \
                   'Ship: 15 x=-12 y=8, Ship: 297 x=-3 y=10, Ship: 21 x=1 y=-6, Ship: 103 x=-15 y=14, ' \
                   'Ship: 139 x=-13 y=13, Ship: 84 x=0 y=-11, Ship: 116 x=-15 y=15, Ship: 49 x=1 y=1, ' \
                   'Ship: 289 x=-9 y=13, Ship: 158 x=-13 y=14, Ship: 34 x=-8 y=8, Ship: 300 x=0 y=8, ' \
                   'Ship: 93 x=-9 y=10, Ship: 305 x=1 y=-13, Ship: 168 x=-13 y=15, Ship: 148 x=1 y=5, ' \
                   'Ship: 281 x=3 y=1, Ship: 72 x=-4 y=8, Ship: 70 x=-4 y=8, Ship: 12 x=0 y=5, ' \
                   'Ship: 318 x=-4 y=12, Ship: 165 x=2 y=2, Ship: 274 x=2 y=6, Ship: 127 x=-4 y=9, ' \
                   'Ship: 88 x=-2 y=8, Ship: 115 x=-6 y=9, Ship: 19 x=1 y=3, Ship: 203 x=0 y=-15, ' \
                   'Ship: 58 x=2 y=1, Ship: 224 x=-10 y=15, Ship: 170 x=-5 y=11, Ship: 85 x=-6 y=10, ' \
                   'Ship: 278 x=3 y=-7, Ship: 77 x=-9 y=12, Ship: 153 x=1 y=-14, Ship: 121 x=-4 y=11, ' \
                   'Ship: 179 x=-9 y=15, Ship: 1 x=1 y=-11, Ship: 155 x=-5 y=12, Ship: 301 x=3 y=-10, ' \
                   'Ship: 233 x=4 y=3, Ship: 188 x=-6 y=14, Ship: 324 x=5 y=5, Ship: 40 x=0 y=-14, ' \
                   'Ship: 125 x=3 y=3, Ship: 248 x=-2 y=12, Ship: 108 x=-5 y=12, Ship: 111 x=3 y=5, ' \
                   'Ship: 296 x=4 y=-12, Ship: 234 x=3 y=-14, Ship: 13 x=1 y=-12, Ship: 227 x=3 y=-10, ' \
                   'Ship: 218 x=4 y=-4, Ship: 247 x=4 y=-13, Ship: 10 x=2 y=-9, Ship: 83 x=3 y=4, ' \
                   'Ship: 20 x=1 y=-14, Ship: 212 x=-1 y=12, Ship: 106 x=2 y=-15, Ship: 208 x=3 y=-15, ' \
                   'Ship: 182 x=4 y=-5, Ship: 76 x=4 y=-2, Ship: 14 x=2 y=-11, Ship: 79 x=2 y=-14, ' \
                   'Ship: 90 x=3 y=-6, Ship: 97 x=3 y=-13, Ship: 86 x=3 y=-13, Ship: 118 x=-2 y=12, ' \
                   'Ship: 32 x=4 y=-2, Ship: 143 x=-4 y=15, Ship: 238 x=4 y=-10, Ship: 41 x=3 y=-13, ' \
                   'Ship: 23 x=4 y=3, Ship: 198 x=4 y=8, Ship: 60 x=-2 y=12, Ship: 242 x=2 y=12, ' \
                   'Ship: 114 x=4 y=-11, Ship: 5 x=4 y=6, Ship: 328 x=7 y=-14, Ship: 295 x=6 y=-2, ' \
                   'Ship: 283 x=1 y=15, Ship: 200 x=3 y=11, Ship: 319 x=5 y=-8, Ship: 228 x=3 y=13, ' \
                   'Ship: 317 x=5 y=-6, Ship: 183 x=5 y=-11, Ship: 256 x=4 y=12, Ship: 68 x=3 y=9, ' \
                   'Ship: 311 x=8 y=1, Ship: 325 x=6 y=13, Ship: 191 x=7 y=6, Ship: 309 x=7 y=15, ' \
                   'Ship: 95 x=0 y=15, Ship: 184 x=7 y=1, Ship: 186 x=9 y=5, Ship: 223 x=8 y=2, ' \
                   'Ship: 2 x=6 y=1, Ship: 159 x=7 y=0, Ship: 252 x=8 y=9, Ship: 260 x=8 y=12, ' \
                   'Ship: 255 x=8 y=8, Ship: 78 x=3 y=13, Ship: 215 x=8 y=-13, Ship: 285 x=9 y=13, ' \
                   'Ship: 306 x=6 y=-8, Ship: 307 x=6 y=-8, Ship: 261 x=9 y=12, Ship: 130 x=8 y=-15, ' \
                   'Ship: 199 x=7 y=-11, Ship: 175 x=9 y=4, Ship: 156 x=7 y=12, Ship: 239 x=10 y=9, ' \
                   'Ship: 123 x=9 y=4, Ship: 323 x=6 y=-5, Ship: 229 x=6 y=-8, Ship: 308 x=6 y=-5, ' \
                   'Ship: 282 x=7 y=-3, Ship: 302 x=13 y=7, Ship: 7 x=8 y=6, Ship: 327 x=14 y=9, ' \
                   'Ship: 87 x=9 y=-14, Ship: 45 x=9 y=8, Ship: 29 x=5 y=14, Ship: 262 x=9 y=-2, ' \
                   'Ship: 270 x=12 y=15, Ship: 220 x=12 y=6, Ship: 329 x=13 y=3, Ship: 236 x=11 y=11, ' \
                   'Ship: 54 x=7 y=12, Ship: 293 x=15 y=-15, Ship: 207 x=10 y=-13, Ship: 129 x=6 y=-8, ' \
                   'Ship: 265 x=11 y=-1, Ship: 134 x=9 y=-1, Ship: 178 x=13 y=9, Ship: 294 x=12 y=-2, ' \
                   'Ship: 216 x=15 y=15, Ship: 279 x=14 y=0, Ship: 314 x=15 y=-2, Ship: 299 x=11 y=-3, ' \
                   'Ship: 92 x=9 y=-1, Ship: 172 x=12 y=4, Ship: 81 x=11 y=4, Ship: 120 x=10 y=0, ' \
                   'Ship: 149 x=14 y=10, Ship: 244 x=10 y=-3, Ship: 164 x=13 y=4, Ship: 131 x=14 y=15, ' \
                   'Ship: 94 x=12 y=6, Ship: 91 x=12 y=14, Ship: 152 x=14 y=3, Ship: 50 x=7 y=-3, ' \
                   'Ship: 315 x=13 y=-4, Ship: 322 x=11 y=-9, Ship: 66 x=14 y=14, Ship: 16 x=12 y=10, ' \
                   'Ship: 82 x=14 y=8, Ship: 98 x=14 y=4, Ship: 251 x=9 y=-8, Ship: 67 x=9 y=-3, ' \
                   'Ship: 6 x=9 y=-11, Ship: 222 x=15 y=-12, Ship: 253 x=13 y=-8, Ship: 316 x=12 y=-6, ' \
                   'Ship: 163 x=15 y=-13, Ship: 225 x=13 y=-10, Ship: 71 x=15 y=6, Ship: 133 x=15 y=1, ' \
                   'Ship: 150 x=14 y=-1, Ship: 209 x=14 y=-10, Ship: 192 x=8 y=-5, Ship: 37 x=15 y=13, ' \
                   'Ship: 99 x=15 y=4, Ship: 59 x=13 y=-1, Ship: 171 x=12 y=-9, Ship: 176 x=14 y=-4, ' \
                   'Ship: 141 x=13 y=-9, Ship: 145 x=13 y=-10, Ship: 8 x=15 y=5, Ship: 28 x=15 y=4, ' \
                   'Ship: 55 x=15 y=-1, Ship: 25 x=12 y=-3, Ship: 160 x=12 y=-6, Ship: 63 x=13 y=-5, ' \
                   'Ship: 52 x=11 y=-8, Ship: 36 x=14 y=-6]'
        self.assertEqual(expected, actual)

        # (4) Test taxicab distance
        actual = str(navigation_test(ships_dict, False))
        expected = '[Ship: 10 x=2 y=-9, Ship: 259 x=0 y=-7, Ship: 319 x=5 y=-8, Ship: 129 x=6 y=-8, ' \
                   'Ship: 229 x=6 y=-8, Ship: 306 x=6 y=-8, Ship: 307 x=6 y=-8, Ship: 113 x=1 y=-6, ' \
                   'Ship: 21 x=1 y=-6, Ship: 287 x=1 y=-6, Ship: 291 x=1 y=-6, Ship: 213 x=0 y=-6, ' \
                   'Ship: 227 x=3 y=-10, Ship: 301 x=3 y=-10, Ship: 238 x=4 y=-10, Ship: 75 x=-8 y=-5, ' \
                   'Ship: 278 x=3 y=-7, Ship: 321 x=-7 y=-7, Ship: 7 x=8 y=6, Ship: 191 x=7 y=6, ' \
                   'Ship: 312 x=-1 y=-6, Ship: 100 x=-8 y=0, Ship: 14 x=2 y=-11, Ship: 237 x=-7 y=2, ' \
                   'Ship: 326 x=-7 y=2, Ship: 140 x=-1 y=-8, Ship: 249 x=-2 y=-6, Ship: 89 x=-2 y=-6, ' \
                   'Ship: 18 x=-1 y=-7, Ship: 114 x=4 y=-11, Ship: 24 x=-4 y=-7, Ship: 183 x=5 y=-11, ' \
                   'Ship: 46 x=0 y=-8, Ship: 26 x=-2 y=-5, Ship: 30 x=-2 y=-5, Ship: 243 x=-6 y=-5, ' \
                   'Ship: 186 x=9 y=5, Ship: 110 x=-5 y=-10, Ship: 157 x=-8 y=8, Ship: 34 x=-8 y=8, ' \
                   'Ship: 272 x=2 y=-5, Ship: 39 x=-7 y=-5, Ship: 123 x=9 y=4, Ship: 175 x=9 y=4, ' \
                   'Ship: 317 x=5 y=-6, Ship: 232 x=-10 y=-9, Ship: 271 x=-7 y=0, Ship: 136 x=-6 y=7, ' \
                   'Ship: 96 x=1 y=-5, Ship: 112 x=-3 y=6, Ship: 240 x=-3 y=6, Ship: 214 x=0 y=-11, ' \
                   'Ship: 324 x=5 y=5, Ship: 84 x=0 y=-11, Ship: 90 x=3 y=-6, Ship: 1 x=1 y=-11, ' \
                   'Ship: 239 x=10 y=9, Ship: 313 x=-6 y=8, Ship: 308 x=6 y=-5, Ship: 323 x=6 y=-5, ' \
                   'Ship: 255 x=8 y=8, Ship: 210 x=-9 y=-6, Ship: 273 x=-9 y=-6, Ship: 211 x=-2 y=1, ' \
                   'Ship: 135 x=-7 y=8, Ship: 266 x=-7 y=8, Ship: 22 x=-7 y=-1, Ship: 53 x=-7 y=-1, ' \
                   'Ship: 73 x=-7 y=-1, Ship: 128 x=-4 y=-5, Ship: 223 x=8 y=2, Ship: 252 x=8 y=9, ' \
                   'Ship: 296 x=4 y=-12, Ship: 48 x=-9 y=-1, Ship: 137 x=-7 y=6, Ship: 310 x=-10 y=-4, ' \
                   'Ship: 204 x=-4 y=6, Ship: 45 x=9 y=8, Ship: 284 x=0 y=-12, Ship: 304 x=0 y=-12, ' \
                   'Ship: 197 x=-10 y=0, Ship: 180 x=-9 y=-2, Ship: 174 x=-9 y=-3, Ship: 311 x=8 y=1, ' \
                   'Ship: 173 x=-5 y=5, Ship: 115 x=-6 y=9, Ship: 250 x=-7 y=5, Ship: 70 x=-4 y=8, ' \
                   'Ship: 72 x=-4 y=8, Ship: 41 x=3 y=-13, Ship: 86 x=3 y=-13, Ship: 97 x=3 y=-13, ' \
                   'Ship: 13 x=1 y=-12, Ship: 199 x=7 y=-11, Ship: 206 x=-8 y=4, Ship: 102 x=-4 y=2, ' \
                   'Ship: 124 x=-3 y=3, Ship: 251 x=9 y=-8, Ship: 320 x=-7 y=4, Ship: 182 x=4 y=-5, ' \
                   'Ship: 117 x=-10 y=1, Ship: 35 x=-10 y=-1, Ship: 166 x=-11 y=-4, Ship: 235 x=-1 y=-1, ' \
                   'Ship: 322 x=11 y=-9, Ship: 57 x=-2 y=3, Ship: 162 x=-5 y=-11, Ship: 245 x=-3 y=4, ' \
                   'Ship: 2 x=6 y=1, Ship: 120 x=10 y=0, Ship: 93 x=-9 y=10, Ship: 184 x=7 y=1, ' \
                   'Ship: 247 x=4 y=-13, Ship: 33 x=-7 y=-11, Ship: 189 x=-10 y=2, Ship: 127 x=-4 y=9, ' \
                   'Ship: 330 x=-4 y=9, Ship: 154 x=-2 y=0, Ship: 241 x=-5 y=4, Ship: 74 x=-4 y=-4, ' \
                   'Ship: 119 x=-6 y=-3, Ship: 85 x=-6 y=10, Ship: 195 x=-2 y=7, Ship: 5 x=4 y=6, ' \
                   'Ship: 81 x=11 y=4, Ship: 185 x=-11 y=2, Ship: 226 x=-9 y=3, Ship: 231 x=-11 y=1, ' \
                   'Ship: 144 x=-12 y=-10, Ship: 38 x=-4 y=-11, Ship: 297 x=-3 y=10, Ship: 305 x=1 y=-13, ' \
                   'Ship: 134 x=9 y=-1, Ship: 92 x=9 y=-1, Ship: 221 x=-10 y=9, Ship: 201 x=1 y=-3, ' \
                   'Ship: 295 x=6 y=-2, Ship: 88 x=-2 y=8, Ship: 218 x=4 y=-4, Ship: 43 x=-10 y=3, ' \
                   'Ship: 104 x=-11 y=7, Ship: 190 x=0 y=0, Ship: 202 x=-11 y=-12, Ship: 52 x=11 y=-8, ' \
                   'Ship: 159 x=7 y=0, Ship: 277 x=-6 y=-12, Ship: 161 x=-12 y=-1, Ship: 234 x=3 y=-14, ' \
                   'Ship: 107 x=-1 y=3, Ship: 286 x=-12 y=-7, Ship: 51 x=-1 y=3, Ship: 236 x=11 y=11, ' \
                   'Ship: 181 x=-10 y=-12, Ship: 282 x=7 y=-3, Ship: 50 x=7 y=-3, Ship: 196 x=-12 y=-5, ' \
                   'Ship: 6 x=9 y=-11, Ship: 146 x=-12 y=-4, Ship: 280 x=-12 y=-4, Ship: 194 x=-12 y=0, ' \
                   'Ship: 244 x=10 y=-3, Ship: 263 x=-12 y=-11, Ship: 56 x=-12 y=-11, Ship: 16 x=12 y=10, ' \
                   'Ship: 67 x=9 y=-3, Ship: 79 x=2 y=-14, Ship: 262 x=9 y=-2, Ship: 170 x=-5 y=11, ' \
                   'Ship: 65 x=-4 y=-12, Ship: 220 x=12 y=6, Ship: 265 x=11 y=-1, Ship: 94 x=12 y=6, ' \
                   'Ship: 42 x=-4 y=-3, Ship: 23 x=4 y=3, Ship: 233 x=4 y=3, Ship: 153 x=1 y=-14, ' \
                   'Ship: 20 x=1 y=-14, Ship: 40 x=0 y=-14, Ship: 192 x=8 y=-5, Ship: 62 x=-13 y=-10, ' \
                   'Ship: 105 x=-10 y=4, Ship: 268 x=-10 y=4, Ship: 121 x=-4 y=11, Ship: 49 x=1 y=1, ' \
                   'Ship: 171 x=12 y=-9, Ship: 299 x=11 y=-3, Ship: 215 x=8 y=-13, Ship: 208 x=3 y=-15, ' \
                   'Ship: 167 x=-13 y=-7, Ship: 303 x=-13 y=-7, Ship: 132 x=-1 y=4, Ship: 111 x=3 y=5, ' \
                   'Ship: 258 x=0 y=-2, Ship: 172 x=12 y=4, Ship: 207 x=10 y=-13, Ship: 122 x=1 y=-2, ' \
                   'Ship: 27 x=-13 y=-2, Ship: 69 x=-13 y=-2, Ship: 58 x=2 y=1, Ship: 106 x=2 y=-15, ' \
                   'Ship: 264 x=-12 y=9, Ship: 142 x=-13 y=-3, Ship: 61 x=-13 y=-3, Ship: 275 x=-10 y=-13, ' \
                   'Ship: 165 x=2 y=2, Ship: 138 x=-6 y=-13, Ship: 261 x=9 y=12, Ship: 198 x=4 y=8, ' \
                   'Ship: 328 x=7 y=-14, Ship: 292 x=-11 y=12, Ship: 11 x=-5 y=-14, Ship: 15 x=-12 y=8, ' \
                   'Ship: 203 x=0 y=-15, Ship: 31 x=1 y=-1, Ship: 288 x=-6 y=-14, Ship: 178 x=13 y=9, ' \
                   'Ship: 101 x=-14 y=0, Ship: 318 x=-4 y=12, Ship: 274 x=2 y=6, Ship: 32 x=4 y=-2, ' \
                   'Ship: 76 x=4 y=-2, Ship: 260 x=8 y=12, Ship: 77 x=-9 y=12, Ship: 108 x=-5 y=12, ' \
                   'Ship: 155 x=-5 y=12, Ship: 141 x=13 y=-9, Ship: 302 x=13 y=7, Ship: 169 x=0 y=3, ' \
                   'Ship: 125 x=3 y=3, Ship: 267 x=-10 y=-14, Ship: 4 x=-14 y=-1, Ship: 12 x=0 y=5, ' \
                   'Ship: 205 x=0 y=5, Ship: 329 x=13 y=3, Ship: 253 x=13 y=-8, Ship: 83 x=3 y=4, ' \
                   'Ship: 3 x=-9 y=-14, Ship: 164 x=13 y=4, Ship: 17 x=-14 y=2, Ship: 118 x=-2 y=12, ' \
                   'Ship: 248 x=-2 y=12, Ship: 60 x=-2 y=12, Ship: 216 x=15 y=15, Ship: 187 x=-14 y=-2, ' \
                   'Ship: 294 x=12 y=-2, Ship: 149 x=14 y=10, Ship: 193 x=-5 y=-15, Ship: 64 x=-5 y=-15, ' \
                   'Ship: 25 x=12 y=-3, Ship: 290 x=-14 y=-3, Ship: 109 x=-14 y=-12, Ship: 300 x=0 y=8, ' \
                   'Ship: 139 x=-13 y=13, Ship: 145 x=13 y=-10, Ship: 225 x=13 y=-10, Ship: 87 x=9 y=-14, ' \
                   'Ship: 177 x=-13 y=7, Ship: 281 x=3 y=1, Ship: 276 x=-14 y=-8, Ship: 91 x=12 y=14, ' \
                   'Ship: 147 x=-7 y=-14, Ship: 217 x=-3 y=-15, Ship: 37 x=15 y=13, Ship: 19 x=1 y=3, ' \
                   'Ship: 130 x=8 y=-15, Ship: 116 x=-15 y=15, Ship: 66 x=14 y=14, Ship: 59 x=13 y=-1, ' \
                   'Ship: 285 x=9 y=13, Ship: 158 x=-13 y=14, Ship: 230 x=-15 y=0, Ship: 103 x=-15 y=14, ' \
                   'Ship: 131 x=14 y=15, Ship: 156 x=7 y=12, Ship: 54 x=7 y=12, Ship: 289 x=-9 y=13, ' \
                   'Ship: 148 x=1 y=5, Ship: 68 x=3 y=9, Ship: 293 x=15 y=-15, Ship: 219 x=-10 y=-15, ' \
                   'Ship: 327 x=14 y=9, Ship: 212 x=-1 y=12, Ship: 80 x=-14 y=-13, Ship: 188 x=-6 y=14, ' \
                   'Ship: 254 x=-15 y=-1, Ship: 269 x=-15 y=11, Ship: 279 x=14 y=0, Ship: 298 x=-15 y=11, ' \
                   'Ship: 168 x=-13 y=15, Ship: 160 x=12 y=-6, Ship: 316 x=12 y=-6, Ship: 246 x=-15 y=-4, ' \
                   'Ship: 151 x=-15 y=-14, Ship: 152 x=14 y=3, Ship: 47 x=-14 y=8, Ship: 270 x=12 y=15, ' \
                   'Ship: 163 x=15 y=-13, Ship: 209 x=14 y=-10, Ship: 126 x=-14 y=7, Ship: 98 x=14 y=4, ' \
                   'Ship: 150 x=14 y=-1, Ship: 9 x=-15 y=-7, Ship: 222 x=15 y=-12, Ship: 325 x=6 y=13, ' \
                   'Ship: 133 x=15 y=1, Ship: 82 x=14 y=8, Ship: 315 x=13 y=-4, Ship: 200 x=3 y=11, ' \
                   'Ship: 143 x=-4 y=15, Ship: 257 x=-15 y=5, Ship: 179 x=-9 y=15, Ship: 63 x=13 y=-5, ' \
                   'Ship: 55 x=15 y=-1, Ship: 224 x=-10 y=15, Ship: 71 x=15 y=6, Ship: 256 x=4 y=12, ' \
                   'Ship: 28 x=15 y=4, Ship: 99 x=15 y=4, Ship: 8 x=15 y=5, Ship: 44 x=-15 y=4, ' \
                   'Ship: 314 x=15 y=-2, Ship: 176 x=14 y=-4, Ship: 242 x=2 y=12, Ship: 29 x=5 y=14, ' \
                   'Ship: 36 x=14 y=-6, Ship: 309 x=7 y=15, Ship: 228 x=3 y=13, Ship: 78 x=3 y=13, ' \
                   'Ship: 95 x=0 y=15, Ship: 283 x=1 y=15]'
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
