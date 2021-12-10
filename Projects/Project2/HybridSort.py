"""
Your Name:
Project 2 - Hybrid Sorting
CSE 331 Fall 2021
Professor Sebnem Onsay
"""
import random
from typing import TypeVar, List, Callable, Tuple, Dict

T = TypeVar("T")  # represents generic type

def merge(data1, data2, data, comparator):
    """
    Merges two sub arrays
    Source in README
    :param data1: Sub array on the left
    :param data2: Sub array on the right
    :param data: Final array that's being sorted in place
    :return: Number of inversions while merging
    """
    inversion = 0
    i = j = 0

    while i + j < len(data):
        # no inversion
        if j == len(data2) or (i < len(data1) and comparator(data1[i], data2[j])):
            data[i + j] = data1[i]
            i += 1

        # inversion occurs
        else:
            inversion += len(data1) - i
            data[i + j] = data2[j]
            j += 1

    return inversion


def merge_sort(data: List[T], threshold: int = 0,
               comparator: Callable[[T, T], bool] = lambda x, y: x <= y) -> int:
    """
    Sorts an array based on a comparator with a divide and conquer method
    Source in README
    :param data: Array that needs to be sorted
    :param threshold: After this threshold, will complete insertion_sort. Set as 0
    :param comparator: How the array is sorted. Set as ordering in ascending order
    :return: Number of iterations that occurs while sorting
    """
    inversion = 0

    length = len(data)

    # when size reaches threshold, insertion sort
    if length == threshold:
        insertion_sort(data, comparator)

    # base case
    if length < 2:
        return inversion

    # divides array in half
    mid = length // 2
    data1 = data[0:mid]
    data2 = data[mid:length]

    # variables used to increment inversion
    # recursive step
    num1 = merge_sort(data1, threshold, comparator)
    num2 = merge_sort(data2, threshold, comparator)

    inversion += merge(data1, data2, data, comparator)

    # after first recursive calls are done add to inversion
    if num1 is not None:
        inversion += num1

    # after second recursive calls are done add to inversion
    if num2 is not None:
        inversion += num2

    return inversion


def insertion_sort(data: List[T], comparator: Callable[[T, T], bool] = lambda x, y: x <= y) -> None:
    """
    Sorts an array based on comparater by building the array one element at a time
    Source in README
    :param data: Array that needs to be sorted
    :param comparator: How the array will be sorted
    """
    # iterate through data
    for i in range(1, len(data)):
        j = i
        # insert data[i] into sorted part of array
        # stopping once data[i] is in correct position
        while j > 0 and comparator(data[j], data[j-1]):
            # swap
            temp = data[j]
            data[j] = data[j-1]
            data[j-1] = temp
            j = j -1


def hybrid_sort(data: List[T], threshold: int,
                comparator: Callable[[T, T], bool] = lambda x, y: x <= y) -> None:
    """
    Wrapper function to call merge_sort
    :param data: Array that needs to be sorted
    :param threshold: At which point the list should be sorted using insertion sort
    :param comparator: How the array will be sorted
    """
    merge_sort(data, threshold, comparator)

def inversions_count(data: List[T]) -> int:
    """
    Wrapper function the gets the number of inverisons within an array
    :param data: Unsorted array
    :return: Number of inversions it takes to sort array in ascending order
    """
    # Should call merge_sort() with no threshold, and the default comparator.
    copy_data = data.copy()
    return merge_sort(copy_data)

def reverse_sort(data: List[T], threshold: int) -> None:
    """
    Wrapper function that sorts data in reverse order (descending)
    :param data: Array that needs to be sorted
    :param threshold: When the merge_sort would switch to insertion sort
    """
    merge_sort(data, threshold, comparator=lambda x, y: x >= y)

# forward reference
Ship = TypeVar('Ship')

# DO NOT MODIFY THIS CLASS
class Ship:
    """
    A class representation of a ship
    """

    __slots__ = ['name', 'x', 'y']

    def __init__(self, name: str, x: int, y: int) -> None:
        """
        Constructs a ship object
        :param name: name of the ship
        :param x: x coordinate of the ship
        :param y: y coordinate of the ship
        """
        self.x, self.y = x, y
        self.name = name

    def __str__(self):
        """
        :return: string representation of the ship
        """
        return "Ship: " + self.name + " x=" + str(self.x) + " y=" + str(self.y)

    __repr__ = __str__

    def __eq__(self, other):
        """
        :return: bool if two ships are equivalent
        """
        return self.x == other.x and self.y == other.y and self.name == other.name

    def __hash__(self):
        """
        Allows Ship to be used as a key in a dictionary (pretty cool, right?)
        :return: _hash of string representation of the ship
        """
        return hash(str(self))

    def euclidean_distance(self, other: Ship) -> float:
        """
        returns the euclidean distance between `self` and `other`
        :return: float
        """
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** .5

    def taxicab_distance(self, other: Ship) -> float:
        """
        returns the taxicab distance between `self` and `other`
        :return: float
        """
        return abs(self.x - other.x) + abs(self.y - other.y)


# MODIFY BELOW
def navigation_test(ships: Dict[Ship, List[Ship]], euclidean: bool = True) -> List[Ship]:
    """
    This function ranks ships based on how many mistakes each ship made
    :param ships: Contains a list for each ship, in order which they were percieved
    :param euclidean: Determines which metric to use to determine the number of mistakes
    :return: A list in order in ascending order based on many mistakes each made
    """
    mistake_list = []

    # iterate through each list based on each ship
    for key in ships:
        # the list of ships
        value = ships[key]

        # get number of mistakes from merge_sort
        mistake = 0
        # comparater depends on whether we're using euclidean or taxicab
        if euclidean:
             mistake = merge_sort(value, 0,
                                  comparator=lambda x, y:
                                  x.euclidean_distance(key) <= y.euclidean_distance(key))

        else:
            mistake = merge_sort(value, 0,
                                 comparator=lambda x, y:
                                 x.taxicab_distance(key) <= y.taxicab_distance(key))

        # put number of inversions and the ship into a list
        mistake_list.append((mistake, key))

    # sort list based on mistakes
    # if they're the same, sort based on name of ship
    merge_sort(mistake_list, threshold=0,
               comparator=lambda x, y: x[1].name <= y[1].name if x[0] == y[0] else x[0] <= y[0])

    # put into final list
    final_list = []
    for i in mistake_list:
        final_list.append(i[1])

    return final_list
