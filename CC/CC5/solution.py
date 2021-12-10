"""
CC5 Student Submission
Name: Maria Pacifico
"""
import heapq
from typing import List, Tuple, Callable

# Merge & Merge sort from project 2
def merge(data1, data2, data, comparator) -> None:
    """
    Merges two sub arrays
    :param data1: Sub array on the left
    :param data2: Sub array on the right
    :param data: Final array that's being sorted in place
    """
    i = j = 0

    while i + j < len(data):
        # intersection doesn't occur
        if j == len(data2) or (i < len(data1) and comparator(data1[i], data2[j])):
            data[i + j] = data1[i]
            i += 1
        # intersection occurs
        else:
            data[i + j] = data2[j]
            j += 1

def merge_sort(data,
               comparator: Callable[[int, int], bool] = lambda x, y: x <= y) -> None:
    """
    Sorts an array based on a comparator with a divide and conquer method
    :param data: Array that needs to be sorted
    :param comparator: How the array is sorted. Set as ordering in ascending order
    """
    length = len(data)

    # base case
    if length < 2:
        return

    # divides array in half
    mid = length // 2
    data1 = data[0:mid]
    data2 = data[mid:length]

    # recursive step
    merge_sort(data1, comparator)
    merge_sort(data2,  comparator)

    # merge data
    merge(data1, data2, data, comparator)


def scooter_rentals(times: List[Tuple[int, int]]) -> int:
    """
    Calculates the number of scooter required at given times
    :param times: Interval of times when scooters are being used
    :return: The total number of scooter needed
    """
    # List is empty
    if len(times) == 0:
        return 0

    # One time in list, return 1
    if len(times) == 1:
        return 1

    # sort data based on starting time
    merge_sort(times, comparator=lambda x, y: x[0] <= y[0])

    # create heap with all end times
    heap = []

    # iterate through time list
    for i in range(0,len(times)):
        # if first element, add to heap
        if i == 0:
            heapq.heappush(heap, times[i][1])
        else:
            # get the end time of root of heap
            end_time = heap[0]

            # get the start time of current time
            start_time = times[i][0]

            # end <= start
            # true: delete min heap root, add current to heap
            if end_time <= start_time:
                heapq.heappop(heap)
                heapq.heappush(heap, times[i][1])

            # false: add to min heap
            else:
                heapq.heappush(heap, times[i][1])

    # return length of heap
    return len(heap)
