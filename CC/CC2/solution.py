"""
CC2
Name:
"""

from typing import List


def solar_power(heights: List[int]) -> int:
    """
    Calculates the maximum area of a given list of heights of a bar graph.
    :param heights: A list of heights of a bar graph
    :return: Maximum area of bar graph
    """
    #To find the area of a histogram, go through each rectangle, determine how much rectangles
    #on the left are greater than or equal to that value and how many rectangles on the right (r)
    #are greater than or equal to that value and find the distance between the right value and the left value
    #and multiply by that value.

    stack = []

    #index will represent number of rectangles to the right
    index = 0

    #initialize a value for the area
    max_area = 0

    while index < len(heights):

        #When there's nothing in the stack, we need to input a number to compare to the next values

        #If the value to the right of the value in the stack is greater than or equal to, then that
        #value will be included in the right side calculation so we increase the index.
        #Also adding that index orginally to the stack so that we can also calculate the area at that value.
        if stack == [] or heights[stack[-1]] <= heights[index]:
            stack.append(index)
            index += 1

        #The next value is not apart of the right
        else:
            #Get the index of the value
            top = stack.pop()

            #When the stack is empty, that means there are no values to the left at the index
            #of the stack that are less than that value, meeaning the length is equivalent to index
            if stack == []:
                area = heights[top] * index

            #index --> how many heights to the right
            #stack[-1] - 1--> how many heights to the left
            #index - stack[-1] - 1 = distance or length of the rectangle
            else:
                area = heights[top] * (index - stack[-1] - 1)

            #Replace the max area if necessary
            if max_area < area or max_area == 0:
                max_area = area

    #When there are still values within the stack, the heights to the right is equal to
    #len(heights), which is represented in index after the first while loop.
    #Iterate through the stack to get remainder areas, compare to the max_area
    while stack:
        top = stack.pop()
        if stack == []:
            area = heights[top] * (index)
        else:
            area = heights[top] * (index - stack[-1] - 1)
        if max_area < area or max_area == 0:
            max_area = area

    return max_area
