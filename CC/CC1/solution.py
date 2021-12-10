"""
CC1 Student Submission
Name: Maria Pacifico
"""

from typing import List, Tuple

def distance_formula(x_1: int, y_1: int, x_2: int, y_2: int) -> float:
    """
    Calculates the distance formula: square root((x2-x1)^2 + (y2-y1)^2)
    :param x1: x value from first point
    :param y1: y value from  first point
    :param x2: x value from second point
    :param y2: y value from second point
    :return: The distance between the two points
    """
    return ((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2) ** (1/2)

def farmer_fencing(points: List[Tuple[int, int]]) -> int:
    """
    Calculates the minimum perimeter required to form a rectangle using the
    given points.
    :param points: The fence posts available to use.
    :return: The minimum perimeter of the fence.
    """
    #Not enough points to make a rectangle
    if len(points) < 4:
        return 0

    #Turn points into a set, easier to search
    set_points = set(points)

    #Final perimeter, set to zero inititally
    perimeter = 0

    for i in points:
        for j in points:
            #Find the corners of the rectangle by finding a diagonal
            #Cannot have the same x and y values
            # [0] = x values
            # [1] = y values
            if i[0] != j[0] and i[1] != j[1]:

                #Find the other corners
                #Has to have the same x and y values
                if (i[0], j[1]) in set_points and (j[0], i[1]) in set_points:

                    #Perimeter equation = 2 * length + 2 * width
                    calc_perimeter = 2 * distance_formula(i[0], i[1], j[0], i[1]) + \
                                     2 * distance_formula(i[0], i[1], i[0], j[1])

                    #Replace with the smallest value
                    if calc_perimeter < perimeter or perimeter == 0:
                        perimeter = calc_perimeter

    return perimeter
