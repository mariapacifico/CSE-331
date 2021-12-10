"""
CC7
Name: Maria Pacifico
"""

from typing import Dict, List

class PerkGraph:
    """
    DO NOT MODIFY
    Implementation of a perk tree
    """
    def __init__(self):
        """
        DO NOT MODIFY
        Initializes a PerkGraph
        :return: None
        """
        self.graph = {}  # Dictionary containing adjacency List
        self.vertices = set()  # Set of vertices

    def add_edge(self, first, second) -> None:
        """
        DO NOT MODIFY
        Adds an edge to a PerkGraph
        :param first: First vertex
        :param second: Second vertex
        :return: None
        """
        if first not in self.vertices:
            self.vertices.add(first)
            self.graph[first] = []
        if second not in self.vertices:
            self.vertices.add(second)
            self.graph[second] = []
        self.graph[first].append(second)

    # def perk_organizer(self, vertex: str, visited: Dict[str, bool], stack: List[str], target: str) -> bool:
    #     """
    #     ** OPTIONAL UTILITY FUNCTION **
    #     REPLACE
    #     Be sure to include :param: and :return: fields!
    #     See CC1 or Project 1 for examples of proper docstrings.
    #     """
    #     pass

    def perk_traversal(self, target: str, points: int = float("inf")) -> List[str]:
        """
        A function that will show the perks you need before getting the target perk
        and whether you have enough perk points to get it
        :param target: The target perk
        :param points: The number of perk points available
        :return: The list of perks that will get you to target if there are enough
        perk points otherwise, return an empty list
        """
        # create a dictionary with the outgoing vertices
        outgoing = {}
        for key in self.graph:
            if self.graph[key]:
                value = self.graph[key][0]
                outgoing[value] = [key]

        final = []

        # find the first value, put in stack
        for value in self.graph:
            # first value in graph
            if value not in outgoing:
                final.append(value)

        if final == []:
            return final

        # add the rest of values to the list
        # initialize new value
        new_value = final[-1]
        while new_value != target:
            new_value = final[-1]

            # stop adding to the list, end loop
            if new_value == target:
                break

            try:
                # next value in graph
                final.append(self.graph[new_value][0])
            # reached the last value of the graph, end loop
            except IndexError:
                break

        # length needs to be less than the number of points
        # otherwise, return an empty list
        if len(final) <= points:
            return final

        return []
