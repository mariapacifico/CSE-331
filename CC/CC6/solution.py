# CC6 FS 2021
# Jacob Caurdy, Alexander Woodring, Jordyn Rosario
# Problem: Create a deepcopy of a graph given a node of the graph
# Return: Copy of the node given which points to a copy of the rest
# of the graph (through adjacency's)

from typing import TypeVar, List

T = TypeVar('T')
NecklaceNode = TypeVar('NecklaceNode')  # NecklaceNode Class Instance


class NecklaceNode:
    """ Class representing a Necklace Bead which can be connected with other
    Nodes to form a necklace"""

    __slots__ = ['adj', 'value']

    def __init__(self, val: T, adj: List[T] = [], adj_V: List[NecklaceNode] = None) -> None:
        """
        DO NOT MODIFY
        Initializes a NecklaceNode
        :param val: Value of the NecklaceNode
        :param adj: List of adjacent necklace nodes
        """
        self.value = val
        if adj:
            # adjacency list (list of neighbor vertices)
            self.adj = [NecklaceNode(value) for value in adj]
        elif adj_V:
            self.adj = adj_V
        else:
            self.adj = []

    def __repr__(self) -> str:
        """
        DO NOT MODIFY
        :return: string representing NecklaceNode object
        """
        if self.adj:
            lst = [f"{n.value}" for n in self.adj]
        else:
            lst = []

        return f"<val: '{self.value}'" + ", Adjacencies: " + ",".join(lst) + ">"

    def __str__(self) -> str:
        """
        DO NOT MODIFY
        :return: string representing NecklaceNode object
        """
        return repr(self)

def dfs(start, discovered):
    """
    Creates a copy of each NecklaceNode in a graph
    :param start: The starting node of graph
    :param discovered: Dictionary where the key is the original
    node and the value is the copied node
    :return: The copy of start
    """
    # discovered[original] = copy
    if start not in discovered:
        # create a copy
        # .adj will be []
        new_start = NecklaceNode(start.value)
        discovered[start] = new_start

        # add adj vertices
        # have to create a copy for each of those as well, go through dfs
        for vertices in start.adj:
            new_start.adj.append(dfs(vertices, discovered)) # recurrence
        # return copy
        return new_start
    # already in the dict, return copy
    return discovered[start]

def ReplicateNecklace(start: NecklaceNode) -> NecklaceNode:
    """
    Given a starting bead in a necklace, return a duplicate of the necklace bead
    :param start: A NecklaceNode object that's the starting point of the necklace
    that we will replicate
    :return: A duplicate NecklaceNode that is the starting point of the new necklace,
    which should be an exact replica
    of the original.
    """
    # call traversal and return copy of start
    return dfs(start, discovered={})
