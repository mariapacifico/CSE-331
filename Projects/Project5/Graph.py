"""
Name: Maria Pacifico
CSE 331 FS21 (Onsay)
Project 5
"""
import copy
import heapq
import itertools
import math
import queue
import random
import time
import csv
from typing import TypeVar, Callable, Tuple, List, Set

import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

T = TypeVar('T')
Matrix = TypeVar('Matrix')  # Adjacency Matrix
Vertex = TypeVar('Vertex')  # Vertex Class Instance
Graph = TypeVar('Graph')    # Graph Class Instance


class Vertex:
    """ Class representing a Vertex object within a Graph """

    __slots__ = ['id', 'adj', 'visited', 'x', 'y']

    def __init__(self, idx: str, x: float = 0, y: float = 0) -> None:
        """
        DO NOT MODIFY
        Initializes a Vertex
        :param idx: A unique string identifier used for hashing the vertex
        :param x: The x coordinate of this vertex (used in a_star)
        :param y: The y coordinate of this vertex (used in a_star)
        """
        self.id = idx
        self.adj = {}             # dictionary {id : weight} of outgoing edges
        self.visited = False      # boolean flag used in search algorithms
        self.x, self.y = x, y     # coordinates for use in metric computations

    def __eq__(self, other: Vertex) -> bool:
        """
        DO NOT MODIFY
        Equality operator for Graph Vertex class
        :param other: vertex to compare
        """
        if self.id != other.id:
            return False
        elif self.visited != other.visited:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex visited flags not equal: self.visited={self.visited},"
                  f" other.visited={other.visited}")
            return False
        elif self.x != other.x:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex x coords not equal: self.x={self.x}, other.x={other.x}")
            return False
        elif self.y != other.y:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex y coords not equal: self.y={self.y}, other.y={other.y}")
            return False
        elif set(self.adj.items()) != set(other.adj.items()):
            diff = set(self.adj.items()).symmetric_difference(set(other.adj.items()))
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex adj dictionaries not equal:"
                  f" symmetric diff of adjacency (k,v) pairs = {str(diff)}")
            return False
        return True

    def __repr__(self) -> str:
        """
        DO NOT MODIFY
        :return: string representing Vertex object
        """
        lst = [f"<id: '{k}', weight: {v}>" for k, v in self.adj.items()]

        return f"<id: '{self.id}'" + ", Adjacencies: " + "".join(lst) + ">"

    def __str__(self) -> str:
        """
        DO NOT MODIFY
        :return: string representing Vertex object
        """
        return repr(self)

    def __hash__(self) -> int:
        """
        DO NOT MODIFY
        Hashes Vertex into a set; used in unit tests
        :return: hash value of Vertex
        """
        return hash(self.id)

#============== Modify Vertex Methods Below ==============#

    def degree(self) -> int:
        """
        Returns the number of outgoing edges from this vertex
        :return: The degree of the vertex
        """
        return len(self.adj)

    def get_edges(self) -> Set[Tuple[str, float]]:
        """
        Returns a set of tuples representing outgoing edges from this vertex
        :return: The set of outgoing edges
        """
        # iterate through adj list
        # add to set
        final = set()
        for i in self.adj:
            # (key, value)
            final.add((i, self.adj[i]))
        return final

    def euclidean_distance(self, other: Vertex) -> float:
        """
        Returns the euclidean distance between this vertex and another vertex
        :param other: The other vertex (x2, y2)
        :return: The euclidean distance
        """
        # d = ((x2-x1)^2 + (y2-y1)^2)^(1/2)
        return ((other.x - self.x)**2 + (other.y - self.y)**2)**0.5

    def taxicab_distance(self, other: Vertex) -> float:
        """
        Returns the taxicab distance between this vertex and another vertex
        :param other: The other vertex (x2, y2)
        :return: The taxicab distance
        """
        # d = |x2-x1| + |y2-y1|
        return abs(other.x - self.x) + abs(other.y - self.y)


class Graph:
    """ Class implementing the Graph ADT using an Adjacency Map structure """

    __slots__ = ['size', 'vertices', 'plot_show', 'plot_delay']

    def __init__(self, plt_show: bool = False, matrix: Matrix = None, csv: str = "") -> None:
        """
        DO NOT MODIFY
        Instantiates a Graph class instance
        :param: plt_show : if true, render plot when plot() is called; else, ignore calls to plot()
        :param: matrix : optional matrix parameter used for fast construction
        :param: csv : optional filepath to a csv containing a matrix
        """
        matrix = matrix if matrix else np.loadtxt(csv, delimiter=',', dtype=str).tolist() if csv else None
        self.size = 0
        self.vertices = {}

        self.plot_show = plt_show
        self.plot_delay = 0.2

        if matrix is not None:
            for i in range(1, len(matrix)):
                for j in range(1, len(matrix)):
                    if matrix[i][j] == "None" or matrix[i][j] == "":
                        matrix[i][j] = None
                    else:
                        matrix[i][j] = float(matrix[i][j])
            self.matrix2graph(matrix)


    def __eq__(self, other: Graph) -> bool:
        """
        DO NOT MODIFY
        Overloads equality operator for Graph class
        :param other: graph to compare
        """
        if self.size != other.size or len(self.vertices) != len(other.vertices):
            print(f"Graph size not equal: self.size={self.size}, other.size={other.size}")
            return False
        else:
            for vertex_id, vertex in self.vertices.items():
                other_vertex = other.get_vertex(vertex_id)
                if other_vertex is None:
                    print(f"Vertices not equal: '{vertex_id}' not in other graph")
                    return False

                adj_set = set(vertex.adj.items())
                other_adj_set = set(other_vertex.adj.items())

                if not adj_set == other_adj_set:
                    print(f"Vertices not equal: adjacencies of '{vertex_id}' not equal")
                    print(f"Adjacency symmetric difference = "
                          f"{str(adj_set.symmetric_difference(other_adj_set))}")
                    return False
        return True

    def __repr__(self) -> str:
        """
        DO NOT MODIFY
        :return: String representation of graph for debugging
        """
        return "Size: " + str(self.size) + ", Vertices: " + str(list(self.vertices.items()))

    def __str__(self) -> str:
        """
        DO NOT MODFIY
        :return: String representation of graph for debugging
        """
        return repr(self)

    def plot(self) -> None:
        """
        DO NOT MODIFY
        Creates a plot a visual representation of the graph using matplotlib
        """
        if self.plot_show:

            # if no x, y coords are specified, place vertices on the unit circle
            for i, vertex in enumerate(self.get_vertices()):
                if vertex.x == 0 and vertex.y == 0:
                    vertex.x = math.cos(i * 2 * math.pi / self.size)
                    vertex.y = math.sin(i * 2 * math.pi / self.size)

            # show edges
            num_edges = len(self.get_edges())
            max_weight = max([edge[2] for edge in self.get_edges()]) if num_edges > 0 else 0
            colormap = cm.get_cmap('cool')
            for i, edge in enumerate(self.get_edges()):
                origin = self.get_vertex(edge[0])
                destination = self.get_vertex(edge[1])
                weight = edge[2]

                # plot edge
                arrow = patches.FancyArrowPatch((origin.x, origin.y),
                                                (destination.x, destination.y),
                                                connectionstyle="arc3,rad=.2",
                                                color=colormap(weight / max_weight),
                                                zorder=0,
                                                **dict(arrowstyle="Simple,tail_width=0.5,"
                                                                  "head_width=8,head_length=8"))
                plt.gca().add_patch(arrow)

                # label edge
                plt.text(x=(origin.x + destination.x) / 2 - (origin.x - destination.x) / 10,
                         y=(origin.y + destination.y) / 2 - (origin.y - destination.y) / 10,
                         s=weight, color=colormap(weight / max_weight))

            # show vertices
            x = np.array([vertex.x for vertex in self.get_vertices()])
            y = np.array([vertex.y for vertex in self.get_vertices()])
            labels = np.array([vertex.id for vertex in self.get_vertices()])
            colors = np.array(
                ['yellow' if vertex.visited else 'black' for vertex in self.get_vertices()])
            plt.scatter(x, y, s=40, c=colors, zorder=1)

            # plot labels
            for j, _ in enumerate(x):
                plt.text(x[j] - 0.03*max(x), y[j] - 0.03*max(y), labels[j])

            # show plot
            plt.show()
            # delay execution to enable animation
            time.sleep(self.plot_delay)

    def add_to_graph(self, start_id: str, dest_id: str = None, weight: float = 0) -> None:
        """
        Adds to graph: creates start vertex if necessary,
        an edge if specified,
        and a destination vertex if necessary to create said edge
        If edge already exists, update the weight.
        :param start_id: unique string id of starting vertex
        :param dest_id: unique string id of ending vertex
        :param weight: weight associated with edge from start -> dest
        :return: None
        """
        if self.vertices.get(start_id) is None:
            self.vertices[start_id] = Vertex(start_id)
            self.size += 1
        if dest_id is not None:
            if self.vertices.get(dest_id) is None:
                self.vertices[dest_id] = Vertex(dest_id)
                self.size += 1
            self.vertices.get(start_id).adj[dest_id] = weight

    def matrix2graph(self, matrix: Matrix) -> None:
        """
        Given an adjacency matrix, construct a graph
        matrix[i][j] will be the weight of an edge between the vertex_ids
        stored at matrix[i][0] and matrix[0][j]
        Add all vertices referenced in the adjacency matrix, but only add an
        edge if matrix[i][j] is not None
        Guaranteed that matrix will be square
        If matrix is nonempty, matrix[0][0] will be None
        :param matrix: an n x n square matrix (list of lists) representing Graph as adjacency map
        :return: None
        """
        for i in range(1, len(matrix)):         # add all vertices to begin with
            self.add_to_graph(matrix[i][0])
        for i in range(1, len(matrix)):         # go back through and add all edges
            for j in range(1, len(matrix)):
                if matrix[i][j] is not None:
                    self.add_to_graph(matrix[i][0], matrix[j][0], matrix[i][j])

    def graph2matrix(self) -> Matrix:
        """
        given a graph, creates an adjacency matrix of the type described in "construct_from_matrix"
        :return: Matrix
        """
        matrix = [[None] + [v_id for v_id in self.vertices]]
        for v_id, outgoing in self.vertices.items():
            matrix.append([v_id] + [outgoing.adj.get(v) for v in self.vertices])
        return matrix if self.size else None

    def graph2csv(self, filepath: str) -> None:
        """
        given a (non-empty) graph, creates a csv file containing data necessary to reconstruct that graph
        :param filepath: location to save CSV
        :return: None
        """
        if self.size == 0:
            return

        with open(filepath, 'w+') as graph_csv:
            csv.writer(graph_csv, delimiter=',').writerows(self.graph2matrix())

#============== Modify Graph Methods Below ==============#

    def reset_vertices(self) -> None:
        """
        FILL OUT DOCSTRING
        """
        for i in self.vertices:
            self.vertices[i].visited = False

    def get_vertex(self, vertex_id: str) -> Vertex:
        """
        FILL OUT DOCSTRING
        """
        for i in self.vertices:
            if i == vertex_id:
                return self.vertices[i]
        return None

    def get_vertices(self) -> Set[Vertex]:
        """
        FILL OUT DOCSTRING
        """
        final = set()
        for i in self.vertices:
            final.add(self.vertices[i])
        return final

    def get_edge(self, start_id: str, dest_id: str) -> Tuple[str, str, float]:
        """
        FILL OUT DOCSTRING
        """
        # get the vertex at the start
        vertex = self.get_vertex(start_id)
        # if it doesn't exist
        if vertex is None:
            return None

        # go through the edges of the vertex
        edges = vertex.get_edges()
        for i in edges:
            # edge exists
            if i[0] == dest_id:
                # return: (start_id, dest_id, weight)
                return (start_id, i[0], i[1])

    def get_edges(self) -> Set[Tuple[str, str, float]]:
        """
        FILL OUT DOCSTRING
        """
        final = set()

        # get all the vertices, represent the start_id
        for start_id in self.vertices:
            # get all the edges within that vertex
            for edges in self.vertices[start_id].get_edges():
                # add to set
                # (start_id, dest_id, weight)
                final.add((start_id, edges[0], edges[1]))

        return final

    def bfs(self, start_id: str, target_id: str) -> Tuple[List[str], float]:
        """
        FILL OUT DOCSTRING
        """
        # CASE 1 and 2: start and target not in graph
        if start_id not in self.vertices or target_id not in self.vertices:
            return ([], 0)

        # Create a queue for BFS
        myQueue = queue.SimpleQueue()
        # Mark the source node as visited and enqueue it
        myQueue.put([start_id])

        final = []

        while not myQueue.empty():
            # Dequeue a vertex from queue
            vertex_list = myQueue.get()
            vertex = vertex_list[-1]

            # check if visited
            if self.vertices[vertex].visited == False:
                # go through adjacent list
                for i in self.vertices[vertex].adj:
                    path = list(vertex_list)
                    path.append(i)
                    myQueue.put(path)
                    if i == target_id:
                        total = 0
                        # find the totals
                        for j in range(1, len(path)):
                            total += self.vertices[path[j-1]].adj[path[j]]
                        return (path, total)
                # set as visited
                self.vertices[vertex].visited = True

        return ([], 0)

    def dfs(self, start_id: str, target_id: str) -> Tuple[List[str], float]:
        """
        Using a depth first search (dfs) approach to find a path
        :param start_id: Start of path
        :param target_id: End of path
        :return: A tuple with first element being the path and the second being the sum of weights
        """

        # CASE 1 and 2: start and target not in graph
        if start_id not in self.vertices or target_id not in self.vertices:
            return ([], 0)

        def dfs_inner(current_id: str, target_id: str,
                      path: List[str] = []) -> Tuple[List[str], float]:
            """
            Recursive function generating path using dfs
            :param current_id: Where the current path is
            :param target_id: Where to end path
            :param path: Current path taken
            :return: A tuple with first element being the path and the second being the sum of weights
            """
            final = ([], 0)

            if self.vertices[current_id].visited == False:
                self.vertices[current_id].visited = True

                # target found, return
                # base case
                if current_id == target_id:
                    path.append(current_id)
                    return (path, 0)

                for i in self.vertices[current_id].adj:
                    # recursive case
                    final = dfs_inner(i, target_id, path+[current_id])
                    if final != ([], 0):
                        #update path
                        path = final[0]
                        # find the totals
                        total = 0
                        for j in range(1, len(path)):
                            total += self.vertices[path[j - 1]].adj[path[j]]
                        return (path, total)

            # path not found
            return final

        return dfs_inner(start_id, target_id, path=[])

    def detect_cycle(self) -> bool:
        """
        Determines if there's a cycle within the graph
        :return: True if the graph contains a cycle, otherwise returns False
        """

        def inner_detect_cycle(start: str) -> bool:
            """
            Finds whether there's a cycles given a starting vertice
            :param start: Start of cycle
            :return: A boolean determining whether there was a cycle
            """
            # The node has been visited
            self.vertices[start].visited = True
            # go through all of the vertices
            for j in self.vertices[start].adj:
                # cycle found
                if self.vertices[j].visited == True:
                    return True
                # otherwise, go through the other adjacent vertices
                # if cycle found in recursion, return true
                if inner_detect_cycle(j):
                    return True
            # no cycle found, restart
            self.vertices[start].visited = False

        # check if each vertice has a cycle or not
        for i in self.vertices:
            if inner_detect_cycle(i):
                return True

        # no cycle found, return false
        return False

    def a_star(self, start_id: str, target_id: str,
               metric: Callable[[Vertex, Vertex], float]) -> Tuple[List[str], float]:
        """
        Find the shortest path using A* algorithm
        :param start_id: Starting point of path
        :param target_id: Ending point
        :param metric: How the heuristic will be calculated
        :return: A tuple containing the path and the sum of weights in the path
        """
        # create priority queue
        myQueue = AStarPriorityQueue()
        # add the start value
        myQueue.push(0, self.vertices[start_id])

        came_from = {start_id: None}
        cost = {start_id: 0}

        # iterate through queue
        while not myQueue.empty():
            # current[1] = vertices
            current = myQueue.pop()

            # found final, break
            if current[1].id == target_id:
                break

            for i in self.vertices[current[1].id].adj:
                # distance (weight) from current to node
                current_cost = self.vertices[current[1].id].adj[i] + cost[current[1].id]
                if i not in cost or current_cost < cost[i]:
                    cost[i] = current_cost
                    # f(x) = distance current to current node + distance from node to the final (heuristic)
                    current_cost += metric(self.vertices[i], self.vertices[target_id])
                    # update queue, or add if doesn't exist
                    try:
                        myQueue.update(current_cost, self.vertices[i])
                    except KeyError:
                        myQueue.push(current_cost, self.vertices[i])
                    # update where node came from
                    came_from[i] = current[1].id

        # get the path
        loop = came_from[target_id]
        final = [target_id]
        while loop != None:
            final.append(loop)
            loop = came_from[loop]
        # reverse
        final = final[::-1]

        return (final, cost[target_id])

class AStarPriorityQueue:
    """
    Priority Queue built upon heapq module with support for priority key updates
    Created by Andrew McDonald
    Inspired by https://docs.python.org/2/library/heapq.html
    """

    __slots__ = ['data', 'locator', 'counter']

    def __init__(self) -> None:
        """
        Construct an AStarPriorityQueue object
        """
        self.data = []                        # underlying data list of priority queue
        self.locator = {}                     # dictionary to locate vertices within priority queue
        self.counter = itertools.count()      # used to break ties in prioritization

    def __repr__(self) -> str:
        """
        Represent AStarPriorityQueue as a string
        :return: string representation of AStarPriorityQueue object
        """
        lst = [f"[{priority}, {vertex}], " if vertex is not None else "" for
               priority, count, vertex in self.data]
        return "".join(lst)[:-1]

    def __str__(self) -> str:
        """
        Represent AStarPriorityQueue as a string
        :return: string representation of AStarPriorityQueue object
        """
        return repr(self)

    def empty(self) -> bool:
        """
        Determine whether priority queue is empty
        :return: True if queue is empty, else false
        """
        return len(self.data) == 0

    def push(self, priority: float, vertex: Vertex) -> None:
        """
        Push a vertex onto the priority queue with a given priority
        :param priority: priority key upon which to order vertex
        :param vertex: Vertex object to be stored in the priority queue
        :return: None
        """
        # list is stored by reference, so updating will update all refs
        node = [priority, next(self.counter), vertex]
        self.locator[vertex.id] = node
        heapq.heappush(self.data, node)

    def pop(self) -> Tuple[float, Vertex]:
        """
        Remove and return the (priority, vertex) tuple with lowest priority key
        :return: (priority, vertex) tuple where priority is key,
        and vertex is Vertex object stored in priority queue
        """
        vertex = None
        while vertex is None:
            # keep popping until we have valid entry
            priority, count, vertex = heapq.heappop(self.data)
        del self.locator[vertex.id]            # remove from locator dict
        vertex.visited = True                  # indicate that this vertex was visited
        while len(self.data) > 0 and self.data[0][2] is None:
            heapq.heappop(self.data)          # delete trailing Nones
        return priority, vertex

    def update(self, new_priority: float, vertex: Vertex) -> None:
        """
        Update given Vertex object in the priority queue to have new priority
        :param new_priority: new priority on which to order vertex
        :param vertex: Vertex object for which priority is to be updated
        :return: None
        """
        node = self.locator.pop(vertex.id)      # delete from dictionary
        node[-1] = None                         # invalidate old node
        self.push(new_priority, vertex)         # push new node
