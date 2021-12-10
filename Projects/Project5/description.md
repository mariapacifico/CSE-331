<h1>Project 5: Graphs</h1>
<p><strong>Due: Thursday, December 2nd @ 8:00 pm</strong></p>
<p><em>This is not a team project, do not copy someone else&rsquo;s work.</em></p>
<p><em>Project based on contributions by Adam Kasumovic, Andrew Haas, Brooke Osterkamp</em></p>
<p>&nbsp;</p>
<h2>Assignment Overview</h2>
<p>Graphs are particularly useful data structures for modeling connections and relationships among objects. In fact, you've likely used an application that relies on graphical modeling today - a few examples include</p>
<ul>
<li><strong>Facebook / Twitter / Instagram</strong>
<ul>
<li>Users are modeled as vertices storing posts, photos, videos, etc.</li>
<li>Edges are modeled as "friendships," "followers," "likes," "favorites," etc.&nbsp;</li>
</ul>
</li>
<li><strong>Google / Apple / Bing Maps</strong>
<ul>
<li>Intersections, cities, and other points of interest are modeled by vertices</li>
<li>Road segments between intersections are modeled by weighted edges, where weights represent the relative speed/traffic of the road segment</li>
</ul>
</li>
</ul>
<p>You will be implementing a directed, weighted Graph ADT using the <strong>adjacency map</strong> design, in which a graph object consists of a map (ordered dictionary) of vertices, and each vertex holds its own map (ordered dictionary) of adjacent vertices, i.e. vertices to which that vertex is connected by an outgoing edge.</p>
<p>In some ways, this project also serves as a capstone to the course- in completing it one utilizes recursion, queues, two-dimensional arrays, hash maps, dynamic programming, and more. You may also notice that trees, linked lists, and even heaps are special cases of the general graph structure and that many graph algorithms can be applied to these earlier structures without modification. To highlight this inheritance, consider the inorder traversal algorithm we applied to AVL trees - really, it was nothing more than a graph depth-first search with a tendency to go left before right.</p>
<p>The goal of this project is to introduce the versatile and flexible nature of graphs, along with the operations and search algorithms that make them so useful.</p>
<p><img src="https://s3.amazonaws.com/mimirplatform.production/files/aea24196-db6a-46f0-9008-3ff19b7d29c7/got_graph.png" alt="got_graph.png" /></p>
<p>&nbsp;</p>
<h2>Assignment Notes</h2>
<ul>
<li>A plotting function is provided to help you visualize the progression of various search algorithms
<ul>
<li>Be sure to read the specs explaining&nbsp;<strong>plot()</strong></li>
<li>If you don't want to use it, just comment out the related import statements and <strong>plot()</strong> function</li>
</ul>
</li>
<li>Python allows representation of the value infinity using <strong>float('inf')</strong></li>
<li>No negative edge weights will ever be added to the graph
<ul>
<li>All edge weights are numeric values greater than or equal to zero</li>
</ul>
</li>
<li>Time complexities are specified in terms of <strong>V</strong> and <strong>E</strong>, where <strong>V</strong> represents the number of vertices in the graph and <strong>E</strong> represents the number of edges in a graph
<ul>
<li>Recall that E is bounded above by <strong>V</strong>^2; a graph has <strong>E</strong> = <strong>V</strong>^2 edges if and only if every vertex is connected to every other vertex</li>
</ul>
</li>
<li>Recall that&nbsp;<strong>list.insert(0, element)&nbsp;</strong>and&nbsp;<strong>list.pop(0)&nbsp;</strong>are both <em>O(N)</em> calls on a Python list
<ul>
<li>Recall that python's 'lists' are not lists in the more common sense of the word: linked lists. They are dynamically managed tuples, stored in memory as contiguous arrays of pointers to elements elsewhere in memory. This allows indexing into a 'list' in constant time. The downside of this, however, is that adding to a python 'list' at a specific index, <em>i, </em>requires shifting the pointer to every element past&nbsp;<em>i&nbsp;</em>by one in the underlying array: a linear operation.</li>
<li>Be careful when implementing&nbsp;<strong>bfs, dfs,</strong> and the Application Problem to ensure you do not break time complexity by popping or inserting from the front of a list when reconstructing a path</li>
<li>Instead of inserting into / popping from the front of the list, simply append to or pop from the end, then reverse the list <em>once</em> at the end
<ul>
<li>If you have N calls to <strong>list.insert(0, element)</strong>, that is <em>O(N^2)</em></li>
<li>If you instead have N calls to <strong>list.append(element)</strong>, followed by a single call to <strong>list.reverse()</strong>, that is <em>O(N)</em></li>
<li>Both methods will result in the same list being constructed, but the second is far more efficient</li>
</ul>
</li>
</ul>
</li>
</ul>
<h2>Assignment Specifications</h2>
<h3>class Vertex:&nbsp;</h3>
<p>Represents a vertex object, the building block of a graph.</p>
<p><strong><em>DO NOT MODIFY the following attributes/functions</em></strong></p>
<ul>
<li><strong>Attributes</strong>
<ul>
<li><strong>id: </strong>A string used to uniquely identify a vertex</li>
<li><strong>adj:&nbsp;</strong>A dictionary of type <strong>{other_id : number} </strong>which represents the connections of a vertex to other vertices; the existence of an entry with key <strong>other_i</strong><strong>d<em>&nbsp;</em></strong>indicates connection from this vertex to the vertex with id <strong>other_id </strong>by an edge with weight <strong>number</strong>
<ul>
<li>Note that as of Python 3.7, <a href="https://stackoverflow.com/a/57072435">insertion ordering</a> in normal dictionaries is guaranteed and ensures traversals will select the next neighbor to visit deterministically</li>
</ul>
</li>
<li><strong>visited:</strong> A boolean flag used in search algorithms to indicate that the vertex has been visited</li>
<li><strong>x:</strong> The x-position of a vertex (used in Application Problem) (defaults to zero)</li>
<li><strong>y:&nbsp;</strong>The y-position of a vertex (used in Application Problem) (defaults to zero)</li>
</ul>
</li>
<li><strong>__init__(self, idx: str, x: float=0, y: float=0) -&gt; None:</strong><br />
<ul>
<li>Constructs a Vertex object</li>
</ul>
</li>
<li><strong>__eq__(self, other: Vertex) -&gt; bool:</strong>
<ul>
<li>Compares this vertex for equality with another vertex</li>
</ul>
</li>
<li><strong>__repr__(self) -&gt; str:</strong>
<ul>
<li>Represents the vertex as a string for debugging</li>
</ul>
</li>
<li><strong>__str__(self) -&gt; str:</strong>
<ul>
<li>Represents the vertex as a string for debugging</li>
</ul>
</li>
<li><strong>__hash__(self) -&gt; int:</strong>
<ul>
<li>Allows the vertex to be hashed into a set; used in unit tests</li>
</ul>
</li>
</ul>
<p><strong><em>IMPLEMENT the following functions</em></strong></p>
<ul>
<li><strong>degree(self) -&gt; int:</strong>
<ul>
<li>Returns the number of outgoing edges from this vertex; i.e., the outgoing degree of this vertex</li>
<li><em>Time Complexity: O(1)</em></li>
<li><em>Space Complexity: O(1)</em></li>
</ul>
</li>
<li><strong>get_edges(self) -&gt; Set[Tuple[str, float]]:</strong>
<ul>
<li>Returns a <strong>set</strong> of tuples representing outgoing edges from this vertex</li>
<li>Edges are represented as tuples <strong>(other_id, weight)</strong><strong>&nbsp;</strong>where&nbsp;
<ul>
<li><strong>other_id</strong> is the unique string id of the destination vertex</li>
<li><strong>weight</strong> is the weight of the edge connecting this vertex to the other vertex</li>
</ul>
</li>
<li>Returns an empty set if this vertex has no outgoing edges</li>
<li><em>Time Complexity: O(degV)</em></li>
<li><em>Space Complexity: O(degV)</em></li>
</ul>
</li>
<li><strong>euclidean_distance(self, other: Vertex) -&gt; float:</strong>
<ul>
<li>Returns the <a href="http://rosalind.info/glossary/euclidean-distance/" target="_blank" rel="noopener noreferrer">euclidean distance</a> [based on two-dimensional coordinates] between this vertex and vertex <strong>other</strong></li>
<li>Used in Application problem</li>
<li><em>Time Complexity: O(1)</em></li>
<li><em>Space Complexity: O(1)</em></li>
</ul>
</li>
<li><strong>taxicab_distance(self, other: Vertex) -&gt; float:</strong>
<ul>
<li>Returns the <a href="https://en.wikipedia.org/wiki/Taxicab_geometry">taxicab distance</a> [based on two-dimensional coordinates] between this vertex and vertex <strong>other</strong></li>
<li>Used in Application problem</li>
<li><em>Time Complexity: O(1)</em></li>
<li><em>Space Complexity: O(1)</em></li>
</ul>
</li>
</ul>
<h3>class Graph:&nbsp;</h3>
<p>Represents a graph object</p>
<p><strong><em>DO NOT MODIFY the following attributes/functions</em></strong></p>
<ul>
<li><strong>Attributes</strong>
<ul>
<li><strong>size:&nbsp;</strong>The number of vertices in the graph</li>
<li><strong>vertices:</strong> A dictionary of type <strong>{id : Vertex}</strong> storing the vertices of the graph, where&nbsp;<strong>id</strong> represents the unique string id of a <strong>Vertex&nbsp;</strong>object
<ul>
<li>Note that as of Python 3.7, <a href="https://stackoverflow.com/a/57072435">insertion ordering</a> in normal dictionaries is guaranteed and ensures <strong style="background-color: transparent;">get_edges(self) </strong><span style="background-color: transparent;">and </span><strong style="background-color: transparent;">get_vertices(self)</strong><span style="background-color: transparent;"> will return deterministically ordered lists</span></li>
</ul>
</li>
<li><strong>plot_show</strong>: If true, calls to&nbsp;<strong>plot()</strong> display a rendering of the graph in matplotlib; if false, all calls to&nbsp;<strong>plot()</strong> are ignored (see&nbsp;<strong>plot()</strong> below)</li>
<li><strong>plot_delay</strong>: Length of delay in&nbsp;<strong>plot()&nbsp; </strong>(see&nbsp;<strong>plot()</strong> below)</li>
</ul>
</li>
<li><strong>__init__(self, plt_show: bool=False) -&gt; None:</strong><br />
<ul>
<li>Constructs a Graph object</li>
<li>Sets <strong>self.plot_show</strong> to False by default</li>
</ul>
</li>
<li><strong>__eq__(self, other: Graph) -&gt; bool:</strong>
<ul>
<li>Compares this graph for equality with another graph</li>
</ul>
</li>
<li><strong>__repr__(self) -&gt; str:</strong>
<ul>
<li>Represents the graph as a string for debugging</li>
</ul>
</li>
<li><strong>__str__(self) -&gt; str:</strong>
<ul>
<li>Represents the graph as a string for debugging</li>
</ul>
</li>
<li><strong>add_to_graph(self, start_id: str, dest_id: str=None, weight: float=0) -&gt; float:</strong>
<ul>
<li>Adds a vertex / vertices / edge to the graph
<ul>
<li>Adds a vertex with id&nbsp;<strong>start_id</strong> to the graph if no such vertex exists</li>
<li>Adds a vertex with id <strong>dest_id </strong>to the graph if no such vertex exists and&nbsp;<strong>dest_id</strong> is not None</li>
<li>Adds an edge with weight&nbsp;<strong>weight</strong> if&nbsp;<strong>dest_id</strong> is not None</li>
</ul>
</li>
<li>If a vertex with id <strong>start_id</strong> or <strong>dest_id&nbsp;</strong>already exists in the graph, this function will not overwrite that vertex with a new one</li>
<li>If an edge already exists from vertex with id&nbsp;<strong>start_id</strong> to vertex with id <strong>dest_id</strong>, this function will overwrite the weight of that edge</li>
</ul>
</li>
<li><strong>matrix2graph(self, matrix: Matrix) -&gt; None:</strong>
<ul>
<li>Constructs a graph from a given adjacency matrix representation</li>
<li><strong>matrix </strong>is guaranteed to be a square 2D list (i.e. list of lists where # rows = # columns), of size <strong>[V+1] </strong>x <strong>[V+1]</strong>
<ul>
<li><strong>matrix[0][0]&nbsp;</strong>is None</li>
<li>The first row and first column of&nbsp;<strong>matrix</strong> hold string ids of vertices to be added to the graph and are symmetric, i.e. <strong>matrix[i][0] = matrix[0][i] </strong>for i = 1, ..., n&nbsp;</li>
<li><strong>matrix[i][j]</strong> is None if no edge exists from the vertex&nbsp;<strong>matrix[i][0]&nbsp;</strong>to the vertex&nbsp;<strong>matrix[0][j]&nbsp;</strong></li>
<li><strong>matrix[i][j]&nbsp;</strong>is a <strong>number </strong>if an edge exists from the vertex&nbsp;<strong>matrix[i][0]&nbsp;</strong>to the vertex&nbsp;<strong>matrix[0][j]&nbsp;</strong>with weight&nbsp;<strong>number</strong></li>
</ul>
</li>
</ul>
</li>
<li><strong>graph2matrix(self) -&gt; None:</strong>
<ul>
<li>Constructs and returns an adjacency matrix from a graph</li>
<li>The output matches the format of matrices described in <strong>matrix2graph</strong></li>
<li>If the graph is empty, returns <strong>None</strong></li>
</ul>
</li>
<li><strong>graph2csv(self, filepath: str) -&gt; None:</strong>
<ul>
<li>Encodes the graph (if non-empty) in a csv file at the given location</li>
</ul>
</li>
</ul>
<p><strong><em>USE the following function however you'd like</em></strong></p>
<ul>
<li><strong>plot(self) -&gt; None:</strong>
<ul>
<li>Renders a visual representation of the graph using matplotlib and displays graphic in PyCharm
<ul>
<li><a href="https://www.jetbrains.com/help/pycharm/installing-uninstalling-and-upgrading-packages.html">Follow this tutorial to install matplotlib and numpy if you do not have them</a>, or follow the tooltip auto-suggested by PyCharm</li>
</ul>
</li>
<li>Provided for use in debugging</li>
<li>If you call this in your searches and <strong>self.</strong><strong>plot_show</strong> is true, the search process will be animated in successive plot renderings (with time between frames controlled by <strong>self.plot_delay</strong>)</li>
<li>Not tested in any testcases
<ul>
<li>All testcase graphs are constructed with <strong>self.plot_show</strong> set to False</li>
</ul>
</li>
<li>If vertices have (x,y) coordinates specified, they will be plotted at those locations</li>
<li>If vertices do not have (x,y) coordinates specified, they will be plotted at a random point on the unit circle</li>
<li>To install the necessary packages (matplotlib and numpy), follow the auto-suggestions provided by PyCharm</li>
<li>Vertices and edges are labeled; edges are color-coded by weight
<ul>
<li>If a bi-directional edge exists between vertices, two color-coded weights will be displayed</li>
</ul>
</li>
</ul>
</li>
</ul>
<p><img src="https://s3.amazonaws.com/mimirplatform.production/files/0aec2496-150a-4b85-b86f-142f235fe4ba/sample_plot.png" alt="sample_plot.png" /></p>
<p><strong><em>IMPLEMENT the following functions</em></strong></p>
<ul>
<li><strong>reset_vertices(self) -&gt; None:</strong>
<ul>
<li>Resets visited flags of all vertices within the graph</li>
<li>Used in unit tests to reset graph between searches</li>
<li><em>Time Complexity: O(V)</em></li>
<li><em>Space Complexity: O(V)</em></li>
</ul>
</li>
<li><strong>get_vertex(self, vertex_id: str) -&gt; Vertex:</strong><br />
<ul>
<li>Returns the Vertex object with id <strong>vertex_id&nbsp;</strong>if it exists in the graph</li>
<li>Returns None if no vertex with unique id <strong>vertex_id</strong> exists</li>
<li><em>Time Complexity: O(1)</em></li>
<li><em>Space Complexity: O(1)</em></li>
</ul>
</li>
<li><strong>get_vertices(self) -&gt; Set[Vertex]:</strong><br />
<ul>
<li>Returns a <strong>set </strong>of all Vertex objects held in the graph</li>
<li>Returns an empty set if no vertices are held in the graph</li>
<li><em>Time Complexity: O(V)</em></li>
<li><em>Space Complexity: O(V)</em></li>
</ul>
</li>
<li><strong>get_edge(self, start_id: str, dest_id: str) -&gt; Tuple[str, str, float]:</strong><br />
<ul>
<li>Returns the edge connecting the vertex with id&nbsp;<strong>start_id</strong> to the vertex with id <strong>dest_id</strong> in a tuple of the form&nbsp;<strong>(start_id, dest_id, weight)</strong></li>
<li>If the edge or either of the associated vertices does not exist in the graph, returns <strong>None</strong></li>
<li><em>Time Complexity: O(1)</em></li>
<li><em>Space Complexity: O(1)</em></li>
</ul>
</li>
<li><strong>get_edges(self) -&gt; Set[Tuple[str, str, float]]:</strong>
<ul>
<li>Returns a <strong>set </strong>of tuples representing all edges within the graph</li>
<li>Edges are represented as tuples <strong>(start_id, other_id, weight)</strong><strong>&nbsp;</strong>where&nbsp;
<ul>
<li><strong>start_id</strong> is the unique string id of the starting vertex</li>
<li><strong>other_id</strong> is the unique string id of the destination vertex</li>
<li><strong>weight</strong> is the weight of the edge connecting the starting vertex to the destination vertex</li>
</ul>
</li>
<li>Returns an empty set if the graph is empty</li>
<li><em>Time Complexity: O(V+E)</em></li>
<li><em>Space Complexity: O(E)</em></li>
</ul>
</li>
<li><strong>bfs(self, start_id: str, target_id: str) -&gt; Tuple[List[str], float]:</strong>
<ul>
<li>Perform a breadth-first search beginning at vertex with id&nbsp;<strong>start_id</strong> and terminating at vertex with id&nbsp;<strong>end_id</strong></li>
<li>As you explore from each vertex, iterate over neighbors using&nbsp;<strong>vertex.adj&nbsp;</strong>(not vertex.get_edges()) to ensure neighbors are visited in proper order</li>
<li>Returns tuple of the form&nbsp;<strong>([path], distance)</strong> where
<ul>
<li><strong>[path]</strong> is a list of vertex id strings beginning with&nbsp;<strong>start_id</strong>, terminating with&nbsp;<strong>end_id</strong>, and including the ids of all intermediate vertices connecting the two</li>
<li><strong>distance</strong> is the sum of the weights of the edges along the&nbsp;<strong>[path]</strong> traveled</li>
</ul>
</li>
<li>If no path exists from vertex with id&nbsp;<strong>start_id&nbsp;</strong>to vertex with <strong>end_id </strong>or if one of the vertices does not exist,&nbsp;returns tuple <strong>([],0)</strong></li>
<li>Guaranteed that <strong>start_id != target_id</strong> (since that would be a trivial path)</li>
<li>Because our adjacency maps use <a href="https://stackoverflow.com/a/57072435">insertion ordering</a>, neighbors will be visited in a deterministic order, and thus you do not need to worry about the order in which you visit neighbor vertices at the same depth</li>
<li>Use the <a href="https://docs.python.org/3/library/queue.html">SimpleQueue</a> class to guarantee O(1) pushes and pops on the queue</li>
<li><em>Time Complexity: O(V+E)</em></li>
<li><em>Space Complexity: O(V)</em></li>
</ul>
</li>
<li><strong>dfs(self, start_id: str, target_id: str) -&gt; Tuple[List[str], float]:</strong>
<ul>
<li>Wrapper function for <strong>dfs_inner</strong>, which MUST BE CALLED within this function
<ul>
<li>The majority of the work of dfs should be done in <strong>dfs_inner</strong></li>
<li>This function makes it simpler for client code to call for a dfs</li>
<li>This function makes it possible to avoid inserting vertex ids at the front of the path list on path reconstruction, which leads to sub-optimal performance (see Assignment Notes)
<ul>
<li>Hint: construct the path in reverse order in <strong>dfs_inner</strong>, then reverse the path in this function to optimize time complexity</li>
</ul>
</li>
</ul>
</li>
<li>Hint: call <strong>dfs_inner</strong> with <strong>current_id</strong> as <strong>start_id,&nbsp;</strong>then reverse the path here and return it</li>
<li><em>Time Complexity: O(V+E)&nbsp; (including calls to dfs_inner)</em></li>
<li><em>Space Complexity: O(V)&nbsp; (including calls to dfs_inner)</em></li>
</ul>
</li>
<li><strong>dfs_inner(self, current_id: str, target_id: str, path: List[str]=[]) -&gt; Tuple[List[str], float]</strong>
<ul>
<li>Performs the recursive work of depth-first search by searching for a path from vertex with id <strong>current_id</strong> to vertex with id&nbsp;<strong>target_id</strong></li>
<li><strong>MUST BE RECURSIVE</strong></li>
<li>As you explore from each vertex, iterate over neighbors using&nbsp;<strong>vertex.adj&nbsp;</strong>(not vertex.get_edges()) to ensure neighbors are visited in proper order</li>
<li>Returns tuple of the form&nbsp;<strong>([path], distance)</strong> where
<ul>
<li><strong>[path]</strong> is a list of vertex id strings beginning with&nbsp;<strong>start_id</strong>, terminating with&nbsp;<strong>end_id</strong>, and including the ids of all intermediate vertices connecting the two</li>
<li><strong>distance</strong> is the sum of the weights of the edges along the&nbsp;<strong>[path]</strong> traveled</li>
</ul>
</li>
<li>If no path exists from vertex with id <strong>current_id </strong>to vertex with<strong> target_id </strong>or if one of the vertices does not exist,&nbsp;returns tuple <strong>([],0)</strong></li>
<li>Guaranteed that <strong>start_id != target_id</strong> (since that would be a trivial path)</li>
<li>Because our adjacency maps use <a href="https://stackoverflow.com/a/57072435">insertion ordering</a>, neighbors will be visited in a deterministic order, and thus you do not need to worry about the order in which you visit neighbor vertices</li>
<li><em>Time Complexity: O(V+E)</em></li>
<li><em>Space Complexity: O(V)</em></li>
</ul>
</li>
<li><strong>detect_cycle(self) -&gt; bool:</strong>
<ul>
<li>returns&nbsp;<strong>True&nbsp;</strong>if the graph contains a cycle, otherwise returns <strong>False</strong></li>
<li>Be careful with time and space complexity, they are easy to violate here</li>
<li>The testcase should give a good idea of what constitutes a cycle and what does not</li>
<li><em>Time Complexity:&nbsp;</em><em>O(V+E)</em></li>
<li><em>Space Complexity: O(V)</em></li>
</ul>
</li>
</ul>
<h2>Application Problem</h2>
<p><img src="https://cdn1.epicgames.com/salesEvent/salesEvent/amoguslandscape_2560x1440-3fac17e8bb45d81ec9b2c24655758075" alt="Among Us | Download and Play Online for PC Now - Epic Games Store" /></p>
<p>Welcome to <a href="https://en.wikipedia.org/wiki/Among_Us">Among Us</a>, the wildly popular 2018 online multiplayer social deduction game! If you aren't familiar with how Among Us is played, see <a href="https://en.wikipedia.org/wiki/Among_Us#Gameplay">here</a>. After gathering several friends, you decide to join a lobby and start a game of Among Us!</p>
<p><img src="https://cdn.vox-cdn.com/thumbor/mCze0k6_0Bv5YMVQHk80zXi2ciI=/1400x1050/filters:format(jpeg)/cdn.vox-cdn.com/uploads/chorus_asset/file/22136080/ss_a0f2416e11bf5b47788eaa3617e092b73962b145.jpg" alt="Nearly half a billion users played Among Us in November - The Verge" /></p>
<p>It looks like you're a <strong>Crewmate</strong> (<em>and always will be for the sake of this assignment</em>)! Off to go do your tasks!&nbsp;</p>
<p><img src="https://i.ytimg.com/vi/1HS0rQADt7w/maxresdefault.jpg" alt="TRAPPING The IMPOSTOR With a 200 IQ PLAY in Among Us - YouTube" /></p>
<p>Oh, my! After completing a task, you just saw the Pink Crewmate murder somebody! They must be the impostor! You want to report the body, but Pink is too close to it and you don't want to end up like your late friend there! All hope is lost until you remember that there is an <em>emergency button&nbsp;</em>in one of the rooms on the spaceship that would also allow you to call an <em>emergency meeting</em>. If you are able to call an <em>emergency meeting</em>, then surely you would be able to convince the crew that <em>Pink is "sus"</em> and have them voted off for the victory!&nbsp;</p>
<p>Uh oh, Pink sees you and knows you will attempt to push the <em>emergency button</em> and is on their way to the button right now! You can't let them get there before you or else you won't be able to push the button! Unfortunately, you are quite new to the game and are very bad with navigating the ship; in addition, <em>it seems that the room containing the emergency button is different every time it is pressed! </em>However, you do have a graph representing the layout of the ship (which also tells you the current room with the emergency button) and one all-powerful ace in the hole: your incredible knowledge of path-finding algorithms!</p>
<p>The objective is simple: you must find the shortest path between the room you are currently in (where you witnessed a murder) and the current room with the emergency button (it is not always the same room). Everyone knows that the impostor can use vents to traverse the ship much faster than you, so your path-finding algorithm needs to be as fast as possible! Depth-first and breadth-first search won't cut it. In fact, even Dijkstra's algorithm falls short of your high-performance standards.</p>
<p>Your only option is to implement <a href="https://en.wikipedia.org/wiki/A*_search_algorithm">A* Search</a> (A-Star Search), an algorithm that accounts for both straight-line and along-edge distance to find the shortest path in fewer iterations. Unlike <a href="https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm">Dijkstra's algorithm</a>, A* will tend to avoid searching vertices that are close but in the wrong direction (<a href="https://www.youtube.com/watch?v=g024lzsknDo">gif from this video</a>). In fact, you're 100% sure that if you do not use A* Search, the impostor will beat you to the <em>emergency button</em> leading to your demise.</p>
<p><img src="https://s3.amazonaws.com/mimirplatform.production/files/002f64b5-cafa-4531-8dc8-e56f0e0de6e6/AStarGif.gif" alt="AStarGif.gif" /></p>
<h3>What is A* Search?</h3>
<p>Instead of searching the "next closest" vertex as is done in Dijkstra's algorithm, A* Search picks the vertex which is "next closest to the goal" by weighting vertices more cleverly.</p>
<p>Recall that in Dijkstra's algorithm, vertices are stored in a priority queue with a priority key equal to the current shortest path to that vertex. If we denote the current shortest path to a vertex&nbsp;<strong>v</strong> by&nbsp;<strong>g(v)</strong>, then on each iteration of Dijkstra's algorithm, we search on the vertex with <strong>min(g(v)).</strong></p>
<p>A* search takes the same approach to selecting the next vertex, but instead of setting the priority key of a vertex equal to <strong>g(v)</strong> and selecting&nbsp;<strong>min(g(v))</strong>, it uses the value&nbsp;<strong>f(v)</strong>&nbsp;and selects the vertex with&nbsp;<strong>min(f(v))</strong> where</p>
<p>&nbsp;</p>
<p><strong>f(v) = g(v) + h(v) </strong></p>
<p><strong>&nbsp; &nbsp; &nbsp; &nbsp;= current_shortest_path_to_v + estimated_distance_between_v_and_target</strong></p>
<p>&nbsp;</p>
<h3>In English, Please...</h3>
<p>A* Search prioritizes vertices <strong>v </strong>to search based on the value&nbsp;<strong>f(v)</strong>, which is the sum of</p>
<ul>
<li><strong>g(v)</strong>, or the current shortest known path to vertex&nbsp;<strong>v</strong>, and</li>
<li><strong>h(v)</strong>, which is the estimated (Euclidean or Taxicab) distance between the vertex <strong>v</strong> and the target vertex you're searching for</li>
</ul>
<p>The result is that A* prioritizes vertices to search that (1) are <em>close to the origin along a known path</em> AND which (2) are <em>in the right direction towards the target.&nbsp;</em>Vertices with a small&nbsp;<strong>g(v)&nbsp;</strong>are <em>close to the origin along a known path&nbsp;</em>and vertices with a small <strong>h(v) </strong>are <em>in the right direction towards the target<strong>, </strong></em>so we pick vertices with the smallest sum of these two values.</p>
<p><a href="https://www.youtube.com/watch?v=ySN5Wnu88nE">We strongly recommend you watch this video to build your intuition behind A* Search!</a></p>
<p>[A* is extremely versatile. Here we use Euclidean and Taxicab distances to prioritize certain directions of search, but note that any <a href="https://en.wikipedia.org/wiki/Metric_(mathematics)#Definition" target="_blank" rel="noopener noreferrer">metric</a> <strong>h(v, target)</strong> could be used should the need arise. See <a href="http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html">here</a> for more information on situations where different metrics may be practical.]</p>
<h3>Your Task</h3>
<p>Implement A* Search on your Graph ADT according to the following specifications</p>
<ul>
<li><strong>a_star(self, start_id, target_id, metric)</strong>
<ul>
<li>Perform an A* search beginning at vertex with id <strong>start_id</strong> and terminating at vertex with id&nbsp;<strong>end_id</strong></li>
<li>As you explore from each vertex, iterate over neighbors using&nbsp;<strong>vertex.adj&nbsp;</strong>(not vertex.get_edges()) to ensure neighbors are visited in proper order</li>
<li>Use the <strong>metric</strong> parameter to estimate h(v), the remaining distance, at each vertex
<ul>
<li>This is a callable parameter and will always be&nbsp;<strong>Vertex.euclidean_distance&nbsp;</strong>or&nbsp;<strong>Vertex.taxicab_distance</strong></li>
</ul>
</li>
<li>Returns tuple of the form&nbsp;<strong>([path], distance)</strong> where
<ul>
<li><strong>[path]</strong> is a list of vertex id strings beginning with&nbsp;<strong>start_id</strong>, terminating with&nbsp;<strong>end_id</strong>, and including the ids of all intermediate vertices connecting the two</li>
<li><strong>distance</strong> is the sum of the weights of the edges along the&nbsp;<strong>[path]</strong> traveled</li>
</ul>
</li>
<li>If no path exists from vertex with id&nbsp;<strong>start_id&nbsp;</strong>to vertex with <strong>end_id </strong>or if one of the vertices does not exist,&nbsp;returns tuple <strong>([],0)</strong></li>
<li>Guaranteed that <strong>start_id != target_id</strong> (since that would be a trivial path)</li>
<li>Recall that vertices are prioritized in increasing order of&nbsp;<strong>f(v) = g(v) + h(v)</strong>, where
<ul>
<li><strong>g(v)&nbsp;</strong>is the shortest known path to this vertex</li>
<li><strong>h(v)&nbsp;</strong>is the Euclidean distance from&nbsp;<strong>v</strong> to the target vertex</li>
</ul>
</li>
<li>Use the given AStarPriorityQueue class to simplify priority key updates in a search priority queue</li>
<li><strong>Implementations of this function which do not utilize the heuristic metric will not receive any credit, visible and hidden testcases included. </strong>
<ul>
<li><strong>Do not simply implement Dijkstra's Algorithm</strong></li>
</ul>
</li>
<li><em>Time Complexity: O(V+E)</em></li>
<li><em>Space Complexity: O(V)</em></li>
</ul>
</li>
</ul>
<p>To simplify the operation of updating a priority key in your search priority queue, use the following given class.</p>
<h3>class AStarPriorityQueue:</h3>
<p><em>DO NOT MODIFY the following attributes/functions</em></p>
<ul>
<li><strong>Attributes</strong>
<ul>
<li><strong>data: </strong>Underlying data list of priority queue</li>
<li><strong>locator:</strong> Dictionary to locate vertices within the priority queue</li>
<li><strong>counter: </strong>Used to break ties between vertices</li>
</ul>
</li>
<li><strong>__init__(self)</strong><br />
<ul>
<li>Constructs an AStarPriorityQueue object</li>
</ul>
</li>
<li><strong>__repr__(self)</strong>
<ul>
<li>Represents the priority queue as a string for debugging</li>
</ul>
</li>
<li><strong>__str__(self)</strong>
<ul>
<li>Represents the priority queue as a string for debugging</li>
</ul>
</li>
<li><strong>empty(self)</strong>
<ul>
<li>Returns boolean indicating whether priority queue is empty</li>
</ul>
</li>
<li><strong>push(self, priority, vertex)</strong><br />
<ul>
<li>Push the <strong>vertex </strong>object onto the priority queue with a given&nbsp;<strong>priority</strong> key</li>
<li>This priority queue has been specially designed to hold Vertex objects as values ranked by priority keys; be sure you push Vertex objects and NOT vertex id strings onto the queue</li>
</ul>
</li>
<li><strong>pop(self)</strong><br />
<ul>
<li>Visit, remove and return the Vertex object with the highest priority (i.e. lowest priority key) as a <strong>(priority, vertex)&nbsp;</strong>tuple</li>
</ul>
</li>
<li><strong>update(self, new_priority, vertex)</strong>
<ul>
<li>Update the priority of the <strong>vertex </strong>object in the queue to have a&nbsp;<strong>new_priority</strong></li>
</ul>
</li>
</ul>
<h3>Example</h3>
<p>To test your algorithm's performance, you use it to find optimal paths between locations on Among Us's most popular map: The Skeld. Below is a map of The Skeld,</p>
<p><img src="https://www.earlygame.com/uploads/images/SKELD_MAP-1.jpg" alt="Among Us: Map Guide to Skeld | EarlyGame" /></p>
<p>and below is a modified and simplified version of The Skeld represented as a graph. <em>Fun fact: the graph below was made with the plot() function we provide you (see above). The yellow points represent visited vertices from the top example testcase below. Be sure to initialize your Graph object with "plt_show=True" before calling plot().</em></p>
<p><img src="https://i.imgur.com/MErPKGV.png" /></p>
<ul>
<li><strong style="background-color: transparent;">a_star('Reactor', 'Cafeteria')</strong><span style="background-color: transparent;"> would return </span>
<ul>
<li><span style="background-color: transparent;">(['Reactor', 'A', 'B', 'G', 'J', 'M', 'Cafeteria'], 22)</span></li>
<li><span style="background-color: transparent;">This finds the same optimal path as Dijkstra</span></li>
</ul>
</li>
<li><strong style="background-color: transparent;">a_star('Reactor', 'Communications')</strong><span style="background-color: transparent;"> would return </span>
<ul>
<li><span style="background-color: transparent;">(['Reactor', 'A', 'B', 'G', 'H', 'D', 'E', 'Communications'], 35)</span></li>
<li><span style="background-color: transparent;">This finds the optimal path (note that BFS would find a sub-optimal path of path length 37)</span></li>
</ul>
</li>
<li><strong><span style="background-color: transparent;">a_star('Cafeteria', 'Med Bay') </span></strong><span style="background-color: transparent;">would return</span>
<ul>
<li><span style="background-color: transparent;">(['Cafeteria', 'M', 'J', 'K', 'Med Bay'], 8)</span></li>
<li><span style="background-color: transparent;">Although two equally-optimal paths exist, A* chooses the one that's closer to the straight line connecting the Cafeteria to the Med Bay</span></li>
</ul>
</li>
</ul>
<p><strong><span style="background-color: transparent;">Note: maps/graphs may not be accurate to actual Among Us maps.</span></strong></p>
<h2>Submission:</h2>
<h3>Deliverables:</h3>
<p>Be sure to upload the following deliverables in a .zip folder to Mimir by 8:00 pm EST, on Thursday, December 2nd.</p>
<p>Your .zip folder can contain other files (for example, description.md and tests.py), but must include (at least) the following:</p>
<pre class="code-line"><code>|- Project5.zip
    |&mdash; Project5/
        |&mdash; README.xml       (for project feedback)
        |&mdash; __init__.py      (for proper Mimir testcase loading)       
        |&mdash; Graph.py         (contains your solution source code)</code></pre>
<h3>Grading</h3>
<ul>
<li>Tests (70)</li>
<li>Manual (30)
<ul>
<li>Time &amp; Space Complexity (30)&nbsp;
<ul>
<li>M2 -<strong> degree, get_edges, distances, reset_vertices&nbsp; &nbsp; &nbsp; &nbsp;</strong>__/3</li>
<li>M3 -&nbsp;<strong>get_vertex, get_vertices, get_edge, get_edges</strong>&nbsp; __/3</li>
<li>M4 - <strong>bfs&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;</strong>__/6</li>
<li>M5 -<strong> dfs</strong>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;__/6</li>
<li>M6 -&nbsp;<strong>detect_cycle&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;</strong>__/6</li>
<li>M7 - <strong>a_star&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; </strong>__/6</li>
</ul>
</li>
</ul>
</li>
</ul>
<h2>Extra:</h2>
<p><img src="https://cdn.vox-cdn.com/thumbor/ot2xqOkZU-kuhd4G4fQWkjZs2CU=/0x60:1920x1020/fit-in/1200x600/cdn.vox-cdn.com/uploads/chorus_asset/file/21874606/Emergency_Meeting.jpg" alt="Why Among Us&amp;#39; Emergency Meeting is the big social media mood - Polygon" /></p>
<p>Get to that emergency button before the <a href="https://youtu.be/grd-K33tOSM">impostor</a>! You can do it!</p>
<p>&nbsp;</p>
