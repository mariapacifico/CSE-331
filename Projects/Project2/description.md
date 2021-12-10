<h1 class="code-line">Project 2: Hybrid Sorting</h1>
<p class="code-line"><strong>Due: October 21st @ 8:00pm EST</strong></p>
<p class="code-line"><em>This is not a team project, do not copy someone else&rsquo;s work.</em></p>
<h2 class="code-line">Assignment Overview</h2>
<p><img style="display: block; margin-left: auto; margin-right: auto;" src="https://s3.amazonaws.com/mimirplatform.production/files/c3e7fecd-4807-44e5-bb0d-a41a9dfd2f0c/sorting_complexities.png" alt="sorting_complexities.png" /></p>
<p class="code-line">A&nbsp;<strong>sorting algorithm</strong>&nbsp;is an algorithm that puts elements in a <a href="https://en.wikipedia.org/wiki/Total_order">certain order</a>. Such algorithms are often used to organize an array or list in numerical or lexicographical order, however their use is not limited in scope to such simple orderings, or to such simple structures - a fact that will be demonstrated in this project.</p>
<p class="code-line">Throughout the 20th century, as the domain of problems to which computers were applied grew, so to did the size of data sets that required sorting. This resulted in the rapid development of sorting algorithms. Clearly&nbsp;<em>O(n<sup>2</sup>)&nbsp;</em>algorithms, such an selection and bubble sort, were inferior to faster <em>O(nlog(n)) </em>algorithms, such as quick or merge sort, but by the 1970s even these weren't achieving speeds desired. This led to the development of ultra-optimized hybrid sorting algorithms, which tie together multiple sorting procedures recursively so as to apply the optimal sort to each chunk of data based on its size.</p>
<p class="code-line">This project focuses on the <em><strong>Insertion Sort</strong> </em>and the&nbsp;<strong><em>Merge Sort </em></strong>algorithms, and on a hybrid of the two. This hybridization is not merely a toy exercise - in fact, Python sorts its built in lists through a <a title="timsort" href="https://hg.python.org/cpython/file/tip/Objects/listsort.txt">hybrid of merge and insertion sort</a>.</p>
<h3 class="code-line">Insertion Sort</h3>
<p class="code-line"><img class="" alt="" /><img style="display: block; margin-left: auto; margin-right: auto;" src="https://s3.amazonaws.com/mimirplatform.production/files/71bf53c7-b452-46c3-9d3b-84d2da030187/insertion_sort.png" alt="insertion_sort.png" /></p>
<p class="code-line">Insertion sort is an in place comparison based sorting algorithm. It builds a final sorted array by comparing one element at a time and inserting it into it's appropriate location.</p>
<p class="code-line">The worst case runtime is <em>O(n^2)</em>. The best case runtime is linear - in the case that the array is already sorted. The space complexity is O(1) for an in place implementation.</p>
<h3 class="code-line">Merge Sort</h3>
<p class="code-line"><img class="" alt="" /><img style="display: block; margin-left: auto; margin-right: auto;" src="https://s3.amazonaws.com/mimirplatform.production/files/5aa69ad1-1cfd-4909-b14f-765027587102/merge_sort.png" alt="merge_sort.png" /></p>
<p class="code-line">Merge sort is one of the most efficient sorting algorithms. It works on the principle of <a href="https://en.wikipedia.org/wiki/Divide-and-conquer_algorithm">Divide and Conquer</a>. Merge sort repeatedly breaks down a list into several sublists until each sublist consists of a single element and merges those sublists in a manner that results in a sorted list.</p>
<p class="code-line">The runtime is <em>O(nlog(n)) and</em>&nbsp;<em>Theta(nlog(n))</em>. The space complexity is <em>O(n)</em> - new arrays are created everytime you divide. (<a href="https://stackoverflow.com/a/28641693">Why isn't the space <em>O(nlogn)</em>?</a>)</p>
<h3 class="code-line">Hybrid Sort</h3>
<p class="code-line">While Merge Sort has a faster average runtime than Insertion Sort, there are instances that an Insertion Sort is more efficient. Due to the overhead of recursively splitting containers with a Merge Sort, Insertion Sort can have a faster performance with small sets of data. Thus, the two algorithms are often combined, with recursive calls to merge sort ceasing once the data is subdivided into small enough arrays.</p>
<h2 class="code-line">Project Details</h2>
<h3 class="code-line">Overview</h3>
<p class="code-line">You will be implementing the Insertion Sort Algorithm and the Merge Sort Algorithm. You will develop your Merge Sort such that it can be used as a Hybrid Sort when given a threshold value. The Hybrid Sort will rely on Merge Sort until the partitioned lists are less than or equal to a given threshold, at which point you will switch to Insertion Sort.</p>
<p class="code-line">These sorting algorithms will have an optional parameter that accepts a&nbsp;<em><strong>callable</strong> -</em> an object that can accept its own arguments and can even return its own object.&nbsp;<em>Callable&nbsp;</em>is the alias used by python's typing module for a <strong><em>Function- </em>in Python, functions are objects, and may be passed as arguments to other functions<em>.</em></strong> For this project, you will be creating simple functions to pass in as callables, which should determine how your sorting algorithms will sort. The default callable argument to our functions is a lambda function that is intended to sort your objects in the usual increasing order. For the application problem, you may need to create more complex functions that cannot be written as a lambda.</p>
<p class="code-line"><img class="" alt="" /><img src="https://s3.amazonaws.com/mimirplatform.production/files/c9ed1cbd-5a82-4e58-add3-3102d4d4ffd6/Project2Lambdaimage.png" alt="Project2Lambdaimage.png" /></p>
<p class="code-line">You'll want to use the above example,&nbsp;<em>comparator</em> to compare each element in the list you are sorting . <em>comparator</em> will only return true if the first parameter passed in, x, is less than or equal to the second parameter passed in, y. To use the comparator shown in the image above in your code, you'd write</p>
<pre class="code-line">if comparator(x, y):&nbsp; &nbsp; # does the same thing as if x &lt;= y in this case</pre>
<p class="code-line">In addition to these sorting algorithms, you will be implementing an algorithm to determine the&nbsp;<em>inversion count</em> of a list of elements. This algorithm will be integrated into your merge_sort function. You will only calculate the inversion count when your function is not being used as a Hybrid Sort, that is, when the threshold is 0.</p>
<p class="code-line">The inversion count is how far away a list of elements is from being sorted. The inversion count of a sorted array is 0. You can think of the number of inversions as the number of pairs of elements that are out of order.</p>
<p class="code-line"><strong>Two elements a[i] and a[j] form an inversion if a[i] &gt; a[j] and i &lt; j</strong></p>
<h5 class="code-line">Examples:</h5>
<p class="code-line"><strong>data = [3,2,9,0]</strong></p>
<ul>
<li class="code-line">data has 4 inversions:
<ul>
<li class="code-line">(3, 2): 3 &gt; 2 but 3 comes before 2</li>
<li class="code-line">(3, 0): 3 &gt; 0 but 3 comes before 0</li>
<li class="code-line">(2, 0): 2 &gt; 0 but 2 comes before 0</li>
<li class="code-line">(9, 0): 9 &gt; 0 but 9 comes before 0</li>
</ul>
</li>
</ul>
<p class="code-line"><strong>data = [1, 2, 3, 4, 5]</strong></p>
<ul>
<li class="code-line">data has 0 inversions</li>
</ul>
<p class="code-line">Note: Although you can swap the elements of the inversions to form a sorted array, the inversion count is not the same as the minimum number of swaps to sort the array.</p>
<p class="code-line">For instance, data = [10, 1, 2, 0].</p>
<p class="code-line">You could sort this in 1&nbsp;<em>swap</em>&nbsp;(10, 0), but there are 5&nbsp;<em>inversions</em>&nbsp;(10, 1), (10, 2), (10, 0), (1, 0), (2, 0).</p>
<h3 class="code-line">Assignment Specs</h3>
<p class="code-line">You are given one file,&nbsp;<a title="http://HybridSort.py" href="http://hybridsort.py/">HybridSort.py</a>. You must complete and implement the following functions. Take note of the specified return values and input parameters.&nbsp;<strong><em>Do not change the function signatures</em>.</strong></p>
<p class="code-line">You must adhere to the required time &amp; space complexities.</p>
<p class="code-line"><strong><a title="http://HybridSort.py" href="http://hybridsort.py/">HybridSort.py</a>:</strong></p>
<ul>
<li><span style="font-family: geomanist, sans-serif;"><strong>insertion_sort(data: List[T], comparator: Callable[[T,T], bool]) -&gt; None:</strong></span><br />
<ul>
<li>Given a list of values and comparator, perform an insertion sort on the list using the comparator.</li>
</ul>
</li>
</ul>
<ul>
<li style="list-style-type: none;">
<ul>
<li class="code-line"><strong>param data:</strong> List of items to be sorted</li>
<li class="code-line"><strong>param comparator:&nbsp;</strong>A function, which takes an argument two objects of the same type as those in the list <strong>data </strong>and returns `True` if the first argument is less than or equal to the second according to some ordering, else `False`.</li>
<li class="code-line"><strong>return:</strong>&nbsp;None</li>
<li class="code-line">Time Complexity: <em>O(n^2)</em></li>
<li class="code-line">Space Complexity: <em>O(1]</em></li>
</ul>
</li>
</ul>
<ul>
<li><strong>merge_sort(data, threshold: int = 0, comparator: Callable[[T,T], bool]) -&gt; int:&nbsp;</strong><br />
<ul>
<li>Given a list of values, perform a merge sort to sort the list and calculate the inversion count. When a threshold is provided, use a merge sort algorithm until the partitioned lists are smaller than or equal to the threshold - then use insertion sort. Be sure to use the comparator.</li>
</ul>
</li>
</ul>
<ul>
<li style="list-style-type: none;">
<ul>
<li class="code-line"><strong>param data:</strong>&nbsp;List of items to be sorted.</li>
<li class="code-line"><strong>param threshold:</strong>&nbsp;int representing the size of the data at which insertion sort should be used. Defaults to 0.</li>
<li><strong>param comparator:&nbsp;</strong>A function, which takes an argument two objects of like type to those in the list&nbsp;<strong>data </strong>and returns `True` if the first argument is less than or equal to the second according to some ordering, else `False`.</li>
<li class="code-line"><strong>return:</strong>&nbsp;int representing inversion count, else 0 if <strong>threshold</strong> &gt; 0.</li>
<li class="code-line"><strong>NOTE:</strong>&nbsp;The inversion count will be calculated when only a <code>merge_sort()</code> algorithm is used! (when threshold is 0) return 0 otherwise.</li>
<li class="code-line">Time Complexity: <em>O(nlog(n))</em></li>
<li class="code-line">Space Complexity: <em>O(n)</em></li>
</ul>
</li>
</ul>
<ul>
<li><strong>hybrid_sort(data: List[T], threshold: int, comparator: Callable[[T,T], boo]) -&gt; None:</strong><br />
<ul>
<li><a href="https://en.wikipedia.org/wiki/Wrapper_function">Wrapper</a> function to call <code>merge_sort()</code> as a Hybrid Sorting Algorithm. Should call <code>merge_sort()</code> with provided threshold, and comparator function.</li>
</ul>
</li>
</ul>
<ul>
<li style="list-style-type: none;">
<ul>
<li class="code-line"><strong>param data:</strong> List of items to be sorted.</li>
<li class="code-line"><strong>param threshold:</strong>&nbsp;int representing the size of the data at which insertion sort should be used.</li>
<li><strong>param comparator:&nbsp;</strong>A function, which takes an argument two objects of like type to those in the list&nbsp;<strong>data </strong>and returns `True` if the first argument is less than or equal to the second according to some ordering, else `False`.</li>
<li class="code-line"><strong>return:</strong>&nbsp;None</li>
<li class="code-line"><strong>NOTE:&nbsp;</strong>If this is more than one line, you're probably doing something wrong.</li>
<li class="code-line">Time Complexity: <em>O(nlog(n))</em></li>
<li class="code-line">Space Complexity: <em>O(n)</em></li>
</ul>
</li>
</ul>
<ul>
<li><strong>inversions_count(data: List[T]) -&gt; int:</strong><br />
<ul>
<li>Wrapper function to call <code>merge_sort()</code> on a <strong>copy</strong> of data to retrieve the inversion count. Should call <code>merge_sort()</code> with&nbsp;<em>no threshold, and the default comparator.</em></li>
<li>This function should <em>copy the original data</em> - we want to determine the number of inversions in its current ordering, not necessarily sort it.</li>
</ul>
</li>
</ul>
<ul>
<li style="list-style-type: none;">
<ul>
<li class="code-line"><strong>param data:</strong>&nbsp;List of items to determine the inversion count of.</li>
<li class="code-line"><strong>return:</strong>&nbsp;int representing inversion count.</li>
<li class="code-line">Time Complexity: <em>O(nlog(n))</em></li>
<li class="code-line">Space Complexity: <em>O(n)</em></li>
</ul>
</li>
</ul>
<ul>
<li><strong>reverse_sort(data: List[T], threshold: int) -&gt; None:</strong><br />
<ul>
<li>Wrapper function to use <code>merge_sort() </code>to sort the data in reverse. Should call <code>merge_sort()</code> with provided threshold, and a comparator you define.</li>
</ul>
</li>
</ul>
<ul>
<li style="list-style-type: none;">
<ul>
<li class="code-line"><strong>param data:</strong> List of items to be sorted.</li>
<li class="code-line"><strong>param threshold:&nbsp;</strong>int represnting the size of the data at which insertion sort should be used.</li>
<li class="code-line"><strong>return:</strong>&nbsp;None</li>
<li class="code-line">Time Complexity: <em>O(nlog(n))</em></li>
<li class="code-line">Space Complexity: <em>O(n)</em></li>
</ul>
</li>
</ul>
<h2>Application: Navigation Troubles</h2>
<p><img src="https://s3.amazonaws.com/mimirplatform.production/files/84c15925-0e0d-4fab-b06f-c913d6c4418c/ah.jpg" alt="ah.jpg" /></p>
<p>Congratulations, you've been hired by the United States Navy to diagnose some serious problems in their fleet's <a href="https://en.wikipedia.org/wiki/Sonar">sonar technology</a>! It turns out that the sonar employed by Navy ships is unable to accurately perceive distance and makes frequent mistakes; often it perceives an object to be further away than it truly is, or closer.</p>
<p>This causes a conundrum. The Navy cannot retire its entire fleet, or attempt to repair it all at once -- the resources to do so simply do not exist. Instead, your colleagues have devised a test to determine which ships are in most urgent need of repair and which can continue service in their current state.</p>
<p>The entire navy fleet has been assembled in an empty patch of the Pacific Ocean and arranged in a way such that each ship can be located via satellite and assigned exact coordinates relative to one another. Each ship sends out a sonar pulse, locating all the other navy ships. If this ship's sonar was functioning correctly, it would first perceive the ship closest to it, and then second closest, and so on. Instead, it perceives nearby ships in seemingly random order (regardless of their physical coordinates). If a further ship is perceived before a closer ship, it is considered a mistake. Your job is to rank ships based on which make the least mistakes.&nbsp;<strong>IF TWO SHIPS MAKE THE SAME NUMBER OF MISTAKES, SORT THOSE SHIPS IN ALPHABETICAL ORDER BASED ON THEIR NAME.</strong></p>
<p>Let's look at an example...</p>
<p><img src="https://s3.amazonaws.com/mimirplatform.production/files/8057bfa5-db20-456d-83b5-452dc20fe962/Figure_1.png" alt="Figure_1.png" /></p>
<p>Here we see four ships plotted, the&nbsp;<em>USS Ian Barber,&nbsp;</em>the&nbsp;<em>USS Andrew,&nbsp;</em>the&nbsp;<em>USS Bank</em>, and the&nbsp;<em>UHH H</em>. These are coordinates <em>(-1,3), (-2,0), (0,6), </em>and <em>(-4,-4)</em><em>&nbsp;</em>respectively.</p>
<p>Let's suppose that while using its sonar, the&nbsp;<em>UHH H</em> perceives the other ships in this order:&nbsp;<em>USS Bank, USS Andrew,&nbsp;</em>and then the&nbsp;<em>USS Ian Barber</em>. Clearly, mistakes have been made.</p>
<p>First, let's look at the distances, as defined by the standard euclidean metric:</p>
<p><strong>D(H, Andrew) = sqrt( (x_1 - x_2)^2 + (y_1 - y_2)^2 ) = sqrt( (-4 - -2)^2 + (-4 - 0)^2 ) = sqrt(20) = 4.47...</strong></p>
<p><strong>D(H, Ian) = 7.62...</strong></p>
<p><strong>D(H, Bank) = 10.77...</strong></p>
<p>The correct order should have been the <em>USS Andrew, USS Ian Barber, </em>and then the <em>USS Bank.</em></p>
<p>The&nbsp;<em>USS Bank&nbsp;</em>was perceived before both the <em>USS </em><em>Andrew&nbsp;</em>and the <em>USS </em><em>Ian. </em>This constitutes two mistakes because we have to swap the <em>USS Bank </em>with the <em>USS Ian Barber </em>and then the <em>USS Bank </em>with the <em>USS Andrew</em> afterward to fix this.</p>
<p>The <em>USS </em><em>Andrew </em>and the <em>USS Ian </em>were perceived in correct order relative to one another (the <em>USS </em><em>Andrew </em>came before the <em>USS Ian </em>as it should have), so this doesn't contribute any other mistakes to the count.</p>
<p>Continue this process for each ship, determining the number of mistakes made. From this, you will sort ships by least to most mistakes made.</p>
<p>Suppose the&nbsp;<em>UHH H </em>and the&nbsp;<em>USS Ian&nbsp;</em>both had equally faulty sonar, and thus both made two mistakes. Now, suppose the&nbsp;<em>USS Andrew&nbsp;</em>made only one mistake, and the&nbsp;<em>USS Bank&nbsp;</em>made zero mistakes.</p>
<p>In this case, we would first order based on mistakes made, giving a result of:</p>
<p><strong>[Bank, Andrew, Ian, H]</strong></p>
<p>However, since alphabetically the&nbsp;<em>UHH H&nbsp;</em>comes before the&nbsp;<em>USS Ian</em>, our final list would be:</p>
<p><strong>[Bank, Andrew, H, Ian]</strong>.</p>
<h3>class Ship:</h3>
<p><em>DO NOT MODIFY the following attributes/functions</em></p>
<ul>
<li><strong>Attributes:</strong>
<ul>
<li><strong>x: int:</strong> The x (longitudinal) coordinate of the ship</li>
<li><strong>y: int:</strong> The y (latitudinal) coordinate of the ship</li>
<li><strong>name: String:</strong> The name of the ship</li>
</ul>
</li>
<li><strong><span class="pl-en"><span class="pl-token">__init__</span></span>(<span class="pl-s1">self</span>, <span class="pl-s1">name</span>: <span class="pl-s1">str</span>, <span class="pl-s1">x</span>: <span class="pl-s1">int</span>, <span class="pl-s1">y</span>: <span class="pl-s1">int</span>) <span class="pl-c1">-&gt;</span> <span class="pl-c1">None</span>:</strong>
<ul>
<li>Constructs a ship object</li>
<li><strong>x: int:</strong> The x (longitudinal) coordinate of the ship</li>
<li><strong>y: int:</strong> The y (latitudinal) coordinate of the ship</li>
<li><strong>name: String:</strong> The name of the ship</li>
</ul>
</li>
<li><strong>__str__(self) -&gt; str&nbsp;</strong>and&nbsp;<strong>__repr__(self) -&gt; str:</strong><br />
<ul>
<li>represent ships as strings, for the sake of debugging</li>
</ul>
</li>
<li><strong>__eq__(self, other: Ship) -&gt; bool</strong><strong>:</strong>
<ul>
<li>returns True iff the two ships are equivalent (same data members)</li>
</ul>
</li>
<li><strong><span class="pl-en"><span class="pl-token">__hash__(self) -&gt; int:</span></span></strong>
<ul>
<li><span class="pl-en"><span class="pl-token">Used for hashing Ship objects for use as keys in dictionaries</span></span></li>
</ul>
</li>
<li><strong><span class="pl-en"><span class="pl-token">euclidean_distance</span></span>(<span class="pl-s1">self</span>, <span class="pl-s1">other</span>: <span class="pl-v"><span class="pl-token">Ship</span></span>) <span class="pl-c1">-&gt;</span> <span class="pl-s1">float</span>:</strong>
<ul>
<li>Computes the euclidean distance (via the formula shown) between two ships</li>
<li><strong>other: Ship:&nbsp;</strong>The other ship involved in the computation</li>
</ul>
</li>
<li><strong><span class="pl-en"><span class="pl-token">taxicab_distance</span></span>(<span class="pl-s1">self</span>, <span class="pl-s1">other</span>: <span class="pl-v"><span class="pl-token">Ship</span></span>) <span class="pl-c1">-&gt;</span> <span class="pl-s1">float</span>:</strong>
<ul>
<li>Computes the taxicab distance (via the formula shown) between two ships</li>
<li><strong>other: Ship:&nbsp;</strong>The other ship involved in the computation</li>
</ul>
</li>
</ul>
<p><em>Implement the following function:</em></p>
<p><strong>navigation_test(<span class="pl-s1">ships</span>: <span class="pl-v">Dict</span>[<span class="pl-v"><span class="pl-token">Ship</span></span>, <span class="pl-v">List</span>[<span class="pl-v"><span class="pl-token">Ship</span></span>]], <span class="pl-s1">euclidean</span>: <span class="pl-s1">bool</span>) <span class="pl-c1">-&gt;</span> <span class="pl-v">List</span>[<span class="pl-v"><span class="pl-token">Ship</span></span>]:</strong></p>
<ul>
<li>This function takes in a dictionary, which maps each ship to a list of other ships in the order in which they are perceived via sonar, and a boolean indicating which distance metric to use.</li>
<li><strong>ships: Dict[Ship, List[Ship]]:&nbsp;</strong> A dictionary, where each key is a&nbsp;<strong>ship</strong> object, and each key is a list of other ships.</li>
<li><strong>euclidean: bool:&nbsp;</strong>This boolean indicator tells you which distance metric to use. If <strong>True</strong>, use&nbsp;<strong>euclidean_distance</strong> to determine the number of sonar mistakes made, else use&nbsp;<strong>taxicab_distance</strong>.&nbsp;</li>
<li><strong>Returns:&nbsp;</strong>A python list of ship objects, ranked in order of the number of mistakes made, from least to most</li>
<li class="code-line">Time Complexity: <em>O(n^2*log(n)) - where `n` is the number of ships</em></li>
<li class="code-line">Space Complexity: <em>O(n)</em></li>
</ul>
<p>&nbsp;</p>
<h2 class="code-line">Submission</h2>
<h4 class="code-line">Deliverables</h4>
<p class="code-line">Be sure to upload the following deliverables in a .zip folder to Mimir by 8:00p Eastern Time on Thursday, 10/21/21.</p>
<p class="code-line">Your .zip folder can contain other files (for example, description.md and tests.py), but must include (at least) the following:</p>
<pre class="code-line"><code>|- Project2.zip
    |&mdash; Project2/
        |&mdash; README.xml       (for project feedback)
        |&mdash; __init__.py      (for proper Mimir testcase loading)       
        |&mdash; HybridSort.py    (contains your solution source code)
</code></pre>
<h4 class="code-line">Grading</h4>
<p class="code-line">The following 100-point rubric will be used to determine your grade on Project2:</p>
<p class="code-line"><em><strong>NOTE THAT THE SPEED TESTS ARE WORTH ZERO POINTS, AS THEY ARE ONLY THERE TO HELP YOU CHECK IF YOU ARE OVER TIME COMPLEXITY.</strong></em></p>
<ul>
<li class="code-line">Tests (70)
<ul>
<li class="code-line">00 - Coding Standard: __/5</li>
<li class="code-line">01 - Insertion Sort Basic: __/4</li>
<li class="code-line">02 - Insertion Sort Comparator: __/4</li>
<li class="code-line">03 - Insertion Sort Comprehensive: __/5</li>
<li class="code-line">04 - Merge Sort Basic: __/2</li>
<li class="code-line">05 - Merge Sort Threshold: __/3</li>
<li class="code-line">06 - Merge Sort Comparator: __/4</li>
<li class="code-line">07 - Merge Sort Comprehensive: __/5</li>
<li class="code-line">08 - Hybrid Sort Basic: __/2</li>
<li class="code-line">09 - Hybrid Sort Comprehensive: __/5</li>
<li class="code-line">10 - Hybrid Sort Speed: __/0</li>
<li class="code-line">11 - Reverse Sort Basic: __/3</li>
<li class="code-line">12 - Reverse Sort Speed: __/0</li>
<li class="code-line">13 - Inversion Count: __/10</li>
<li class="code-line">14 - Application Small: __/5</li>
<li class="code-line">15 - Application Large: __/10</li>
<li class="code-line">16 - README.xml validity: __/3</li>
</ul>
</li>
<li class="code-line">Manual (30)
<ul>
<li>Time and space complexity points are <strong>all-or-nothing</strong> for each function. If you fail to meet time <strong>or</strong> space complexity in a given function, you do not receive manual points for that function.&nbsp;</li>
<li class="code-line">Insertion Sort Complexities: __/4</li>
<li class="code-line">Merge Sort Complexities: __/8</li>
<li class="code-line">Hybrid Sort Complexities: __/4</li>
<li class="code-line">Reverse Sort Complexities: __/4</li>
<li class="code-line">Application Complexities: __/10</li>
</ul>
</li>
</ul>
<h2 class="code-line">Tips, Tricks &amp; Notes</h2>
<ul>
<li class="code-line"><em><strong>You must fill out function docstrings to receive Coding Standard points.</strong></em></li>
<li class="code-line"><em><strong>You may not use Python's built in sort, or any imported sorting methods</strong></em></li>
<li class="code-line">There are different ways to implement merge sort, but make sure you are aiming for a solution that will fit the time complexity! If your recursive calls are some form of <code>merge_sort(start + 1: end)</code>, this will <em>not</em> be <em>O(nlog(n))</em>.</li>
<li class="code-line">Your wrapper functions (<code>hybrid_sort(), inversions_count(), reverse_sort()</code>)&nbsp; should be simple functions containing one call to <code>merge_sort()</code>. This may seem useless, but see <a title="Wrapper Functions" href="https://en.wikipedia.org/wiki/Wrapper_function" target="_blank" rel="noopener noreferrer">here</a> for the significance of wrapper functions.</li>
<li class="code-line">Test cases will only test the functions specified.
<ul>
<li class="code-line">Note: The Merge Sort testcases will not test for inversion count</li>
</ul>
</li>
<li class="code-line">Try these web applications to visualize sorting algorithms:
<ul>
<li class="code-line"><a title="https://visualgo.net/bn/sorting" href="https://visualgo.net/bn/sorting">https://visualgo.net/bn/sorting</a>&nbsp;(includes inversion count - "inversion index")</li>
<li class="code-line"><a title="https://opendsa-server.cs.vt.edu/embed/mergesortAV" href="https://opendsa-server.cs.vt.edu/embed/mergesortAV">https://opendsa-server.cs.vt.edu/embed/mergesortAV</a>&nbsp;(good merge sort visualization)</li>
<li class="code-line"><a title="https://www.cs.usfca.edu/~galles/visualization/ComparisonSort.html" href="https://www.cs.usfca.edu/~galles/visualization/ComparisonSort.html">https://www.cs.usfca.edu/~galles/visualization/ComparisonSort.html</a></li>
</ul>
</li>
<li>The file&nbsp;<code>plot.py</code>&nbsp;has been provided for comparing your sorting function's runtime.
<ul>
<li class="code-line">Run this file to see a graphical representation of your functions' performances.</li>
<li class="code-line">You may need to install matplotlib and numpy.
<ul>
<li class="code-line">If you are not familiar with the terminal instructions are here:&nbsp;<a title="https://www.jetbrains.com/help/pycharm/installing-uninstalling-and-upgrading-packages.html" href="https://www.jetbrains.com/help/pycharm/installing-uninstalling-and-upgrading-packages.html">https://www.jetbrains.com/help/pycharm/installing-uninstalling-and-upgrading-packages.html</a></li>
<li class="code-line">Otherwise use these commands:
<ul>
<li class="code-line">pip install matplotlib</li>
<li class="code-line">pip install numpy</li>
</ul>
</li>
</ul>
</li>
<li class="code-line">You do not have to use this.</li>
</ul>
</li>
<li class="code-line">You should already have a function handy to help you count sonar mistakes in the application problem.</li>
</ul>
<p class="code-line"><em>This project was created by Andrew Haas, Elizabeth Deback, and Adam Kasumovic.</em></p>
