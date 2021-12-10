<h1>CC7 - FS21 - You're Finally Awake!</h1>
<p><strong>Due: Tuesday, November 23rd, 2021 by 8:00p ET</strong></p>
<p><em>This is not a team project. Do not copy someone else&rsquo;s work.</em></p>
<h2>Introduction</h2>
<p><img src="https://media.giphy.com/media/LBXFiLVoDE8K3I7Vpk/giphy.gif" alt="Skyrim gif" width="100%" /></p>
<p>As you begin to work on the latest CSE 331 coding challenge, you suddenly blackout and awake to find yourself inside the world of <a href="https://en.wikipedia.org/wiki/The_Elder_Scrolls_V:_Skyrim">Skyrim</a>. You are shackled in the back of a horse-drawn carriage along with multiple people. Another prisoner, Ralof, has noticed that you are conscious and greets you. He explains that you were ambushed while crossing the border from Cyrodil to Skyrim. Apparently, the Imperial Army has captured you and your future isn't looking so bright. Luckily, fate is on your side because a dragon ambushes the fort you are &nbsp;detained in and you escape into the wilderness of Skyrim.</p>
<p>The wilderness of Skyrim is dangerous, but your new friend and fellow escapee Ralof shows you the power of perks. You see, as you traverse the world, finish quests, slay monsters, pick locks, smith dagger, etc... you will gain XP which will allow you to level up and earn perk points. Using these perk points, you will unlock perks that permit you to enhance your abilities in a myriad of ways, improving your chances of surviving and thriving in this new world. Abilities include smithing powerful armor, casting stronger spells, increasing your damage output with bows and swords, being able to easily persuade merchants and guards, and more! As a farewell gift, Ralof gives you a list of the best perks to unlock for the best builds he knows of.</p>
<p>Now on your own, you examine the list and you are excited to spend your first perk point on the 'Power Shot' perk since you've decided to become a powerful stealth archer (how original). Yet, as you try to unlock the perk, you realize Ralof left out one critical detail! The perk system does not let you unlock any perk as you see fit. The perks are arranged in such a way that you must unlock certain perks before reaching others, which are usually more powerful. On top of that, the perks are organized into different trees (or graphs). Thus, you are unsure of which is the best to pick first or if you have enough perk points to get to the perk you want in a certain tree.</p>
<p>Upon closer inspection, you realize that the organization of the graphs (or perk trees) strikes a resemblance to something that was covered in a lecture the other day: the perk graphs are just <a href="https://en.wikipedia.org/wiki/Directed_acyclic_graph">Directed Acyclic Graphs</a>, or DAGs, for short. This means that you can traverse them in certain ways as you learned in the lecture (HINT HINT) to find the order in which perks need to be obtained before getting to the good one Ralof told you about, while also counting the number of perk points it will take to get there. You begin designing an algorithm to return the ordering of the perk tree so you can figure out what perks you need before getting the target perk and whether you have enough perk points to get it.</p>
<h2>Challenge</h2>
<h4>Overview</h4>
<p>Given a DAG representing a perk tree, return a list representing an order of perks such that for each perk, it is only listed after every perk it depends on is listed first. In other words, for every perk pair (p1, p2) where p1 has an edge to p2, p1 will be listed before p2 in the output list. The list of dependencies should end at the target perk. Each perk costs one perk point and if the total amount of perks it takes to get to the target perk (including getting the target perk itself) is greater than the number of points, then return an empty list indicating it is impossible to get this perk with the given amount of points.</p>
<h4>Function Signature</h4>
<p><code>perk_traversal(self, target: str, points: int) -&gt; List[str]:</code></p>
<p><code></code><strong><em>OPTIONAL</em>&nbsp;Utility Function:</strong></p>
<p><code>perk_organizer(self, vertex: str, visited: Dict[str], stack: List[str], target: str) -&gt; bool:</code><code></code></p>
<p><code></code></p>
<h4>Input</h4>
<ul>
<li><code>perk_traversal</code>
<ul>
<li>A <code>PerkGraph</code> object that represents a DAG in the form of a perk tree</li>
<li><code>target</code>: a string that indicates the desired perk</li>
<li><code>points</code>: an integer that indicates how many perk points we have to spend</li>
</ul>
</li>
<li><code>perk_organizer</code></li>
</ul>
<ul>
<li style="list-style-type: none;">
<ul>
<li>A <code>PerkGraph</code> object that represents a DAG in the form of a perk tree</li>
<li><code>vertex</code>: a string that represents a vertex in the <code>PerkGraph</code></li>
<li><code>visited</code>: a Python <code>Dic</code><code>t</code>&nbsp;that keeps track of which vertices have been visited</li>
<li><code>stack</code>:&nbsp;a Python <code>List</code>&nbsp;implementation of a stack</li>
<li><code>target</code>: a string that indicates the desired perk</li>
</ul>
</li>
</ul>
<h4>Output</h4>
<ul>
<li><code>perk_traversal</code>
<ul>
<li>A Python&nbsp;<code>List</code>&nbsp;that represents the order of perks that must be obtained to acquire the target perk, such that there are sufficient points to spend</li>
</ul>
</li>
</ul>
<ul>
<li><code>perk_organizer</code></li>
</ul>
<ul>
<li style="list-style-type: none;">
<ul>
<li>A <code>bool</code>&nbsp;that indicates whether the vertex <code>v</code>&nbsp;reaches the <code>target</code> perk</li>
</ul>
</li>
</ul>
<h4>Complexity</h4>
<ul>
<li>Time: <em>O(V+E)</em></li>
<li>Space: <em>O(V)</em></li>
</ul>
<h4>Examples</h4>
<ul>
<li><strong>Example 1.</strong> A <code>PerkGraph</code>&nbsp;includes the perk pair&nbsp;<code>('A', 'B')</code>
<ul>
<li><code>'B'</code>&nbsp;is the target perk, and you have 2 perk points available</li>
<li><code>['A', 'B']</code>&nbsp;should be returned</li>
</ul>
</li>
</ul>
<ul>
<li><strong>Example 2.</strong> A <code>PerkGraph</code>&nbsp;includes the perk pairs <code>('A', 'B')</code> and&nbsp;<code>(</code><code>'B', 'C')</code>
<ul>
<li><code>'C'</code>&nbsp;is the target perk, and you have 2 perk points available</li>
<li><code>[]</code>&nbsp;should be returned</li>
</ul>
</li>
<li><strong>Example 3.</strong> A <code>PerkGraph</code>&nbsp;includes the perk pairs <code>('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')</code> and&nbsp;<code>(</code><code>'E', 'F')</code>
<ul>
<li><code>'F'</code>&nbsp;is the target perk, and you have 6 perk points available</li>
<li><code>['A', 'B', 'C', 'D', 'E', 'F']</code>&nbsp;should be returned<code></code></li>
</ul>
</li>
</ul>
<h4>Guarantees</h4>
<ul>
<li><code>PerkGraph</code> will be a DAG</li>
<li>The number of vertices in a DAG will be 0 &lt;= V &lt;= 300</li>
<li>The number of edges in a DAG will be 0 &lt;= E &lt;= 299</li>
<li>Perks are&nbsp;<strong>not&nbsp;</strong>guaranteed to be in ascending order</li>
</ul>
<h4><code></code></h4>
<h2>Submission&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;</h2>
<h2><img src="https://s3.amazonaws.com/mimirplatform.production/files/77532a14-ce8b-4010-a55e-0aa2b1d63463/st%2Csmall%2C507x507-pad%2C600x600%2Cf8f8f8.u2-trimmy.jpg" alt="st,small,507x507-pad,600x600,f8f8f8.u2-trimmy.jpg" width="684" height="273" /></h2>
<h4>Deliverables</h4>
<p>Be sure to upload the following deliverables in a .zip folder to Mimir by <strong>8:00PM</strong> Eastern Time on <strong>Tuesday, November 23rd</strong>.</p>
<p>Your .zip folder can contain other files (for example, <code>description.md</code> and <code>tests.py</code>), but must include (at least) the following:</p>
<pre><code>CC7.zip
&nbsp; &nbsp; |<span class="hljs-type">&mdash; CC7</span>/
&nbsp; &nbsp; &nbsp; &nbsp; |<span class="hljs-type">&mdash; README</span>.xml &nbsp; &nbsp; &nbsp; (<span class="hljs-keyword">for</span> coding challenge feedback)
&nbsp; &nbsp; &nbsp; &nbsp; |<span class="hljs-type">&mdash; __init__</span>.py &nbsp; &nbsp; &nbsp;(<span class="hljs-keyword">for</span> proper Mimir testcase loading)
&nbsp; &nbsp; &nbsp; &nbsp; |<span class="hljs-type">&mdash; solution</span>.py &nbsp; &nbsp; &nbsp;(contains your solution source code)
</code></pre>
<h4>Grading</h4>
<p>The following 100-point rubric will be used to determine your grade on CC7:</p>
<ul>
<li>Tests (70)
<ul>
<li>00 - Coding Standard: __/5
<ul>
<li>You must complete a properly-formatted docstring to be eligible for these 5 points.</li>
</ul>
</li>
<li>01 - Test Empty: __/5</li>
<li>02 - Test Trivial: __/10</li>
<li>03 - Test Basic: __/10</li>
<li>04 - Test Non-empty: __/15</li>
<li>05 - Test Comprehensive: __/20</li>
<li>99 - Test README.xml Validity: __/5</li>
</ul>
</li>
<li>Manual (30)
<ul>
<li>M1 - Time Complexity:&nbsp;<em>O(V+E)&nbsp;</em>: __/20</li>
<li>M2 - Space Complexity:&nbsp;<em>O(V)&nbsp;</em>: __/10</li>
<li>You must pass all automated tests (excluding "00 - Coding Standard" and "99 - Test README.xml Validity") to be eligible for the 30 manual points.</li>
</ul>
</li>
</ul>
<h2>Tips, Tricks &amp; Notes</h2>
<ul>
<li>We have included an<strong> optional&nbsp;</strong>utility function for this coding challenge. This function could be used to:
<ul>
<li>visit a vertex and all of the vertices adjacent to it</li>
<li>check if the target vertex can be reached by the current vertex</li>
<li>if so, push the current vertex to the stack, which stores the path to the target vertex</li>
</ul>
</li>
<li>For further guidance, refer to the lecture contents from&nbsp;<a href="https://d2l.msu.edu/d2l/le/content/1493941/viewContent/11043994/View" target="_blank" rel="noopener noreferrer">Week 11</a>&nbsp;and these activities on <a href="https://learn.zybooks.com/zybook/MSUCSE331OnsayFall2021/chapter/27/section/11" target="_blank" rel="noopener noreferrer">Zybooks</a>.</li>
<li>Remember that all challenges are opportunities, in this course and beyond. The journey to your solution is the true reward, so make the most of it. Enjoy!</li>
</ul>
<h2>Music</h2>
<p>Feel free to listen to a Skyrim soundtrack while you code!</p>
<ul>
<li><a href="https://www.youtube.com/watch?v=nRe3xFeyhVY">General Ambiance</a></li>
<li><a href="https://www.youtube.com/watch?v=dd10InDdvJE">Tavern Music</a></li>
<li><a href="https://www.youtube.com/watch?v=aK4JSwhdcd">Night Ambiance</a></li>
</ul>
<p>Authored by Jacob Caurdy, Jordyn Rosario and Caroline Gormely</p>