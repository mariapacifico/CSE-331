<h1><strong>CC2 - FS21 - Solar Power</strong></h1>
<p><strong>Due: Tuesday, September 28th, 2021 by 8:00p ET</strong></p>
<p><em>This is not a team project, do not copy someone else&rsquo;s work.</em></p>
<h1><strong>Introduction</strong></h1>
<p><img src="https://s3.amazonaws.com/mimirplatform.production/files/2c7365a8-2175-403b-9f37-8126a9626d44/solar-farm-spotlight.jpg" alt="solar-farm-spotlight.jpg" /></p>
<p>Congratulations, you've just been hired by a solar farm startup, Solr, as a software engineer! However, before Solr can start building their solar farms, they need to find space for them. Solr has contracted some time on a really nice, really expensive satellite, and has just received images from their areas of interest. Another engineer has already preprocessed the images into simple diagrams (like the one below), each representing available, open land. Your job is to find the largest contiguous open land area so that Solr can maximize its area usage.</p>
<p><img src="https://s3.amazonaws.com/mimirplatform.production/files/d8e52360-8f9c-4349-a574-4ffe465e452d/land.png" alt="land.png" width="291" height="288" /></p>
<h1><strong>Challenge</strong></h1>
<h2><strong>Overview</strong></h2>
<p>You will create a function that takes a list of integers called <span style="text-decoration: underline;"><strong>heights</strong></span>, which is a list of the heights of each plot of the available land. It is assumed that the width of each plot is 1 and that each plot begins on the x-axis. So, in a list of [4, 5, 6, 7], plot 0 has a height of 4 and a width of 1, plot 1 has a height of 5 and a width of 1, and so on and so forth.</p>
<p>The goal of the assignment is to use the <span style="text-decoration: underline;"><strong>heights</strong></span> list to determine the largest possible <strong>rectangular</strong> area.</p>
<h4><strong>Input</strong></h4>
<ul>
<li><span style="font-weight: 400;">One python list of integers</span>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;"><span style="text-decoration: underline;"><strong>heights</strong></span>: A list of the heights of each plot of the availble land.</span>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;"><strong>Example</strong>: [3, 3, 1]</span></li>
</ul>
</li>
</ul>
</li>
</ul>
<h4><strong>Output</strong></h4>
<ul>
<li style="font-weight: 400;">A python integer that is the maximum possible rectangular area.</li>
</ul>
<h4><strong>Guarantees</strong></h4>
<ul>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">The length of the <span style="text-decoration: underline;"><strong>heights </strong></span></span><span style="font-weight: 400;">list will not exceed 500, 1 &lt;= n &lt;= 500</span></li>
<li style="font-weight: 400;">All heights will be integers. So, 4 is a valid height, but 4.2 is not.</li>
</ul>
</ul>
<p><strong>Requirements</strong></p>
<ul>
<li>Your function must be within O(nlog(n)) time complexity.</li>
</ul>
<p>&nbsp;</p>
<h4><strong>Examples:</strong></h4>
<p><em><span style="text-decoration: underline;"><span style="font-weight: 400;">Ex. 1:</span></span></em></p>
<p><img src="https://s3.amazonaws.com/mimirplatform.production/files/3e44e57e-94ab-42b9-91f8-10e97a54138d/land.png" alt="land.png" width="289" height="286" />&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<img src="https://s3.amazonaws.com/mimirplatform.production/files/50e24e86-552a-4bc8-9630-a67b26752077/land-area.png" alt="land-area.png" width="289" height="287" /></p>
<p><em><span style="font-weight: 400;"><span style="text-decoration: underline;"><strong>heights</strong></span>: </span></em>[2, 1, 5, 6, 2, 3]</p>
<p><em><span style="font-weight: 400;">Result: 10</span></em></p>
<p><span style="font-weight: 400;">In this case, the maximum area occurs between the two longest areas of land (the largest heights of the plots). It has a height of 5 and a width of 2. </span></p>
<p>&nbsp;</p>
<p><span style="text-decoration: underline;"><em>Ex. 2:</em></span></p>
<p><img src="https://s3.amazonaws.com/mimirplatform.production/files/01d4da69-5ab6-47fd-972a-0b5708341716/land2.png" alt="land2.png" width="208" height="374" /></p>
<p><em><span style="font-weight: 400;"><span style="text-decoration: underline;"><strong>heights</strong></span>: </span></em>[2, 4]</p>
<p><em><span style="font-weight: 400;">Result: 4</span></em></p>
<p>In this case, we have two different ways to maximize the area<span style="font-weight: 400;">. We can create one area of height 4 and width 1, and another of height 2 and width 2. For the purposes of this problem, it does not matter which area your function finds as the maximum, as long as it correctly determines that 4 is the maximum area.</span></p>
<p>&nbsp;</p>
<h1><strong>Submission</strong></h1>
<p><img src="https://s3.amazonaws.com/mimirplatform.production/files/fc65c52f-9bcf-48a7-9173-bb4a7523fff6/solar%20meme.jpg" alt="solar meme.jpg" width="683" height="519" /></p>
<h2><strong>Deliverables</strong></h2>
<p><span style="font-weight: 400;">Be sure to upload the following deliverables in a .zip folder to Mimir by </span><strong>8:00PM</strong><span style="font-weight: 400;"> Eastern Time on </span><strong>Tuesday, 09/28/2021</strong><span style="font-weight: 400;">.</span></p>
<p><span style="font-weight: 400;">Your .zip folder can contain other files (for example, specs.md and tests.py), but must include (at least) the following:</span></p>
<pre><span style="font-weight: 400;">CC1.zip</span><br /><span style="font-weight: 400;"> &nbsp;&nbsp;&nbsp;|&mdash; CC1/</span><br /><span style="font-weight: 400;"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&mdash; README.xml &nbsp; &nbsp; &nbsp; (for coding challenge feedback)</span><br /><span style="font-weight: 400;"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&mdash; __init__.py&nbsp; &nbsp; &nbsp; (for proper Mimir testcase loading)</span><br /><span style="font-weight: 400;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&mdash; solution.py&nbsp; &nbsp; &nbsp; (contains your solution source code)</span></pre>
<h2><strong>Grading</strong></h2>
<p><span style="font-weight: 400;">The following 100-point rubric will be used to determine your grade on CC1:</span></p>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Tests (70)</span>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">00 - Coding Standard: __/5</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">01 - Test Basic: __/10</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">02 - Test Squares: __/10</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">03 - Test Duplicates: _/10</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">05 - Test Small Comprehensive: __/15</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">06 - Test Large Comprehensive: __/15</span></li>
<li><span style="font-weight: 400;">99 - Test README.xml Validity:</span><span style="font-weight: 400;">&nbsp;__/5</span></li>
</ul>
</li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Manual (30)</span><br />
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">M1 - Time Complexity &nbsp;Expected : O(nlog(n)) or better: __/30</span>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">If you do not meet O(nlog(n)) time complexity, you are not eligible to earn any of the 30 manual points. In other words, these manual complexity points are "all-or-nothing."</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Additionally, if you do not pass&nbsp;<strong>ALL</strong> of the automated test cases, you are not eligible to earn any of the 30 manual points.</span></li>
</ul>
</li>
</ul>
</li>
</ul>
<p>&nbsp;</p>
<h1><strong>Tips, Tricks, and Notes</strong></h1>
<ul>
<li>There are two general approaches to this problem. One uses a recursive algorithm, and the other relies on a data structure that will very soon be covered in class. The recursion approach <em>may </em>be more straightforward, but it is not as fast as the data structure solution. Both, however, meet the time complexity requirements. <strong>You do NOT have to use either of these approaches</strong>.</li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Remember that all challenges are opportunities, in this course and beyond. The journey to your solution is the true reward, so make the most of it. Enjoy!</span></li>
</ul>
<p><span style="font-weight: 400;">Created by </span>Dr. Onsay, Lukas Richters, Ian Barber, and Jordyn Rosario</p>
