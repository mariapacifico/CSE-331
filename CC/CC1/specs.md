# **CC1 - FS21 - Farmer's Fencing**

**Due: Tuesday, September 21st, 2021 by 8:00p ET**

_This is not a team project, do not copy someone else’s work._

# **Introduction**

![7305559-tornado-fencing.jpg](https://s3.amazonaws.com/mimirplatform.production/files/abca575d-b683-441a-afe8-87b391b1e628/7305559-tornado-fencing.jpg)

Michigan State is a school known far and wide, and has the largest single-campus student population for any university in the entire state of Michigan. The university was founded in 1855 and has always had deep agricultural roots, so deep the name of the universitywas originally M.A.C. (Michigan Agricultural College) up until it was renamed in 1964 to "Michigan State University" as we know it today.

One of the agriculture programs at MSU has reached out to **you**, seeking help in the construction of a new livestock pen. Unfortunately, due to monetary constraints, the pen has to be constructed using existing pen posts! Additionally, the agriculture folks have asked for you to minimize the perimeter of the pen, allowing them to reduce fencing material costs further. The only constraint is that you must keep the pen rectangular.

# **Challenge**

## **Overview**

You will create a function that takes a list of tuples called <span style="text-decoration: underline;">**points**</span>, which is a list of coordinates represented by two-element tuples of integers where there is currently a fence post located. There are no fence posts available for use not listed in the <span style="text-decoration: underline;">**points**</span> list.

The goal of the assignment is to use the <span style="text-decoration: underline;">**points**</span> list to determine the smallest possible perimeter of a fence made using those points. All of the sides of the fence _must _be perpendicular to the x and y axes, that is, fencing from (0, 4) to (0, 6); or from (0, 4) to (4, 4) is okay, but fencing from (0, 6) to (4, 4) is not!

#### **Input**

*   <span style="font-weight: 400;">One python list of tuples</span>
    *   <span style="font-weight: 400;"><span style="text-decoration: underline;">**points**</span>: A list of coordinates of fence post locations, formatted as two-element tuples of integers within the list.</span>
        *   <span style="font-weight: 400;">**Example**: [(0, 4), (4, 4), (4, 0), (0, 0)]</span>

#### **Output**

*   A python integer that is the minimum possible perimeter of a rectangular pen with sides parallel to the x/y axis.

*   If no rectangular pen can be created, return 0.

#### **Guarantees**

*   <span style="font-weight: 400;">The length of the </span><span style="font-weight: 400;">**<u>points</u>** list, points, will not exceed 3000, 1 <= n <= 3000</span>
*   All fence posts will be at integer locations, that is, a point such as (4, 2) is valid but a point such as (4.5, 2.2) will not be tested.

#### **Examples:**

_<span style="text-decoration: underline;"><span style="font-weight: 400;">Ex. 1:</span></span>_

_<span style="text-decoration: underline;"><span style="font-weight: 400;">![ex1.png](https://s3.amazonaws.com/mimirplatform.production/files/0b0b4bbb-93d7-4ac9-b13e-f3208dc68e57/ex1.png)</span></span>_

_<span style="font-weight: 400;"><span style="text-decoration: underline;">**points**</span>:</span> _[(1,1),(1,3),(3,1),(3,3),(2,2)]

_<span style="font-weight: 400;">Result: 8</span>_

<span style="font-weight: 400;">Looking at Ex. 1, there is only one possible rectangle, which uses the points (1,1), (1,3), (3,1), and (3,3). Since this is the only possible rectangle, it is also the one with the smallest perimeter, so the perimeter of this rectangle, 8, is our answer.</span>

<span style="text-decoration: underline;">_Ex. 2:_</span>

<span style="text-decoration: underline;">_![ex2.png](https://s3.amazonaws.com/mimirplatform.production/files/22120df5-92bd-4a7e-a50d-2492e682c5d5/ex2.png)_</span>

_<span style="font-weight: 400;"><span style="text-decoration: underline;">**points**</span>:</span> _[(1,1),(1,3),(3,1),(3,3),(4,1),(4,3)]

_<span style="font-weight: 400;">Result: 6</span>_

Looking at Ex 2, we can make 3 rectangles - one using the same points as in example 1 - <span style="font-weight: 400;">(1,1), (1,3), (3,1), and (3,3), which has a perimeter of 8 - as well as a rectangle using the points (3,1), (3,3), (4,1), and (4,3) - which has a perimeter of 6 - or one using all external points (1,1), (1,3), (4,3), (4,1) - which has a perimeter of 10\. Since the minimum perimeter is 6, that is our answer.</span>

<span style="text-decoration: underline;">_Ex. 3:_</span>

<span style="text-decoration: underline;">_![ex3.png](https://s3.amazonaws.com/mimirplatform.production/files/aaeddadf-f756-4b52-926b-fc34a3a9ae82/ex3.png)_</span>

_<span style="font-weight: 400;"><span style="text-decoration: underline;">**points**</span>:</span> _[(1,1),(1,3)]

_<span style="font-weight: 400;">Result: 0</span>_

Looking at Ex 3, there is no possible rectangle, so we return 0.

# **Submission**

![work.jpg](https://s3.amazonaws.com/mimirplatform.production/files/ea4a6bb4-985c-43ed-8603-e7e5ddcc656d/work.jpg)

## **Deliverables**

<span style="font-weight: 400;">Be sure to upload the following deliverables in a .zip folder to Mimir by</span> **8:00PM** <span style="font-weight: 400;">Eastern Time on</span> **Tuesday, 09/21/2021**<span style="font-weight: 400;">.</span>

<span style="font-weight: 400;">Your .zip folder can contain other files (for example, specs.md and tests.py), but must include (at least) the following:</span>

<pre><span style="font-weight: 400;">CC1.zip</span>  
 <span style="font-weight: 400;">|— CC1/</span>  
 <span style="font-weight: 400;">|— README.xml       (for coding challenge feedback)</span>  
 <span style="font-weight: 400;">|— __init__.py      (for proper Mimir testcase loading)</span>  
<span style="font-weight: 400;">        |— solution.py      (contains your solution source code)</span></pre>

## **Grading**

<span style="font-weight: 400;">The following 100-point rubric will be used to determine your grade on CC1:</span>

*   <span style="font-weight: 400;">Tests (70)</span>
    *   <span style="font-weight: 400;">00 - Coding Standard: __/5</span>
    *   <span style="font-weight: 400;">01 - Test No Rectangle: __/10</span>
    *   <span style="font-weight: 400;">02 - Test One Rectangle: __/10</span>
    *   <span style="font-weight: 400;">03 - Test Multiple Rectangles: _/10</span>
    *   <span style="font-weight: 400;">05 - Test Small Comprehensive: __/15</span>
    *   <span style="font-weight: 400;">06 - Test Large Comprehensive: __/15</span>
    *   <span style="font-weight: 400;">99 - Test README.xml Validity:</span><span style="font-weight: 400;"> __/5</span>
*   <span style="font-weight: 400;">Manual (30)</span>  

    *   <span style="font-weight: 400;">M1 - Time Complexity (O(n^2)): __/30</span>
        *   <span style="font-weight: 400;">If you do not meet O(n^2) time complexity, you are not eligible to earn any of the 30 manual points. In other words, these manual complexity points are "all-or-nothing."</span>

# **Tips, Tricks, and Notes**

*   Keep the time complexity requirement in mind - theoretically, you can't have more than 2 nested loops that grow with the size of the points data set.
    *   For an extra challenge (and also as a hint), this coding challenge is possible in O(n) space complexity!
*   Python's sets ([https://www.w3schools.com/python/python_sets.asp](https://www.w3schools.com/python/python_sets.asp)) are a powerful tool you may find helpful in this coding challenge!
    *   A note on sets - lookup in a set is an O(1) operation, since it uses a concept called hashing, which we explain more later in the semester.
        *   Tuples, which are the data structure we are using to represent our points, are hashable objects, that is, you can put tuples into a set and get O(1) tuple access!
*   Note that the test cases have a feature where they use the matplotlib package to allow you to plot the points while you run the test cases. You will need to install matplotlib in order to use this feature, see intructions here ([https://matplotlib.org/stable/users/installing.html](https://matplotlib.org/stable/users/installing.html))
    *   If you do not plan to use this trick (it is not required), you can just comment out the import and function to plot at the top of the test file and disregard installation.
*   <span style="font-weight: 400;">Refer to Onsay's PyCharm setup and debugging tutorial to configure your working environment; now is the best time to get comfortable with PyCharm, as we'll be using it for the rest of the semester!</span>
*   <span style="font-weight: 400;">Remember that all challenges are opportunities, in this course and beyond. The journey to your solution is the true reward, so make the most of it. Enjoy!</span>

<span style="font-weight: 400;">Created by Ian Barber, Bank Premsri, and Lukas Richters</span>