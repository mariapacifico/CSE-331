CC6 - FS21 - A Cut Above
========================

**Due: Tuesday, November 16th, 2021 by 8:00p ET**

_This is not a team project. Do not copy someone else’s work._

Introduction
------------

Congratulations! You've just been recruited to join the security team for the most exclusive party on the first Monday in May. As you know, the [Met Gala](https://en.wikipedia.org/wiki/Met_Gala) is one of the most high-profile events of the year. [Anna Wintour](https://en.wikipedia.org/wiki/Anna_Wintour)'s annual Ball hosts hundreds of celebrities as a fundraiser to benefit the [Costume Institute](https://en.wikipedia.org/wiki/Anna_Wintour_Costume_Center) at the Metropolitan Museum of Arts in New York City.

The night before the event, you are alerted by reputable sources that a successful group of criminals has big plans for a jewelry heist in the [Cartier](https://www.cartier.com/en-us/home) exhibit. This year, the Met is featuring a collection of the most magnificent and expensive jewelry to date. As luck would have it, you are responsible for the security of the most expensive piece in the museum: the Onsay. The Onsay is an imperial necklace comprised of none other than the finest (ethically sourced) diamonds in the world.

You immediately notify Mrs. Wintour about the problem, but since the Met Ball is such a highly publicized event,  she does not want to cause a commotion for the guests. Instead, you are left to come up with something more inconspicuous. You decide that you are going to let the criminals carry out their scheme, but not before you replace the necklace with a fake one!

As a CSE 331 student, you recognize this as a graph problem that requires traversal of each of the jewels in a necklace. After all, you'll need to know which jewels go where in your replica so that the thieves do no suspect anything.

In this coding challenge, you'll be implementing that algorithm with an efficient _O(V+E)_ and _O(V)_ time and space complexity, respectively.

Challenge
---------

#### Overview

Given a starting bead in a necklace, return a duplicate of the necklace bead, such that it can be traversed to create an exact replica of the original necklace.

The `NecklaceNode` class includes an attribute `adj`, which stores a `list` of adjacent `NecklaceNode` objects.

Your task is to complete the `ReplicateNecklace` function with a time complexity of _O(V+E)_ and space complexity of _O(V)_.

![diamonds.webp](https://s3.amazonaws.com/mimirplatform.production/files/8b8ffb03-9d9d-4b49-a190-665d2417e8df/diamonds.webp)

#### Function Signature

`ReplicateNecklace(start: NecklaceNode) -> NecklaceNode:`

#### Input

*   A `NecklaceNode` object that happens to be the starting point of the necklace that we will replicate

#### Output

*   A duplicate `NecklaceNode` that is the starting point of the new necklace, which should be an exact replica of the original. All NecklaceNodes pointed at by the new NecklaceNode should also be duplicates of their respective original Necklace.

#### Complexity

*   Time: _O(V+E)_
*   Space: _O(V)_

#### Examples

*   **Example 1.** Given input `NecklaceNode(1)`, return `NecklaceNode(1)`
    *   This necklace includes two additional `NecklaceNodes`, organized in a ring pattern
    *   The adjacency lists for all three `NecklaceNode` must mirror the originals:
    *   `NecklaceNode(1).adj` = \[`NecklaceNode(2),``NecklaceNode(3)``]`
    *   `NecklaceNode(2).adj` = \[`NecklaceNode(3),``NecklaceNode(1)``]`
    *   `NecklaceNode(3).adj` = \[`NecklaceNode(1),``NecklaceNode(2)``]`
*   **Example 2.** Given input `NecklaceNode(1)`, return `NecklaceNode(1)`
    *   This necklace includes four additional `NecklaceNodes`, organized in a cross pattern
    *   The adjacency lists for all five `NecklaceNode` must mirror the originals:
    *   `NecklaceNode(1).adj` = \[`NecklaceNode(5)``]`
    *   `NecklaceNode(2).adj` = \[`NecklaceNode(5)``]`
    *   `NecklaceNode(3).adj` = \[`NecklaceNode(5)]`
    *   `NecklaceNode(4).adj` = \[`NecklaceNode(5)]`
    *   `NecklaceNode(5).adj` = `[NecklaceNode(1),``NecklaceNode(2),``NecklaceNode(3),``NecklaceNode(4)]`

#### Guarantees

*   `1 <= V <= 100`
*   The input `start` is guaranteed to be non empty
*   A necklace is **not** guaranteed to be ring-shaped
*   A necklace is **not** guaranteed to be a single string of diamonds

Submission
----------

#### Deliverables

Be sure to upload the following deliverables in a .zip folder to Mimir by **8:00PM** Eastern Time on **Tuesday, November 16th**.

Your .zip folder can contain other files (for example, `description.md` and `tests.py`), but must include (at least) the following:

    CC6.zip
        |— CC6/
            |— README.xml       (for coding challenge feedback)
            |— __init__.py      (for proper Mimir testcase loading)
            |— solution.py      (contains your solution source code)
    

#### Grading

The following 100-point rubric will be used to determine your grade on CC6:

*   Tests (70)
    *   00 - Coding Standard: \_\_/5
        *   You must complete a properly-formatted docstring to be eligible for these 5 points.
    *   01 - Test Trivial: \_\_/5
    *   02 - Test Basic 1: \_\_/10
    *   03 - Test Basic 2: \_\_/10
    *   04 - Test Complex 1: \_\_/10
    *   05 - Test Complex 2: \_\_/10
    *   06 - Test Comprehensive: \_\_/15
    *   99 - Test README.xml Validity: \_\_/5
*   Manual (30)
    *   M1 - Time Complexity: _O(V+E)_ : \_\_/20
    *   M2 - Space Complexity: _O(V)_ : \_\_/10
    *   You must pass all automated tests (excluding "00 - Coding Standard" and "99 - Test README.xml Validity") to be eligible for the 30 manual points.

Tips, Tricks & Notes
--------------------

*   We **strongly suggest** using a BFS or DFS traversal to solve this problem!
    *   You cannot simply return the original starting point of the necklace
    *   You must be able to use the starting jewel in such a way that you can traverse the necklace and create an exact replica of the original
    *   That is, the jewels must follow the original structure of the necklace (they have an adjacency list of duplicate jewels), we are not looking for duplicates of just the jewels
*   For further guidance, refer to the [Week 11](https://d2l.msu.edu/d2l/le/content/1493941/Home) lecture contents from D2L and the [Chapter 27](https://learn.zybooks.com/zybook/MSUCSE331OnsayFall2021/chapter/27/section/3) contents on Zybooks.
*   Remember that all challenges are opportunities, in this course and beyond. The journey to your solution is the true reward, so make the most of it. Enjoy!

Authored by Jacob Caurdy, Jordyn Rosario, Alexander Woodring, and Sebnem Onsay