<h1><strong>CC3 - FS21 - Tournament Arc: CSE 331 Edition</strong></h1>
<p><strong>Due: Tuesday, October 5th, 2021 by 8:00p ET</strong></p>
<p><em>This is not a team project, do not copy someone else&rsquo;s work.</em></p>
<h1><strong>Introduction</strong></h1>
<p><strong><img style="display: block; margin-left: auto; margin-right: auto;" src="https://s3.amazonaws.com/mimirplatform.production/files/d9cbed9a-8b48-4a03-adf4-e8e5824ba69d/over%209000.gif" alt="over 9000.gif" width="715" height="379" /></strong></p>
<p>&nbsp;</p>
<p>It has been your lifelong dream to be reincarnated in an *isekai* (a subgenre of fantasy in which a character is suddenly transported from their world into a new or unfamiliar one)...</p>
<p>Lo and behold, your dream comes true...</p>
<p>You wake up. You thought you were in the crossover you always wanted but it was the one you deserved. You're finally in an anime world, but you're not in 2D??? What's more, you don't even have special powers?? It looks like you're playing the role of a scorekeeper in some weird tournament...</p>
<p>To your confusion, the lines of anime characters around you fire a punch into the air in sequence, one after another. If you know anything about anime, you know that once a punch is fired, immense air pressure is generated. This world that you're in had its physics coded up by an overworked and underpaid programmer, so punches from characters with higher power levels just continue to travel through those from characters with equal or lower power levels, while lower ones stop the moment they encounter a higher one. As scorekeeper, you've been told that the performance of a participant is simply the sum of all power levels that their punch passes through.&nbsp;</p>
<p>The logic behind the rules of this tournament is straight out of anime. (Why do they need you to be logistician when they have Light Yagami around?? Well, it is anime logic so don't ask too many questions 😃)&nbsp; Nevertheless, it turns out you can only go back to your world once you submit an implementation of an algorithm to determine the performance of all the participants in this tournament, measured according to the rules described above. While time does move weird in anime, you do have a deadline and it just so happens to coincide with the deadline of this Coding Challenge 😉.</p>
<p><strong><img class="n3VNCb" src="https://wallpaperaccess.com/full/958907.jpg" alt="Isaac Netero Wallpapers - Top Free Isaac Netero Backgrounds -  WallpaperAccess" /></strong></p>
<pre>&nbsp;</pre>
<h1><strong>Challenge</strong></h1>
<h2><strong>Overview</strong></h2>
<p>Your objective will be to write a function that takes in a participant list that consists of names in English and/or Japanese, along with a second list that contains reference tuples of participant names in English and Japanese and the associated participant's power level. You will return a list of tuples which contains the name and score of each participant.</p>
<p>A punch fired by a character <strong>will travel forward until it encounters a character whose power level is greater than the character's own</strong>. A given character's score is the sum of the power levels of all the characters that their punch passes through. You need to calculate the score of every character in the supplied list, and return the name and score of every character in a list of tuples.</p>
<p><em>Modify the following function</em></p>
<p><strong>calculate_results(participants: List[str], character_details: List[Tuple[str, str, int]]) -&gt; List[Tuple[str, int]]</strong></p>
<ul>
<li><strong>participants</strong><span style="font-weight: 400;">: A Python list of length <strong>n</strong> that contains the names of each participant in the tournament in the order they are lined up. May contain duplicates.</span></li>
<li><strong>details</strong>: A Python list of length <strong>m</strong> containing tuples, where each tuple contains the English and Japanese name of each character, along with the power level of the associated character. The <strong>power level will always be at index 2</strong> in the tuple, and the <strong>English and Japanese names will be at index 0 and 1, in random order</strong>. There will be no duplicates in this list. The English name will always start with an uppercase ASCII English letter and the Japanese name will never start with an ASCII character.</li>
<li><strong>Return:</strong><span style="font-weight: 400;"> A Python list of tuples that contains the name and score of each participant at<strong> index 0 and 1 of the tuple</strong>&nbsp;<strong>respectively</strong>. If a participant's power level is over 9000, their name should be returned in Japanese. Otherwise, it should be returned in English. The participants should be in the same order as the input list.</span></li>
<li style="font-weight: 400;"><strong>Time Complexity:</strong><span style="font-weight: 400;"> O(n + m).</span></li>
</ul>
<p>&nbsp;</p>
<h4><strong>Examples:</strong></h4>
<p><strong>Example 1</strong></p>
<p><strong>participants </strong>= ['Lamperouge, Lelouch', 'Roronoa, Zoro', 'Okabe, Rintarou', 'Monkey D., Luffy', 'Lawliet, L']</p>
<p><strong>details&nbsp;</strong>= [</p>
<div>
<div>&nbsp; &nbsp; ('Lamperouge, Lelouch', 'ルルーシュ・ランペルージ', 6503),</div>
<div>&nbsp;&nbsp;&nbsp;&nbsp;('エル&nbsp;ローライト',&nbsp;'Lawliet,&nbsp;L',&nbsp;8101),</div>
<div>&nbsp;&nbsp;&nbsp;&nbsp;('Monkey&nbsp;D.,&nbsp;Luffy',&nbsp;'モンキー・D・ルフィ',&nbsp;3828),</div>
<div>&nbsp;&nbsp;&nbsp;&nbsp;('Okabe,&nbsp;Rintarou',&nbsp;'岡部&nbsp;倫太郎',&nbsp;6910),</div>
<div>&nbsp;&nbsp;&nbsp;&nbsp;('Roronoa, Zoro', 'ロロノア・ゾロ', 8018)</div>
<div>]</div>
</div>
<p>It helps to first convert the participants list into a list of power levels since they are what we are actually comparing.</p>
<p><strong>power_levels_list = [6503, 8018, 6910, 3828, 8101]</strong></p>
<p>Since 6503 &lt; 8018, Lelouch's punch does not pass through anyone, making his score <strong>0.</strong></p>
<p>Since 8018 &gt; 6910 and 8018 &gt; 3828 but 8018 &lt; 8101, Zoro's punch passes through 3828 (Luffy) and 6910 (Rintarou), but stops at L making his score 3828 + 6910 = <strong>10738.</strong></p>
<p>6910 &gt; 3828 but 6910 &lt; 8101 so the score is <strong>3828</strong> for Rintarou.</p>
<p>3828 &lt; 8101 so like Lelouch, Luffy gets a score of <strong>0</strong>.</p>
<p>There is no one in front of L, so he automatically gets a score of <strong>0</strong>.</p>
<p><strong>output </strong>= [</p>
<p>('Lamperouge, Lelouch', 0),</p>
<p>('Roronoa, Zoro', 10738),</p>
<p>('Okabe, Rintarou', 3828),</p>
<p>('Monkey D., Luffy', 0),</p>
<p>('Lawliet, L', 0)</p>
<p>]</p>
<p><strong>Example 2</strong></p>
<p><strong>participants </strong>= ['Lamperouge, Lelouch', 'Levi', '岡部 倫太郎', 'Monkey D., Luffy', 'Lawliet, L']</p>
<p><strong>details&nbsp;</strong>= [</p>
<p><span style="background-color: transparent;">&nbsp;('Lamperouge, Lelouch', 'ルルーシュ・ランペルージ', 6503),</span></p>
<div>
<div>&nbsp;('エル ローライト', 'Lawliet, L', 8101),</div>
<div>&nbsp;</div>
<div>('Monkey D., Luffy', 'モンキー・D・ルフィ', 3828),</div>
<div>&nbsp;</div>
<div>('Okabe, Rintarou', '岡部 倫太郎', 6910),</div>
<div>&nbsp;</div>
<div>
<div>
<div>
<div>('リヴァイ', 'Levi', 9848)</div>
</div>
</div>
<div>&nbsp;</div>
</div>
<div>]</div>
</div>
<p>Again, it helps to first convert the participants list into a list of power levels since they are what we are actually comparing.</p>
<p><strong>power_levels_list = [6503, 9848, 6910, 3828, 8101]</strong></p>
<p>Since 6503 &lt; 9848, Lelouch's punch does not pass through anyone, making his score <strong>0.</strong></p>
<p>Since 9848 is greater than everything in front of it, Levi's punch passes through everyone in front of him making his score 6910 + 3828 + 8101 = <strong>18839.&nbsp;</strong>Levi's power level is also more than 9000 so, his name must be returned in Japanese.</p>
<p>6910 &gt; 3828 but 6910 &lt; 8101 so the score is <strong>3828</strong> for 倫太郎 (Rintarou).</p>
<p>3828 &lt; 8101 so like Lelouch, Luffy gets a score of <strong>0</strong>.</p>
<p>There is no one in front of L, so he automatically gets a score of <strong>0</strong>.</p>
<p><strong>output </strong>= [</p>
<p>('Lamperouge, Lelouch', 0),</p>
<p>('<span style="background-color: transparent;">リヴァイ</span>', 18839),</p>
<p>('Okabe, Rintarou', 3828),</p>
<p>('Monkey D., Luffy', 0),</p>
<p>('Lawliet, L', 0)</p>
<p>]</p>
<h1><strong>Submission</strong></h1>
<p><strong><img class="n3VNCb" src="https://media0.giphy.com/media/arbHBoiUWUgmc/200.gif" alt="One Punch Man GIFs - Get the best GIF on GIPHY" width="693" height="387" /></strong></p>
<h2><strong>Deliverables</strong></h2>
<p><span style="font-weight: 400;">Be sure to upload the following deliverables in a .zip folder to Mimir by </span><strong>8:00PM</strong><span style="font-weight: 400;"> Eastern Time on </span><strong>Tuesday, 10/05/2021</strong><span style="font-weight: 400;">.</span></p>
<p><span style="font-weight: 400;">Your .zip folder can contain other files (for example, specs.md and tests.py), but must include (at least) the following:</span></p>
<pre><span style="font-weight: 400;">CC3.zip</span><br /><span style="font-weight: 400;"> &nbsp;&nbsp;&nbsp;|&mdash; CC3/</span><br /><span style="font-weight: 400;"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&mdash; README.xml &nbsp; &nbsp; &nbsp; (for coding challenge feedback)</span><br /><span style="font-weight: 400;"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&mdash; __init__.py&nbsp; &nbsp; &nbsp; (for proper Mimir testcase loading)</span><br /><span style="font-weight: 400;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&mdash; solution.py&nbsp; &nbsp; &nbsp; (contains your solution source code)</span></pre>
<h2><strong>Grading</strong></h2>
<p><span style="font-weight: 400;">The following 100-point rubric will be used to determine your grade on CC3:</span></p>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Tests (70)</span>
<ul>
<li style="font-weight: 400;">00 - Coding Standard: __/5</li>
<li style="font-weight: 400;"><span style="font-weight: 400;">01 &nbsp;- Test monotone: __/5</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">02 - Test pyramid: __/10</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">03 - Test intermediate: __/10</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">04 - Test over 9000: __/10</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">05 - Test mix input language: __/10</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">06 - Test comprehensive: __/15</span></li>
<li><span style="font-weight: 400;">99 - Test README.xml Validity:</span><span style="font-weight: 400;">&nbsp;__/5</span></li>
</ul>
</li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Manual (30)</span><br />
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">M1 - Time Complexity (O(n + m)): __/30</span>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">If you do not meet O(n + m) time complexity, you are not eligible to earn any of the 30 manual points. In other words, these manual complexity points are "all-or-nothing."</span></li>
<li>If you do not pass <strong>ALL</strong> of the automated test cases, you are not eligible to earn any of the 30 manual points.</li>
</ul>
</li>
</ul>
</li>
</ul>
<h1><strong>Tips, Tricks, and Notes</strong></h1>
<ul>
<li style="font-weight: 400;">Python's dictionary (<a href="https://www.w3schools.com/python/python_dictionaries.asp">https://www.w3schools.com/python/python_dictionaries.asp</a>) is a powerful tool offering O(1) lookup that you may find helpful in this coding challenge!</li>
<li style="font-weight: 400;">Stacks are a helpful data structure for this problem.</li>
<li style="font-weight: 400;">This problem has 2 distinct parts: data parsing (preprocessing names and power levels to a common form) and algorithmic implementation (computing scores given power levels). Separating the two is recommended.</li>
<li style="font-weight: 400;">Figuring out whether a string is in Japanese is an important part of this problem. Note that the English name will always start with an uppercase ASCII English letter and the Japanese name will never start with an ASCII character.</li>
<li>Remember that all challenges are opportunities, in this course and beyond. The journey to your solution is the true reward, so make the most of it. Enjoy!</li>
</ul>
<p><span style="font-weight: 400;">Created by Bank Premsri, Joseph Pallipadan, Zach Matson</span></p>