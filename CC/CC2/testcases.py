"""
CC2 Testcases
Created by: Dr. Onsay, Lukas Richters, Ian Barber, and Jordyn Rosario
"""

import unittest
import random
from solution import solar_power
from xml.dom import minidom

class CC2TestCases(unittest.TestCase):

    def test_basic(self):
        """
        Test algorithm on basic cases.
        """
        # 1) Plot with 3 bars
        heights0 = [3,3,1]
        expected0 = 6
        actual0 = solar_power(heights0)
        self.assertEqual(expected0,actual0)
        # 2) Plot with 4 bars
        heights1 = [2,4,6,8]
        expected1 = 12
        actual1 = solar_power(heights1)
        self.assertEqual(expected1,actual1)
        # 3) Plot with 5 bars
        heights2 = [1,3,5,7,9]
        expected2 = 15
        actual2 = solar_power(heights2)
        self.assertEqual(expected2,actual2)
        # 4) Plot with 6 bars
        heights3 = [2,1,5,6,2,3]
        expected3 = 10
        actual3 = solar_power(heights3)
        self.assertEqual(expected3,actual3)
        # 5) Plot with 7 bars
        heights4 = [6,2,5,4,5,1,6]
        expected4 = 12
        actual4 = solar_power(heights4)
        self.assertEqual(expected4,actual4)
        # 6) Plot with 8 bars
        heights5 = [1,2,4,5,3,3,2,4]
        expected5 = 14
        actual5 = solar_power(heights5)
        self.assertEqual(expected5,actual5)

    def test_squares(self):
        """
        Test algorithm on rectangles that also 
        happen to be squares.
        """
        # 1) 3x3 square
        heights0 = [2,5,3,3]
        expected0 = 9
        actual0 = solar_power(heights0)
        self.assertEqual(expected0,actual0)
        # 2) 4x4 square
        heights1 = [1,2,3,5,4,8,4]
        expected1 = 16
        actual1 = solar_power(heights1)
        self.assertEqual(expected1,actual1)
        # 3) 5x5 square
        heights2 = [7,6,5,6,7]
        expected2 = 25
        actual2 = solar_power(heights2)
        self.assertEqual(expected2,actual2)
        # 4) 6x6 square
        heights3 = [5,3,6,11,13,9,6,7,4,2]
        expected3 = 36
        actual3 = solar_power(heights3)
        self.assertEqual(expected3,actual3)

    def test_duplicates(self):
        """
        Test algorithm on plots that have more
        than one way to maximize area.
        """
        # 1) 1x4 or 2x2 rectangles
        heights0 = [2,4]
        expected0 = 4
        actual0 = solar_power(heights0)
        self.assertEqual(expected0,actual0)
        # 2) Two 2x4 rectangles
        heights1 = [2,4,4,2]
        expected1 = 8
        actual1 = solar_power(heights1)
        self.assertEqual(expected1,actual1)
        # 3) Two 3x5 rectangles
        heights2 = [3,5,7,6,2,4]
        expected2 = 15
        actual2 = solar_power(heights2)
        self.assertEqual(expected2,actual2)
        # 4) Two 6x7 rectangles
        heights3 = [4,14,6,10,15,12,6,8,3,5,2,7,7,7,10,8,9]
        expected3 = 42
        actual3 = solar_power(heights3)
        self.assertEqual(expected3,actual3)

    def test_small_comprehensive(self):
        """
        Test algorithm on small, complex cases.
        """
        random.seed(331)
        # 1) Randomly generated plot with 25 bars
        heights0 = [random.randint(3,31) for i in range(25)]
        expected0 = 195
        actual0 = solar_power(heights0)
        self.assertEqual(expected0,actual0)
        # 2) Randomly generated plot with 50 bars
        heights1 = random.sample(range(100), 25)
        expected1 = 324
        actual1 = solar_power(heights1)
        self.assertEqual(expected1,actual1)
        # 3) Randomly generated plot with 50 bars
        heights2 = [random.randint(3,31) for i in range(50)]
        expected2 = 176
        actual2 = solar_power(heights2)
        self.assertEqual(expected2,actual2)
        # 4) Randomly generated plot with 50 bars
        heights3 = random.sample(range(331), 50)
        expected3 = 2128
        actual3 = solar_power(heights3)
        self.assertEqual(expected3,actual3)

    def test_large_comprehensive(self):
        """
        Test algorithm on large, complex cases.
        """
        random.seed(333111)
        # 1) Randomly generated plot with 200 bars (1)
        heights0 = random.sample(range(331), 200)
        expected0 = 1440
        actual0 = solar_power(heights0)
        self.assertEqual(expected0,actual0)
        # 2) Randomly generated plot with 200 bars (2)
        heights1 = random.sample(range(870), 200)
        expected1 = 5390
        actual1 = solar_power(heights1)
        self.assertEqual(expected1,actual1)
        # 3) Randomly generated plot with 331 bars (1)
        heights2 = random.sample(range(444), 331)
        expected2 = 2544
        actual2 = solar_power(heights2)
        self.assertEqual(expected2,actual2)
        # 4) Randomly generated plot with 331 bars (2)
        heights3 = [random.randint(9,28) for i in range(331)]
        expected3 = 2979
        actual3 = solar_power(heights3)
        self.assertEqual(expected3,actual3)
        # 5) Randomly generated plot with 500 bars (1)
        heights4 = [random.randint(3,31) for i in range(500)]
        expected4 = 1500
        actual4 = solar_power(heights4)
        self.assertEqual(expected4,actual4)
        # 6) Randomly generated plot with 500 bars (2)
        heights5 = [random.randint(1,1111) for i in range(500)]
        expected5 = 5590
        actual5 = solar_power(heights5)
        self.assertEqual(expected5,actual5)

    def test_readme_xml_validity(self):

        path = "README.xml"
        xml_doc = minidom.parse(path)
        response = {}
        tags = ["netid", "feedback", "difficulty", "time", "citations", "type", "number"]

        # Assert that we can access all tags
        for tag in tags:
            raw = xml_doc.getElementsByTagName(tag)[0].firstChild.nodeValue
            lines = [s.strip() for s in raw.split("\n")]  # If multiple lines, strip each line
            clean = " ".join(lines).strip()  # Rejoin lines with spaces and strip leading space
            self.assertNotEqual("REPLACE", clean)  # Make sure entry was edited
            response[tag] = clean  # Save each entry

        # Assert that difficulty is a float between 0-10
        difficulty_float = float(response["difficulty"])
        self.assertGreaterEqual(difficulty_float, 0.0)
        self.assertLessEqual(difficulty_float, 10.0)

        # Assert that hours is a float between 0-100 (hopefully it didn't take 100 hours!)
        time_float = float(response["time"])
        self.assertGreaterEqual(time_float, 0.0)
        self.assertLessEqual(time_float, 100.0)

        # Assert assignment type and number was not changed
        self.assertEqual("CC", response["type"])
        self.assertEqual("2", response["number"])
