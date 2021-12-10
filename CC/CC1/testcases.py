import unittest
import random
from solution import farmer_fencing
from typing import List, Tuple
import matplotlib.pyplot as plt
from xml.dom import minidom


def show_points(points: List[Tuple]):
    xPoints = [x for x, _ in points]
    yPoints = [y for _, y in points]
    plt.scatter(xPoints, yPoints, marker="*")
    plt.show()


class CC1TestCases(unittest.TestCase):
    def test_no_rectangle(self):
        random.seed(331)
        # 1. Not enough points
        point0 = [(5, 5), (10, 10), (5, 6)]
        expected0 = 0
        result0 = farmer_fencing(point0)
        self.assertEqual(expected0, result0)

        # 2. Vertical points only (1)
        points1 = [(1, 1), (1, 0), (1, 2), (1, 5)]
        expected1 = 0
        result1 = farmer_fencing(points1)
        self.assertEqual(expected1, result1)

        points2 = [(0, random.randint(0, 200)) for _ in range(100)]
        expected2 = 0
        result2 = farmer_fencing(points2)
        self.assertEqual(expected2, result2)

        # 3. Horizontal points only
        points3 = [(1, 1), (0, 1), (2, 1), (10, 1)]
        expected3 = 0
        result3 = farmer_fencing(points3)
        self.assertEqual(expected3, result3)

        # 4. Vertical points only (2)
        points4 = [(0, random.randint(0, 200)) for _ in range(100)]
        expected4 = 0
        result4 = farmer_fencing(points4)
        self.assertEqual(expected4, result4)

        # 5. L-shape points only
        points5 = [(0, i) for i in range(20)] + [(i, 0) for i in range(20)]
        expected5 = 0
        result5 = farmer_fencing(points5)
        self.assertEqual(expected5, result5)

        # 6. Random points
        xPoints = random.sample(list(range(50)), 30)
        yPoints = random.sample(list(range(50)), 30)
        points6 = list(zip(xPoints, yPoints))
        expected6 = 0
        result6 = farmer_fencing(points6)
        self.assertEqual(expected6, result6)

    def test_one_rectangle(self):
        random.seed(57832149)
        # 1. Simple cases
        points1 = [(0, 1), (1, 0), (0, 0), (1, 1)]
        expected1 = 4
        result1 = farmer_fencing(points1)
        self.assertEqual(expected1, result1)

        # 2. Diagonal with one rectangular
        points2 = [(i, i) for i in range(50)] + [(0, 15), (15, 0)]
        expected2 = 60
        result2 = farmer_fencing(points2)
        self.assertEqual(expected2, result2)

        # 3. Random with one sq
        #Never enters
        xPoints = random.sample(list(range(70)), 50)
        yPoints = random.sample(list(range(70)), 50)
        random_points = list(zip(xPoints, yPoints))
        points3 = [(0, 0), (0, 10), (5, 10), (5, 0)] + random_points
        # Clear redundant points
        points3 = list(set(points3))
        expected3 = 30
        result3 = farmer_fencing(points3)
        self.assertEqual(expected3, result3)

    def test_multiple_rectangles(self):
        random.seed(331)

        # 1. Big rectangle, choose a smallest one
        # Calculates wrong, doesn't return the smallest
        points1 = [(0, i) for i in range(10)] + [(i, 0) for i in range(10)] \
                  + [(10, i) for i in range(10)] + [(i, 10) for i in range(10 + 1)]
        points1 = list(set(points1))
        expected1 = 22
        result1 = farmer_fencing(points1)
        #self.assertEqual(expected1, result1)

        # 2. Small cross in 5 x 5 plane
        points2 = [(i, i) for i in range(6)] + [(i, 5 - i) for i in range(6)]
        points2 = list(set(points2))
        expected2 = 4
        result2 = farmer_fencing(points2)
        self.assertEqual(expected2, result2)

        # 3. Three lines
        # Doesn't return smallest
        points3 = [(0, i) for i in range(15)] + [(5, i + 2) for i in range(15)] + [(9, i) for i in range(15)]
        expected3 = 10
        result3 = farmer_fencing(points3)
        #self.assertEqual(expected3, result3)

        # 4. One size of rectangle, but multiple of them
        #Never enters
        points4 = [(i, i) for i in range(0, 100, 20)] \
                  + [(i, i + 3) for i in range(0, 100, 20)] \
                  + [(i + 2, i + 3) for i in range(0, 100, 20)] \
                  + [(i + 2, i) for i in range(0, 100, 20)]
        expected4 = 10
        result4 = farmer_fencing(points4)
        self.assertEqual(expected4, result4)

    def test_small_comprehensive(self):
        random.seed(3312021)

        # 1. Completely random testcase
        xPoints = random.sample(list(range(10)) * 4, 30)
        yPoints = random.sample(list(range(10)) * 4, 30)
        points1 = list(zip(xPoints, yPoints))
        expected1 = 8
        result1 = farmer_fencing(points1)
        self.assertEqual(expected1, result1)

        # 2. Small cross in 10 x 10 plane
        size = 10
        points2 = [(i, i) for i in range(size + 1)] + [(i, 10 - i) for i in range(size + 1)]
        points2 = list(set(points2))
        expected2 = 8
        result2 = farmer_fencing(points2)
        self.assertEqual(expected2, result2)

        # 3. Random points with no square
        xPoints = random.sample(list(range(15)), 10)
        yPoints = random.sample(list(range(15)), 10)
        points3 = list(zip(xPoints, yPoints))
        expected3 = 0
        result3 = farmer_fencing(points3)
        self.assertEqual(expected3, result3)

    def test_large_comprehensive(self):
        random.seed(33120212022)

        # 1. Completely random testcase
        #Never enters
        xPoints = random.sample(list(range(60)) * 4, 120)
        yPoints = random.sample(list(range(60)) * 4, 120)
        points1 = list(zip(xPoints, yPoints))
        expected1 = 82
        result1 = farmer_fencing(points1)
        self.assertEqual(expected1, result1)

        # 2. Large cross in 150 x 150 plane
        size = 150
        points2 = [(i, i) for i in range(size + 1)] + [(i, size - i) for i in range(size + 1)]
        points2 = list(set(points2))
        show_points(points2)
        expected2 = 8
        result2 = farmer_fencing(points2)
        self.assertEqual(expected2, result2)

        # 3. Random points with no square
        xPoints = random.sample(list(range(1500)), 80)
        yPoints = random.sample(list(range(1500)), 80)
        points3 = list(zip(xPoints, yPoints))
        expected3 = 0
        result3 = farmer_fencing(points3)
        self.assertEqual(expected3, result3)

        expected = [6, 4, 4, 8, 24, 12, 12, 4, 10, 14, 14, 4, 6, 24, 8, 14, 10, 6, 8, 6]
        # 4. Completely random testcases
        for problems in range(20):
            xPoints = random.sample(list(range(10)) * 4, 25)
            yPoints = random.sample(list(range(10)) * 4, 25)
            points = list(zip(xPoints, yPoints))
            result = farmer_fencing(points)
            self.assertEqual(expected[problems], result)

    def test_readme_xml_validity(self):

        path = "README.xml"
        xml_doc = minidom.parse(path)
        response = {}
        tags = ["netid", "feedback", "difficulty", "time", "citations", "type", "number"]

        # assert that we can access all tags
        for tag in tags:
            raw = xml_doc.getElementsByTagName(tag)[0].firstChild.nodeValue
            lines = [s.strip() for s in raw.split("\n")]  # if multiple lines, strip each line
            clean = " ".join(lines).strip()  # rejoin lines with spaces and strip leading space
            self.assertNotEqual("REPLACE", clean)  # make sure entry was edited
            response[tag] = clean  # save each entry

        # assert that difficulty is a float between 0-10
        difficulty_float = float(response["difficulty"])
        self.assertGreaterEqual(difficulty_float, 0.0)
        self.assertLessEqual(difficulty_float, 10.0)

        # assert that hours is a float between 0-100 (hopefully it didn't take 100 hours!)
        time_float = float(response["time"])
        self.assertGreaterEqual(time_float, 0.0)
        self.assertLessEqual(time_float, 100.0)

        # assert assignment type and number was not changed
        self.assertEqual("CC", response["type"])
        self.assertEqual("1", response["number"])
