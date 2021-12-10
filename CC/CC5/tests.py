"""
CC5 Tests
Lead: Adam Kasumovic
Assist: Zach Matson, Andrew McDonald, Sebnem Onsay
"""

import unittest
import random
import cProfile
from xml.dom import minidom
from solution import scooter_rentals
random.seed(331)


class CC5Tests(unittest.TestCase):

    def test_trivial(self):
        random.seed(331)
        # (1) Empty input case
        input = []
        expected = 0
        actual = scooter_rentals(input)
        self.assertEqual(expected, actual)

        # (2) Single input case
        input = [(1, 2)]
        expected = 1
        actual = scooter_rentals(input)
        self.assertEqual(expected, actual)

    def test_basic(self):
        # random.seed(331)
        # (1) Example input list
        input = [(0, 2), (1, 4), (4, 6), (0, 4), (7, 8), (9, 11), (3, 10)]
        expected = 3
        actual = scooter_rentals(input)
        self.assertEqual(expected, actual)

        # (2) Shuffled example input list
        random.shuffle(input)
        expected = 3
        actual = scooter_rentals(input)
        self.assertEqual(expected, actual)

        # (3) Random size 10 list
        input_left = [random.randint(1, 1000) for i in range(10)]
        input_right = [random.randint(input_left[i] + 1, input_left[i] + 999) for i in range(10)]
        input = [i for i in zip(input_left, input_right)]
        random.shuffle(input)
        expected = 9
        actual = scooter_rentals(input)
        self.assertEqual(expected, actual)

    def test_no_intersections(self):
        random.seed(331)
        # You will not earn points for this testcase by simply returning 1 when times matches these input lists.
        # This is an example of hardcoding and could lead to further penalties.

        # (1) Double input case -- no intersections
        input = [(1, 2), (2, 3)]
        expected = 1
        actual = scooter_rentals(input)
        self.assertEqual(expected, actual)

        # (2) 9 tuples
        input = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
        expected = 1
        actual = scooter_rentals(input)
        self.assertEqual(expected, actual)

        # (3) 9 shuffled tuples
        random.shuffle(input)
        expected = 1
        actual = scooter_rentals(input)
        self.assertEqual(expected, actual)

        # (4) 5 tuples whose endpoints do not touch
        input = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
        expected = 1
        actual = scooter_rentals(input)
        self.assertEqual(expected, actual)

        # (5) 5 shuffled tuples
        random.shuffle(input)
        expected = 1
        actual = scooter_rentals(input)
        self.assertEqual(expected, actual)

    def test_intersections(self):
        random.seed(331)
        # (1) Double input case -- intersection
        input = [(1, 3), (2, 3)]
        expected = 2
        actual = scooter_rentals(input)
        self.assertEqual(expected, actual)

        # (2) Only 2 intersections at a time
        input = [(1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10)]
        expected = 2
        actual = scooter_rentals(input)
        self.assertEqual(expected, actual)

        # (3) 3 intersections at a time
        input = [(1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10)]
        expected = 3
        actual = scooter_rentals(input)
        self.assertEqual(expected, actual)

        # (4) All intersections
        input = [(1, 2)] * 10
        expected = 10
        actual = scooter_rentals(input)
        self.assertEqual(expected, actual)

        # (5) All intersections with non-identical tuples, shuffled
        input = [i for i in zip([1]*10, [j for j in range(2, 12)])]
        random.shuffle(input)
        expected = 10
        actual = scooter_rentals(input)
        self.assertEqual(expected, actual)

    def test_comprehensive(self):
        random.seed(331)
        # (1) Example input list 100 times
        input = [(0, 2), (1, 4), (4, 6), (0, 4), (7, 8), (9, 11), (3, 10)] * 100
        random.shuffle(input)
        expected = 3*100
        actual = scooter_rentals(input)
        self.assertEqual(expected, actual)

        # (2) Many shuffled tuples -- no intersections
        input = [i for i in zip([j for j in range(0, 2000, 2)], [k for k in range(1, 2001, 2)])]
        random.shuffle(input)
        expected = 1
        actual = scooter_rentals(input)
        self.assertEqual(expected, actual)

        # (3) Even more shuffled tuples -- no intersections
        input = [i for i in zip([j for j in range(5000)], [k for k in range(1, 5001)])]
        random.shuffle(input)
        expected = 1
        actual = scooter_rentals(input)
        self.assertEqual(expected, actual)

        # (4) Large comprehensive intersection test
        input_left = [i for i in range(500)]
        input_right = [random.randint(input_left[i] + 1, input_left[i] + 99) for i in range(500)]
        input = [(1, 600)] * 500
        input.extend([i for i in zip(input_left, input_right)])
        random.shuffle(input)
        expected = 556
        actual = scooter_rentals(input)
        self.assertEqual(expected, actual)

        # (5) Large random comprehensive test
        input_left = [random.randint(1, 1000) for i in range(1000)]
        input_right = [random.randint(input_left[i] + 1, input_left[i] + 999) for i in range(1000)]
        input = [i for i in zip(input_left, input_right)]
        random.shuffle(input)
        expected = 503
        actual = scooter_rentals(input)
        self.assertEqual(expected, actual)

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
        self.assertEqual("5", response["number"])

    def test_cProfile(self):

        # Use cProfile to gain insight into your code's runtime complexity!
        # It is a very powerful tool for use in CSE 331 and beyond.
        #  - Change the first argument to runctx to profile different testcases
        #  - Change the sort argument to one of "calls", "cumulative", "tottime", "name", line", "module"
        # More information is available at https://docs.python.org/3/library/profile.html

        # Comment out `self.assertEqual(True, False)` and uncomment the following line to profile one of your testcases.
        # Comment out both `cProfile` and `self.assertEqual(True, False)` lines to pass this testcase.
        # cProfile.runctx("CC5Tests.test_comprehensive(self)", globals(), locals(), sort="calls")
        # self.assertEqual(True, False)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
