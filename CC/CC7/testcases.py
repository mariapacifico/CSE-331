import random
import string
import unittest
from xml.dom import minidom
from solution import PerkGraph


class CC7TestCases(unittest.TestCase):

    def test_empty(self):
        # (1a) Empty graph, 0 points
        g = PerkGraph()
        expected = []
        actual = g.perk_traversal(' ', 0)
        self.assertEqual(expected, actual)

        # (1b) Empty graph
        g = PerkGraph()
        expected = []
        actual = g.perk_traversal('F')
        self.assertEqual(expected, actual)

    def test_trivial(self):
        # (1a) 2 vertices, 1 point
        g = PerkGraph()
        g.add_edge('A', 'B')
        expected = ['A']
        actual = g.perk_traversal('A', 1)
        self.assertEqual(expected, actual)

        # (1b) 2 vertices, 2 points
        expected = ['A', 'B']
        actual = g.perk_traversal('B', 2)
        self.assertEqual(expected, actual)

        # (2a) 3 vertices
        g.add_edge('B', 'C')
        expected = ['A', 'B']
        actual = g.perk_traversal('B')
        self.assertEqual(expected, actual)

        # (2b) 3 vertices, 3 points
        expected = ['A', 'B', 'C']
        actual = g.perk_traversal('C', 3)
        self.assertEqual(expected, actual)

    def test_basic(self):
        # (1) 4 vertices,  points
        g = PerkGraph()
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'D')
        expected = ['A', 'B', 'C']
        actual = g.perk_traversal('C', 3)
        self.assertEqual(expected, actual)

        # (2a) 6 vertices,
        g.add_edge('D', 'E')
        g.add_edge('E', 'F')
        expected = ['A', 'B', 'C', 'D', 'E', ]
        actual = g.perk_traversal('E', 6)
        self.assertEqual(expected, actual)

        # (2b) 6 vertices
        expected = ['A', 'B', 'C', 'D', 'E', 'F']
        actual = g.perk_traversal('F', 6)
        self.assertEqual(expected, actual)

        # (3a) 10 vertices
        g.add_edge('F', 'G')
        g.add_edge('G', 'H')
        g.add_edge('H', 'I')
        g.add_edge('I', 'J')
        expected = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        actual = g.perk_traversal('H')
        self.assertEqual(expected, actual)

        # (3b) 10 vertices
        expected = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        actual = g.perk_traversal('I', 12)
        self.assertEqual(expected, actual)

        # (3c) 10 vertices
        expected = []
        actual = g.perk_traversal('J', 6)
        self.assertEqual(expected, actual)

    def test_nonempty(self):
        # (1a) 12 vertices
        g = PerkGraph()
        g.add_edge('L', 'A')
        g.add_edge('K', 'G')
        g.add_edge('U', 'R')
        g.add_edge('T', 'S')
        g.add_edge('E', 'U')
        g.add_edge('A', 'K')
        g.add_edge('D', 'J')
        g.add_edge('G', 'P')
        g.add_edge('S', 'L')
        g.add_edge('J', 'E')
        g.add_edge('P', 'D')
        expected = ['T']
        actual = g.perk_traversal('T')
        self.assertEqual(expected, actual)

        # (1b) 12 vertices
        expected = ['T', 'S']
        actual = g.perk_traversal('S', 2)
        self.assertEqual(expected, actual)

        # (1c) 12 vertices
        expected = ['T', 'S', 'L']
        actual = g.perk_traversal('L', 7)
        self.assertEqual(expected, actual)

        # (1d) 12 vertices
        expected = ['T', 'S', 'L', 'A', 'K', 'G', 'P']
        actual = g.perk_traversal('P', 331)
        self.assertEqual(expected, actual)

        # (1e) 12 vertices
        expected = []
        actual = g.perk_traversal('E', 5)
        self.assertEqual(expected, actual)

        # (1f) 12 vertices
        expected = ['T', 'S', 'L', 'A', 'K', 'G', 'P', 'D', 'J', 'E', 'U', 'R']
        actual = g.perk_traversal('R', 20)
        self.assertEqual(expected, actual)

        # (2a) 14 vertices
        g.add_edge('R', 'C')
        g.add_edge('C', 'W')
        expected = ['T', 'S', 'L', 'A']
        actual = g.perk_traversal('A', 5)
        self.assertEqual(expected, actual)

        # (2b) 14 vertices
        expected = ['T', 'S', 'L', 'A', 'K', 'G', 'P', 'D', 'J']
        actual = g.perk_traversal('J', 22)
        self.assertEqual(expected, actual)

        # (2c) 14 vertices
        expected = []
        actual = g.perk_traversal('C', 9)
        self.assertEqual(expected, actual)

        # (2d) 14 vertices
        expected = ['T', 'S', 'L', 'A', 'K', 'G', 'P', 'D', 'J', 'E', 'U', 'R', 'C', 'W']
        actual = g.perk_traversal('W', 25)
        self.assertEqual(expected, actual)

    def test_comprehensive(self):
        random.seed(1989)

        # Generate perks
        alphabet = list(string.ascii_uppercase)
        perks = []
        for letter in alphabet:
            for i in range(len(alphabet)):
                perks.append(letter + ''.join(alphabet[i]))

        # (1a) 100 vertices
        g = PerkGraph()
        vertices = [i for i in random.sample(perks, 100)]
        for i in range(1, len(vertices)):
            g.add_edge(vertices[i - 1], vertices[i])

        expected = []
        actual = g.perk_traversal(vertices[i], 99)
        self.assertEqual(expected, actual)

        # (1b) 100 vertices
        g = PerkGraph()
        vertices = [i for i in random.sample(perks, 100)]
        for i in range(1, len(vertices)):
            g.add_edge(vertices[i - 1], vertices[i])

        for i in range(len(vertices)):
            expected = []
            for j in range(i + 1):
                expected.append(vertices[j])
            actual = g.perk_traversal(vertices[i], 111)
            self.assertEqual(expected, actual)

        # (2a) 200 vertices
        g = PerkGraph()
        vertices = [i for i in random.sample(perks, 200)]
        for i in range(1, len(vertices)):
            g.add_edge(vertices[i - 1], vertices[i])

        expected = []
        actual = g.perk_traversal(vertices[i], 6)
        self.assertEqual(expected, actual)

        # (2b) 200 vertices
        g = PerkGraph()
        vertices = [i for i in random.sample(perks, 200)]
        for i in range(1, len(vertices)):
            g.add_edge(vertices[i - 1], vertices[i])

        for i in range(len(vertices)):
            expected = []
            for j in range(i + 1):
                expected.append(vertices[j])
            actual = g.perk_traversal(vertices[i], 222)
            self.assertEqual(expected, actual)

        # (2c) 200 vertices
        g = PerkGraph()
        vertices = [i for i in random.sample(perks, 200)]
        for i in range(1, len(vertices)):
            g.add_edge(vertices[i - 1], vertices[i])

        for i in range(len(vertices)):
            expected = []
            for j in range(i + 1):
                expected.append(vertices[j])
            actual = g.perk_traversal(vertices[i], 331)
            self.assertEqual(expected, actual)

        # (1a) 300 vertices
        g = PerkGraph()
        vertices = [i for i in random.sample(perks, 300)]
        for i in range(1, len(vertices)):
            g.add_edge(vertices[i - 1], vertices[i])

        for i in range(len(vertices)):
            expected = []
            for j in range(i + 1):
                expected.append(vertices[j])
            actual = g.perk_traversal(vertices[i], 331)
            self.assertEqual(expected, actual)

    def test_README(self):
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
        self.assertEqual("7", response["number"])
