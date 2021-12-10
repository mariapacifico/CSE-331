import random
import unittest
from xml.dom import minidom
from solution import NecklaceNode, ReplicateNecklace

class CC6TestCases(unittest.TestCase):

    def test_trivial(self):
        # A single jewel on the necklace
        jewel1 = NecklaceNode(1, [])
        # Make the fake jewel
        new_jewel1 = ReplicateNecklace(jewel1)
        # Testing to make sure the fake jewel is in place
        self.assertTrue(str(jewel1) == str(new_jewel1))
        self.assertFalse(jewel1 is new_jewel1)

    def test_basic1(self):

        # A single jewel on a pendant.
        jewel1 = NecklaceNode(1)
        jewel1.adj = [jewel1]
        # Make the fake necklace
        new_jewel1 = ReplicateNecklace(jewel1)
        # Make sure the fake jewel is in place
        expected = str(jewel1)
        actual = str(new_jewel1)
        self.assertEqual(expected, actual)
        self.assertFalse(jewel1 is new_jewel1)

        # Unclipped bracelet -- 2 jewels, in a line
        jewel1 = NecklaceNode(1)
        jewel2 = NecklaceNode(2)
        jewel1.adj = [jewel2]
        jewel2.adj = [jewel1]
        new_jewel1 = ReplicateNecklace(jewel1)
        new_jewel2 = new_jewel1.adj[0]

        # make sure fake jewels are in place
        # Check 1
        expected = str(jewel1)
        actual = str(new_jewel1)
        self.assertEqual(expected, actual)
        self.assertFalse(jewel1 is new_jewel1)

        # check 2
        expected = str(jewel2)
        actual = str(new_jewel2)
        self.assertEqual(expected, actual)
        self.assertFalse(jewel2 is new_jewel2)

    def test_basic2(self):
        # Simple necklace of 3 jewels on a loop.
        jewel1 = NecklaceNode(1)
        jewel2 = NecklaceNode(2)
        jewel3 = NecklaceNode(3)
        jewel1.adj = [jewel2, jewel3]
        jewel2.adj = [jewel1, jewel3]
        jewel3.adj = [jewel1, jewel2]
        # Make the necklace with fake jewels
        new_jewel1 = ReplicateNecklace(jewel1)
        new_jewel1.adj.sort(key=lambda x: x.value)
        new_jewel2 = new_jewel1.adj[0]
        new_jewel3 = new_jewel1.adj[1]

        new_jewel2.adj.sort(key=lambda x: x.value)
        new_jewel3.adj.sort(key=lambda x: x.value)

        # Check jewel 1
        expected = str(jewel1)
        actual = str(new_jewel1)
        self.assertEqual(expected, actual)
        self.assertFalse(jewel1 is new_jewel1)

        # Check jewel 2
        expected = str(jewel2)
        actual = str(new_jewel2)
        self.assertEqual(expected, actual)
        self.assertFalse(jewel2 is new_jewel2)

        # Check jewel 3
        expected = str(jewel3)
        actual = str(new_jewel3)
        self.assertEqual(expected, actual)
        self.assertFalse(jewel3 is new_jewel3)

        # 5 jewels, organized in a cross, with 5 in the middle
        jewel1 = NecklaceNode(1)
        jewel2 = NecklaceNode(2)
        jewel3 = NecklaceNode(3)
        jewel4 = NecklaceNode(4)
        jewel5 = NecklaceNode(5)

        jewel1.adj = [jewel5]
        jewel2.adj = [jewel5]
        jewel3.adj = [jewel5]
        jewel4.adj = [jewel5]
        jewel5.adj = [jewel1, jewel2, jewel3, jewel4]

        new_jewel5 = ReplicateNecklace(jewel5)
        new_jewel5.adj.sort(key=lambda x: x.value)
        new_jewel1, new_jewel2, new_jewel3, new_jewel4 = new_jewel5.adj

        # Check jewel 1
        expected = str(jewel1)
        actual = str(new_jewel1)
        self.assertEqual(expected, actual)
        self.assertFalse(jewel1 is new_jewel1)

        # Check jewel 2
        expected = str(jewel2)
        actual = str(new_jewel2)
        self.assertEqual(expected, actual)
        self.assertFalse(jewel2 is new_jewel2)

        # Check jewel 3
        expected = str(jewel3)
        actual = str(new_jewel3)
        self.assertEqual(expected, actual)
        self.assertFalse(jewel3 is new_jewel3)

        # Check jewel 4
        expected = str(jewel4)
        actual = str(new_jewel4)
        self.assertEqual(expected, actual)
        self.assertFalse(jewel4 is new_jewel4)

        # Check jewel 5
        expected = str(jewel5)
        actual = str(new_jewel5)
        self.assertEqual(expected, actual)
        self.assertFalse(jewel5 is new_jewel5)

    def test_complex_1(self):
        # Star-shaped necklace
        star_sides = 5
        jewels = [NecklaceNode(i) for i in range(star_sides)]
        for index, jewel in enumerate(jewels):
            jewel.adj = sorted([jewels[(index + 2) % star_sides], jewels[(index + 3) % star_sides]],\
                               key=lambda x: x.value)

        # Check that the graphs are equivalent
        new_jewel0 = ReplicateNecklace(jewels[0])
        checked = star_sides * [False]
        stack_new = [new_jewel0]
        stack = [jewels[0]]
        while(stack):
            curr = stack.pop()
            curr_new = stack_new.pop()
            curr_new.adj.sort(key=lambda x: x.value)
            expected = str(curr)
            actual = str(curr_new)
            self.assertEqual(expected, actual)
            self.assertFalse(curr is curr_new)

            checked[curr.value] = True

            for adjacent in curr.adj:
                if not checked[adjacent.value]:
                    stack.append(adjacent)

            for adjacent in curr_new.adj:
                if not checked[adjacent.value]:
                    stack_new.append(adjacent)

        self.assertTrue(len(stack_new) == 0)

        # 6-sided star
        star_sides = 6
        jewels = [NecklaceNode(i) for i in range(star_sides)]
        for index, jewel in enumerate(jewels):
            jewel.adj = sorted([jewels[(index + 2) % star_sides], jewels[(index + 4) % star_sides]],
                               key=lambda x: x.value)

        # Check that the graphs are equivalent
        new_jewel0 = ReplicateNecklace(jewels[0])
        checked = star_sides * [False]
        stack_new = [new_jewel0]
        stack = [jewels[0]]
        while (stack):
            curr = stack.pop()
            curr_new = stack_new.pop()
            curr_new.adj.sort(key=lambda x: x.value)
            expected = str(curr)
            actual = str(curr_new)
            self.assertEqual(expected, actual)
            self.assertFalse(curr is curr_new)

            checked[curr.value] = True

            for adjacent in curr.adj:
                if not checked[adjacent.value]:
                    stack.append(adjacent)

            for adjacent in curr_new.adj:
                if not checked[adjacent.value]:
                    stack_new.append(adjacent)

        self.assertTrue(len(stack_new) == 0)

    def test_complex2(self):
        # 2 attached necklaces of 6 jewels each
        necklace_size = 6
        ring1 = [NecklaceNode(i) for i in range(necklace_size)]
        ring2 = [NecklaceNode(i) for i in range(necklace_size, necklace_size * 2)]
        for index, jewel in enumerate(ring1):
            ring1[index].adj = sorted([ring1[(index + 1) % necklace_size], ring1[(index - 1) % necklace_size]],
                                      key=lambda x: x.value)
            ring2[index].adj = sorted([ring2[(index + 1) % necklace_size], ring2[(index - 1) % necklace_size]],
                                      key=lambda x: x.value)
        # connect the 2 necklaces up
        ring1[0].adj.append(ring2[0])
        ring2[0].adj.insert(0, ring1[0])

        # Check that the graphs are equivalent
        new_jewel0 = ReplicateNecklace(ring1[0])
        checked = necklace_size * 2 * [False]
        stack_new = [new_jewel0]
        stack = [ring1[0]]
        while (stack):
            curr = stack.pop()
            curr_new = stack_new.pop()
            curr_new.adj.sort(key=lambda x: x.value)
            expected = str(curr)
            actual = str(curr_new)
            self.assertEqual(expected, actual)
            self.assertFalse(curr is curr_new)

            checked[curr.value] = True

            for adjacent in curr.adj:
                if not checked[adjacent.value]:
                    stack.append(adjacent)

            for adjacent in curr_new.adj:
                if not checked[adjacent.value]:
                    stack_new.append(adjacent)

        self.assertTrue(len(stack_new) == 0)


        # The onsay: 4 interlocking rings of 10 diamonds each!
        necklace_size = 10
        ring_count = 4
        the_onsay = []
        for i in range(ring_count):
            ring = [NecklaceNode(j) for j in range(necklace_size * i, necklace_size * (i + 1))]
            for index, jewel in enumerate(ring):
                ring[index].adj = sorted([ring[(index + 1) % necklace_size], ring[(index - 1 ) % necklace_size]],
                                         key=lambda x: x.value)
            the_onsay.append(ring)

        for i in range(ring_count - 1):
            the_onsay[i][0].adj.append(the_onsay[i + 1][0])
            the_onsay[i + 1][0].adj.insert(0, the_onsay[i][0])

        # Check that the graphs are equivalent
        new_jewel0 = ReplicateNecklace(the_onsay[0][0])
        checked = necklace_size * ring_count * [False]
        stack_new = [new_jewel0]
        stack = [the_onsay[0][0]]
        while (stack):
            curr = stack.pop()
            curr_new = stack_new.pop()
            curr_new.adj.sort(key=lambda x: x.value)
            expected = str(curr)
            actual = str(curr_new)
            self.assertEqual(expected, actual)
            self.assertFalse(curr is curr_new)

            checked[curr.value] = True

            for adjacent in curr.adj:
                if not checked[adjacent.value]:
                    stack.append(adjacent)

            for adjacent in curr_new.adj:
                if not checked[adjacent.value]:
                    stack_new.append(adjacent)

        self.assertTrue(len(stack_new) == 0)

        print("Congragulations! The heist will sure to be a failure!")

    def test_comprehensive(self):
        # Who made this necklace? It's a random mess!
        random.seed(331)
        necklace_size = 100
        jewels = [NecklaceNode(i) for i in range(necklace_size)]

        for jewel in jewels:
            jewel.adj = sorted(random.sample(jewels, random.randint(2, 10)), key=lambda x: x.value)

        # Check that the graphs are equivalent
        new_jewel0 = ReplicateNecklace(jewels[0])
        checked = necklace_size * [False]
        stack_new = [new_jewel0]
        stack = [jewels[0]]
        while (stack):
            curr = stack.pop()
            curr_new = stack_new.pop()
            curr_new.adj.sort(key=lambda x: x.value)
            expected = str(curr)
            actual = str(curr_new)
            self.assertEqual(expected, actual)
            self.assertFalse(curr is curr_new)

            checked[curr.value] = True

            for adjacent in curr.adj:
                if not checked[adjacent.value]:
                    stack.append(adjacent)

            for adjacent in curr_new.adj:
                if not checked[adjacent.value]:
                    stack_new.append(adjacent)

        self.assertTrue(len(stack_new) == 0)

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
        self.assertEqual("6", response["number"])
