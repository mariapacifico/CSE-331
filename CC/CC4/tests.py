import unittest
from solution import Node, smaller_product
import random
from xml.dom import minidom


def insert(root: Node, value: int, index: int) -> Node:
    """
    Inserts a Node with a value and index into a BST.

    :param root: root of the tree
    :param value: Node's val
    :param index: Node's index
    :returns root of the tree
    """
    if root is None:
        return Node(value, index=index)
    if value < root.val:
        root.left = insert(root.left, value, index=index)
    else:
        root.right = insert(root.right, value, index=index)
    return root


class CC4Tests(unittest.TestCase):
    def test_empty_and_single(self):
        # (1)
        values = []
        root = None
        actual = smaller_product(root)
        expected = []
        self.assertEqual(expected, actual)

        # (2)
        values = [5]
        root = None
        for i, value in enumerate(values):
            root = insert(root, value, i)
        actual = smaller_product(root)
        expected = [None]
        self.assertEqual(expected, actual)

    def test_basic(self):
        # (1)
        values = [4, 6, 1, 9, 2]
        root = None
        for i, value in enumerate(values):
            root = insert(root, value, i)
        actual = smaller_product(root)
        expected = [2, 8, None, 48, 1]
        self.assertEqual(expected, actual)
        """
               4
              /  \
            1      6
             \      \
               2      9
        """

        # (2)
        values = [10, 4, 7, 8, 1]
        root = None
        for i, value in enumerate(values):
            root = insert(root, value, i)
        actual = smaller_product(root)
        expected = [224, 1, 4, 28, None]
        self.assertEqual(expected, actual)

        # (3)
        values = [15, 3, 6, 1]
        root = None
        for i, value in enumerate(values):
            root = insert(root, value, i)
        actual = smaller_product(root)
        expected = [18, 1, 3, None]
        self.assertEqual(expected, actual)

    def test_medium(self):

        # (1)
        values = [1, -9, 10, -13, -14, 8, 3]
        root = None
        for i, value in enumerate(values):
            root = insert(root, value, i)
        actual = smaller_product(root)
        expected = [-1638, 182, -39312, -14, None, -4914, -1638]
        self.assertEqual(expected, actual)

        # (2)
        values = [-100, -200, 300, 400, 5, -8]
        root = None
        for i, value in enumerate(values):
            root = insert(root, value, i)
        actual = smaller_product(root)
        expected = [-200, None, -800000, -240000000, -160000, 20000]
        self.assertEqual(expected, actual)

        # (3)
        values = [-1, -2, -500, 3, 4, 17, -9]
        root = None
        for i, value in enumerate(values):
            root = insert(root, value, i)
        actual = smaller_product(root)
        expected = [-9000, 4500, None, 9000, 27000, 108000, -500]
        self.assertEqual(expected, actual)

        # (4)
        values = [-700, -600, -500, -400, -300, -200, -100]
        root = None
        for i, value in enumerate(values):
            root = insert(root, value, i)
        actual = smaller_product(root)
        expected = [None, -700, 420000, -210000000, 84000000000, -25200000000000, 5040000000000000]
        self.assertEqual(expected, actual)

    def test_all_negative(self):

        random.seed(331)

        # (1)
        values = random.sample(range(-11, -1), 10)
        root = None
        for i, value in enumerate(values):
            root = insert(root, value, i)
        actual = smaller_product(root)
        expected = [-55440, 110, -990, 332640, -19958400, None, -1663200, -11, 6652800, 7920]
        self.assertEqual(expected, actual)

        # (2)
        values = random.sample(range(-27, -1), 25)
        root = None
        for i, value in enumerate(values):
            root = insert(root, value, i)
        actual = smaller_product(root)
        expected = [-15123429792247711334400000, -124903451312640000, 90740578753486268006400000,
                    -270061246290137702400000, -453702893767431340032000000, -27, 1361108681302294020096000000,
                    -3000680514334863360000, -1700755056000, 272789137666805760000, 30613591008000, 8326896754176000,
                    213127200, 89513424000, -22732428138900480000, -17550, 702, -9687600, 30006805143348633600000,
                    None, 1748648318376960000, 421200, -4475671200, -520431047136000, 2160489970321101619200000]
        self.assertEqual(expected, actual)

    def test_comprehensive_small(self):
        random.seed(331)

        # (1)
        values = random.sample(range(-10, 10), 17)
        root = None
        for i, value in enumerate(values):
            root = insert(root, value, i)
        actual = smaller_product(root)
        expected = [-403200, 3360, 67200, 80, 0, 0, 403200, 0, 0, 0, -16800, 0, -201600, None, 0, -560, -10]
        self.assertEqual(expected, actual)

        # (2)
        values = random.sample(range(-10, 10), 20)
        root = None
        for i, value in enumerate(values):
            root = insert(root, value, i)
        actual = smaller_product(root)
        expected = [0, 3628800, 0, -604800, 5040, -30240, 0, -3628800, 0, 90, -10, 0, 0, 0, 151200, 1814400,
                    None, -720, 0, 0]

        self.assertEqual(expected, actual)

        # (3)
        values = random.sample(range(1,11), 10)
        root = None
        for i, value in enumerate(values):
            root = insert(root, value, i)
        actual = smaller_product(root)
        expected = [2, 5040, 6, 720, None, 362880, 40320, 120, 1, 24]
        self.assertEqual(expected, actual)

    def test_comprehensive_large(self):
        random.seed(331)

        # (1)
        values = random.sample(range(-15, 15), 29)
        root = None
        for i, value in enumerate(values):
            root = insert(root, value, i)
        actual = smaller_product(root)
        expected = [0, 0, -838252800, 27720, 2494800, 0, 210, 0, 100590336000,
                    -100590336000, 0, 0, 50295168000, 0, 0, None, -277200, -19958400, -2520,
                    -16765056000, 139708800, 0, 0, 4191264000, 0, -15, 0, 0, 0]
        self.assertEqual(expected, actual)

        # (2)
        values = random.sample(range(-17, 17), 25)
        root = None
        for i, value in enumerate(values):
            root = insert(root, value, i)
        actual = smaller_product(root)
        expected = [-3536, 0, -980179200, 0, 0, 980179200, 0, 0, -17, None, 0, 0, 163363200, 0, 272,
                    -490089600, 0, 0, 42432, -32672640, 0, 0, 4667520, -466752, 0]
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
        self.assertEqual("4", response["number"])

if __name__ == '__main__':
    unittest.main()
