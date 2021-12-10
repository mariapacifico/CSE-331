"""
Project 6
CSE 331 S21 (Onsay)
Maria Pacifico
hashtable.py
"""

import random
from typing import TypeVar, List, Tuple

T = TypeVar("T")
HashNode = TypeVar("HashNode")
HashTable = TypeVar("HashTable")


class HashNode:
    """
    DO NOT EDIT
    """
    __slots__ = ["key", "value", "deleted"]

    def __init__(self, key: str, value: T, deleted: bool = False) -> None:
        self.key = key
        self.value = value
        self.deleted = deleted

    def __str__(self) -> str:
        return f"HashNode({self.key}, {self.value})"

    __repr__ = __str__

    def __eq__(self, other: HashNode) -> bool:
        return self.key == other.key and self.value == other.value

    def __iadd__(self, other: T) -> None:
        self.value += other


class HashTable:
    """
    Hash Table Class
    """
    __slots__ = ['capacity', 'size', 'table', 'prime_index']

    primes = (
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
        89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
        181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277,
        281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389,
        397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499,
        503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617,
        619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739,
        743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859,
        863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991,
        997)

    def __init__(self, capacity: int = 8) -> None:
        """
        DO NOT EDIT
        Initializes hash table
        :param capacity: capacity of the hash table
        """
        self.capacity = capacity
        self.size = 0
        self.table = [None] * capacity

        i = 0
        while HashTable.primes[i] <= self.capacity:
            i += 1
        self.prime_index = i - 1

    def __eq__(self, other: HashTable) -> bool:
        """
        DO NOT EDIT
        Equality operator
        :param other: other hash table we are comparing with this one
        :return: bool if equal or not
        """
        if self.capacity != other.capacity or self.size != other.size:
            return False
        for i in range(self.capacity):
            if self.table[i] != other.table[i]:
                return False
        return True

    def __str__(self) -> str:
        """
        DO NOT EDIT
        Represents the table as a string
        :return: string representation of the hash table
        """
        represent = ""
        bin_no = 0
        for item in self.table:
            represent += "[" + str(bin_no) + "]: " + str(item) + '\n'
            bin_no += 1
        return represent

    __repr__ = __str__

    def _hash_1(self, key: str) -> int:
        """
        ---DO NOT EDIT---
        Converts a string x into a bin number for our hash table
        :param key: key to be hashed
        :return: bin number to insert _hash item at in our table, None if key is an empty string
        """
        if not key:
            return None
        hashed_value = 0

        for char in key:
            hashed_value = 181 * hashed_value + ord(char)
        return hashed_value % self.capacity

    def _hash_2(self, key: str) -> int:
        """
        ---DO NOT EDIT---
        Converts a string x into a hash
        :param key: key to be hashed
        :return: a hashed value
        """
        if not key:
            return None
        hashed_value = 0

        for char in key:
            hashed_value = 181 * hashed_value + ord(char)

        prime = HashTable.primes[self.prime_index]

        hashed_value = prime - (hashed_value % prime)
        if hashed_value % 2 == 0:
            hashed_value += 1
        return hashed_value

###############################################################################
#                          Implement the following:                           #
###############################################################################

    def __len__(self) -> int:
        """
        Getter for size
        :return: size of hashtable
        """
        return self.size

    def __setitem__(self, key: str, value: T) -> None:
        """
        Sets value with an associated key in the HashTable
        :param key: The key we are hashing
        :param value: The associated value we are storing
        :return: None
        """
        self._insert(key=key, value=value)

    def __getitem__(self, key: str) -> T:
        """
        Looks up the value with an associated key in the HashTable
        :param item: string key of item to retrieve from tabkle
        :return: value associated with the item
        """
        # If the key does not exist in the table, raise a KeyError
        if not self._get(key):
            raise KeyError
        return self._get(key).value

    def __delitem__(self, key: str) -> None:
        """
        Deletes the value with an associated key in the HashTable
        :param key: The key we are deleting the associated value of
        :return: None
        """
        # If the key does not exist in the table, raise a KeyError
        if not self._get(key):
            raise KeyError
        self._delete(key)

    def __contains__(self, key: str) -> bool:
        """
        Determines if a node with the key denoted by the parameter exists in the table
        :param key: The key we are checking to be a part of the hash table
        :return: Whether or not the key exists
        """
        if self._get(key) is not None:
            return True
        return False

    def _hash(self, key: str, inserting: bool = False) -> int:
        """
        Hash Method
        :param key: key to hash
        :param inserting: bool, are we inserting or not
        :return: hashed bin for table
        """
        # double hash until you find the right index
        index = self._hash_1(key)
        probe = 0
        ind = index
        node = self.table[index]

        # need to insert and this node has been deleted
        if node and node.deleted and inserting:
            return index

        # found the element with the key and it isn't deleted
        if node and not node.deleted and node.key == key:
            return index

        # collision, find the next avaliable index
        while node is not None:
            # need to check same conditions with new node
            if node and node.deleted and inserting:
                return index
            if node and not node.deleted and node.key == key:
                return index

            probe += 1  # Increment probe
            # New index is hash_1() + (p * hash_2()) % capacity
            index = (ind + probe * self._hash_2(key)) % self.capacity
            node = self.table[index]
        return index

    def _insert(self, key: str, value: T) -> None:
        """
        Add a HashNode to the hash table
        :param key: The key associated with the value we are storing
        :param value: The associated value we are storing
        """
        # get index from hash function
        index = self._hash(key, inserting=True)

        if index is not None:
            node = self.table[index]  # Get the node at that index

            # If the node exists and the keys match, just update the value
            if node and node.key == key:
                node.value = value
            # empty, add node
            else:
                self.table[index] = HashNode(key=key, value=value)
                self.size += 1

            if (self.size / self.capacity) >= 0.5:  # check load factor
                self._grow()

    def _get(self, key: str) -> HashNode:
        """
        Find the HashNode with the given key in the hash table
        :param key: key of _hash node to find in _hash table
        :return: value in table if key exists, else None
        """
        index = self._hash(key)

        # If the key is valid
        if index is not None:
            node = self.table[index]
            if not node:  # The key does not exist in this case
                return
            # Check if the index contains the correct key
            if node.key == key:
                return node

        return None

    def _delete(self, key: str) -> None:
        """
        Delete a key from the dictionary
        :param key: The key of the Node we are looking to delete
        :return: None
        """
        index = self._hash(key)

        # key has to be is valid
        if index is not None:
            node = self.table[index]
            # nothing to delete
            if not node:
                return
            # index has the correct key
            if node.key == key:
                # set as empty node
                self.table[index] = HashNode(None, None, True)
                self.size -= 1

    def _grow(self) -> None:
        """
        Double the capacity of the existing hash table.
        :return: None
        """
        old = self.table

        # reset hash table
        self.size = 0
        # double
        self.capacity *= 2
        self.table = [None] * (self.capacity)

        j = self.prime_index

        while HashTable.primes[j] < self.capacity:
            j += 1
        self.prime_index = j - 1

        # re insert old values in new HashTable
        for i in old:
            if i and not i.deleted:
                self._insert(i.key, i.value)

    def update(self, pairs: List[Tuple[str, T]] = []) -> None:
        """
        Updates the hash table using an iterable of key value pairs
        :param pairs: list of tuples (key, value) being updated
        :return: None
        """
        for pair in pairs:
            key, value = pair
            self._insert(key, value)

    def keys(self) -> List[str]:
        """
        Makes a list that contains all of the keys in the table
        :return: List of the keys
        """
        keys = []
        for node in self.table:
            if node and not node.deleted:
                keys.append(node.key)
        return keys

    def values(self) -> List[T]:
        """
        Makes a list that contains all of the values in the table
        :return: List of the values
        """
        values = []
        for node in self.table:
            if node and not node.deleted:
                values.append(node.value)
        return values

    def items(self) -> List[Tuple[str, T]]:
        """
        Makes a list that contains all of the key value pairs in the table
        :return: List of Tuples of the form (key, value)
        """
        items = []
        for node in self.table:
            if node and not node.deleted:
                items.append((node.key, node.value))
        return items

    def clear(self) -> None:
        """
        Clears the hash table
        :return: None
        """
        for i in range(len(self.table)):
            self.table[i] = None
        self.size = 0

class ExecuteOnlyOnce:
    """
    Represents a request handler.
    """

    def __init__(self, max_time) -> None:
        """
        Design data structure
        :param max_time: maximum time steps
        """
        self.total = HashTable()
        self.passed_time = HashTable()
        self.max_time = max_time


    def handle_request(self, time: int, request_id: str, client_id: str) -> int:
        """
        A function that returns the number of times this request has been seen already
        :param time: The current time stamp
        :param request_id: The ID from the request
        :param client_id: The ID from the client who's making the request
        :return: The number of times this request from this client has been seen
        """
        # first put ID's into Hash Maps
        # total[ID] = total seen, initial total = 0
        # passed_time[ID] = time
        both_id = request_id + client_id
        if both_id not in self.total:
            self.total[both_id] = 0
            self.passed_time[both_id] = time
            return self.total[both_id]

        # get the previous time
        prev_time = self.passed_time[both_id]

        # reset time dictionary
        self.passed_time[both_id] = time

        # reset total if there have been greater than max_time time steps
        if time - prev_time > self.max_time:
            self.total[both_id] = 0
        # otherwise, add to total
        else:
            self.total[both_id] += 1

        return self.total[both_id]
