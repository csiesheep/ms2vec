#!/usr/bin/python
# -*- encoding: utf8 -*-

import unittest

from ds import network
from ds.graphlet import GraphletCompleter, GraphletMatcher


__author__ = "sheep"


class GraphCompleterTest(unittest.TestCase):

    def setUp(self):
        g = network.HIN()

        g.add_edge('A', 'U', 'B', 'U', '0')
        g.add_edge('B', 'U', 'A', 'U', '0')
        g.add_edge('A', 'U', 'C', 'U', '1')
        g.add_edge('C', 'U', 'A', 'U', '1')
        g.add_edge('B', 'U', 'C', 'U', '0')
        g.add_edge('C', 'U', 'B', 'U', '0')
        g.add_edge('B', 'U', 'D', 'U', '0')
        g.add_edge('D', 'U', 'B', 'U', '0')
        g.add_edge('B', 'U', 'E', 'U', '1')
        g.add_edge('E', 'U', 'B', 'U', '1')
        g.add_edge('C', 'U', 'D', 'U', '0')
        g.add_edge('D', 'U', 'C', 'U', '0')
        g.add_edge('D', 'U', 'E', 'U', '0')
        g.add_edge('E', 'U', 'D', 'U', '0')
        g.add_edge('E', 'U', 'F', 'U', '0')
        g.add_edge('F', 'U', 'E', 'U', '0')

        self.g = g

    def testSimple1(self):
        nodes = [0, 1, 2, 3]
        edges = [0, 0, 0]

        expected = set([
            (0, 1, 0),
            (0, 2, 1),
            (1, 2, 0),
            (1, 3, 0),
            (2, 3, 0),
        ])
        actual = GraphletCompleter.complete(self.g, nodes, edges)
        self.assertEquals(expected, actual)

    def testSimple2(self):
        nodes = [0, 1, 2]
        edges = [0, 0]

        expected = set([
            (0, 1, 0),
            (0, 2, 1),
            (1, 2, 0),
        ])
        actual = GraphletCompleter.complete(self.g, nodes, edges)
        self.assertEquals(expected, actual)

    def testSimple3(self):
        nodes = [3, 2, 1, 0]
        edges = [0, 0, 0]

        expected = set([
            (3, 2, 0),
            (3, 1, 0),
            (2, 0, 1),
            (2, 1, 0),
            (1, 0, 0),
        ])
        actual = GraphletCompleter.complete(self.g, nodes, edges)
        self.assertEquals(expected, actual)

    def testSimple4(self):
        nodes = [1, 2, 3, 4]
        edges = [0, 0, 0]

        expected = set([
            (1, 2, 0),
            (1, 3, 0),
            (1, 4, 1),
            (2, 3, 0),
            (3, 4, 0),
        ])
        actual = GraphletCompleter.complete(self.g, nodes, edges)
        self.assertEquals(expected, actual)


class GraphletMatcherTest(unittest.TestCase):

    def testGetGraphlet(self):
        '''
                   / 2 \
            0 -- 1   |  4 -- 5
                   \ 3 /

            id2classes: 0,1,3,4 are 100; 2,5 are 200
        '''
        matcher = GraphletMatcher()

        # 0 -- 1
        id2classes = {0: 100, 1: 100}
        edges = set([
            (0, 1, 0),
        ])
        gid, role_ids, node_ids, node_classes = matcher.get_graphlet_id_and_role_ids(id2classes, edges)
        self.assertEquals(0, gid)
        self.assertEquals([0, 0], role_ids)
        self.assertEquals((1, 0), node_ids)
        self.assertEquals((100, 100), node_classes)
        self.assertEquals(1, matcher.rid_offset)
        self.assertEquals(1, len(matcher.graphlets))

        # 0 -- 1 -- 2
        id2classes = {0: 100, 1: 100, 2: 200}
        edges = set([
            (0, 1, 0),
            (1, 2, 1),
        ])
        gid, role_ids, node_ids, node_classes = matcher.get_graphlet_id_and_role_ids(id2classes, edges)
        self.assertEquals(1, gid)
        self.assertEquals([1, 2, 2], role_ids)
        self.assertEquals((1, 2, 0), node_ids)
        self.assertEquals((100, 200, 100), node_classes)
        self.assertEquals(3, matcher.rid_offset)
        self.assertEquals(2, len(matcher.graphlets))

        # 0 -- 1 -- 2 -- 3
        id2classes = {0: 100, 1: 100, 2: 200, 3:100}
        edges = set([
            (0, 1, 0),
            (1, 2, 1),
            (1, 3, 0),
            (2, 3, 1),
        ])
        gid, role_ids, node_ids, node_classes = matcher.get_graphlet_id_and_role_ids(id2classes, edges)
        self.assertEquals(2, gid)
        self.assertEquals([3, 4, 4, 5], role_ids)
        self.assertEquals((1, 2, 3, 0), node_ids)
        self.assertEquals((100, 200, 100, 100), node_classes)
        self.assertEquals(6, matcher.rid_offset)
        self.assertEquals(3, len(matcher.graphlets))

        # 1 -- 2
        id2classes = {1: 100, 2: 200}
        edges = set([
            (1, 2, 1),
        ])
        gid, role_ids, node_ids, node_classes = matcher.get_graphlet_id_and_role_ids(id2classes, edges)
        self.assertEquals(3, gid)
        self.assertEquals([6, 6], role_ids)
        self.assertEquals((2, 1), node_ids)
        self.assertEquals((200, 100), node_classes)
        self.assertEquals(7, matcher.rid_offset)
        self.assertEquals(4, len(matcher.graphlets))

        # 1 -- 2 -- 3
        id2classes = {1: 100, 2: 200, 3:100}
        edges = set([
            (1, 2, 1),
            (1, 3, 0),
            (2, 3, 1),
        ])
        gid, role_ids, node_ids, node_classes = matcher.get_graphlet_id_and_role_ids(id2classes, edges)
        self.assertEquals(4, gid)
        self.assertEquals([7, 7, 7], role_ids)
        self.assertEquals((2, 3, 1), node_ids)
        self.assertEquals((200, 100, 100), node_classes)
        self.assertEquals(8, matcher.rid_offset)
        self.assertEquals(5, len(matcher.graphlets))

        # 1 -- 2 -- 3 -- 4
        id2classes = {1: 100, 2: 200, 3:100, 4:100}
        edges = set([
            (1, 2, 1),
            (1, 3, 0),
            (2, 3, 1),
            (2, 4, 1),
            (3, 4, 0),
        ])
        gid, role_ids, node_ids, node_classes = matcher.get_graphlet_id_and_role_ids(id2classes, edges)
        self.assertEquals(5, gid)
        self.assertEquals([8, 8, 9, 9], role_ids)
        self.assertEquals((2, 3, 4, 1), node_ids)
        self.assertEquals((200, 100, 100, 100), node_classes)
        self.assertEquals(10, matcher.rid_offset)
        self.assertEquals(6, len(matcher.graphlets))

        # 2 -- 3
        id2classes = {2: 200, 3:100}
        edges = set([
            (2, 3, 1),
        ])
        gid, role_ids, node_ids, node_classes = matcher.get_graphlet_id_and_role_ids(id2classes, edges)
        self.assertEquals(3, gid)
        self.assertEquals([6, 6], role_ids)
        self.assertEquals((2, 3), node_ids)
        self.assertEquals((200, 100), node_classes)
        self.assertEquals(10, matcher.rid_offset)
        self.assertEquals(6, len(matcher.graphlets))

        # 2 -- 3 -- 4
        id2classes = {2: 200, 3:100, 4:100}
        edges = set([
            (2, 3, 1),
            (2, 4, 1),
            (3, 4, 0),
        ])
        gid, role_ids, node_ids, node_classes = matcher.get_graphlet_id_and_role_ids(id2classes, edges)
        self.assertEquals(4, gid)
        self.assertEquals([7, 7, 7], role_ids)
        self.assertEquals((2, 4, 3), node_ids)
        self.assertEquals((200, 100, 100), node_classes)
        self.assertEquals(10, matcher.rid_offset)
        self.assertEquals(6, len(matcher.graphlets))

    def testToCodeIsomorphism(self):
        id2classes = {0: 100, 1: 100, 2: 100, 3: 200}
        edges = set([
            (0, 1, 0),
            (1, 2, 0),
            (1, 3, 1),
            (2, 3, 1),
        ])
        ids, degrees, classes = GraphletMatcher.to_code(id2classes, edges)
        expected_ids= (1, 3, 2, 0)
        expected_degrees = (3, 2, 2, 1)
        expected_classes = (100, 200, 100, 100)
        self.assertEquals(expected_ids, ids)
        self.assertEquals(expected_degrees, degrees)
        self.assertEquals(expected_classes, classes)

        id2classes = {0: 100, 1: 100, 2: 200, 3: 100}
        edges = set([
            (0, 1, 0),
            (1, 2, 1),
            (1, 3, 0),
            (2, 3, 1),
        ])
        ids, degrees, classes = GraphletMatcher.to_code(id2classes, edges)
        expected_ids= (1, 2, 3, 0)
        expected_degrees = (3, 2, 2, 1)
        expected_classes = (100, 200, 100, 100)
        self.assertEquals(expected_ids, ids)
        self.assertEquals(expected_degrees, degrees)
        self.assertEquals(expected_classes, classes)

        id2classes = {0: 200, 1: 100, 2: 100, 3: 100}
        edges = set([
            (0, 1, 1),
            (0, 2, 1),
            (1, 2, 0),
            (2, 3, 0),
        ])
        ids, degrees, classes = GraphletMatcher.to_code(id2classes, edges)
        expected_ids= (2, 0, 1, 3)
        expected_degrees = (3, 2, 2, 1)
        expected_classes = (100, 200, 100, 100)
        self.assertEquals(expected_ids, ids)
        self.assertEquals(expected_degrees, degrees)
        self.assertEquals(expected_classes, classes)

    def testToCodeDifferentGraphlet(self):
        '''
                   / 2 \
            0 -- 1   |  4 -- 5
                   \ 3 /

            id2classes: 0,1,3,4 are 100; 2,5 are 200
        '''
        # 0 -- 1
        id2classes = {0: 100, 1: 100}
        edges = set([
            (0, 1, 0),
        ])
        ids, degrees, classes = GraphletMatcher.to_code(id2classes, edges)
        expected_ids= (1, 0)
        expected_degrees = (1, 1)
        expected_classes = (100, 100)
        self.assertEquals(expected_ids, ids)
        self.assertEquals(expected_degrees, degrees)
        self.assertEquals(expected_classes, classes)

        # 0 -- 1 -- 2
        id2classes = {0: 100, 1: 100, 2: 200}
        edges = set([
            (0, 1, 0),
            (1, 2, 1),
        ])
        ids, degrees, classes = GraphletMatcher.to_code(id2classes, edges)
        expected_ids= (1, 2, 0)
        expected_degrees = (2, 1, 1)
        expected_classes = (100, 200, 100)
        self.assertEquals(expected_ids, ids)
        self.assertEquals(expected_degrees, degrees)
        self.assertEquals(expected_classes, classes)

        # 0 -- 1 -- 2 -- 3
        id2classes = {0: 100, 1: 100, 2: 200, 3:100}
        edges = set([
            (0, 1, 0),
            (1, 2, 1),
            (1, 3, 0),
            (2, 3, 1),
        ])
        ids, degrees, classes = GraphletMatcher.to_code(id2classes, edges)
        expected_ids= (1, 2, 3, 0)
        expected_degrees = (3, 2, 2, 1)
        expected_classes = (100, 200, 100, 100)
        self.assertEquals(expected_ids, ids)
        self.assertEquals(expected_degrees, degrees)
        self.assertEquals(expected_classes, classes)

        # 1 -- 2
        id2classes = {1: 100, 2: 200}
        edges = set([
            (1, 2, 1),
        ])
        ids, degrees, classes = GraphletMatcher.to_code(id2classes, edges)
        expected_ids= (2, 1)
        expected_degrees = (1, 1)
        expected_classes = (200, 100)
        self.assertEquals(expected_ids, ids)
        self.assertEquals(expected_degrees, degrees)
        self.assertEquals(expected_classes, classes)

        # 1 -- 2 -- 3
        id2classes = {1: 100, 2: 200, 3:100}
        edges = set([
            (1, 2, 1),
            (1, 3, 0),
            (2, 3, 1),
        ])
        ids, degrees, classes = GraphletMatcher.to_code(id2classes, edges)
        expected_ids= (2, 3, 1)
        expected_degrees = (2, 2, 2)
        expected_classes = (200, 100, 100)
        self.assertEquals(expected_ids, ids)
        self.assertEquals(expected_degrees, degrees)
        self.assertEquals(expected_classes, classes)

        # 1 -- 2 -- 3 -- 4
        id2classes = {1: 100, 2: 200, 3:100, 4:100}
        edges = set([
            (1, 2, 1),
            (1, 3, 0),
            (2, 3, 1),
            (2, 4, 1),
            (3, 4, 0),
        ])
        ids, degrees, classes = GraphletMatcher.to_code(id2classes, edges)
        expected_ids= (2, 3, 4, 1)
        expected_degrees = (3, 3, 2, 2)
        expected_classes = (200, 100, 100, 100)
        self.assertEquals(expected_ids, ids)
        self.assertEquals(expected_degrees, degrees)
        self.assertEquals(expected_classes, classes)

        # 2 -- 3
        id2classes = {2: 200, 3:100}
        edges = set([
            (2, 3, 1),
        ])
        ids, degrees, classes = GraphletMatcher.to_code(id2classes, edges)
        expected_ids= (2, 3)
        expected_degrees = (1, 1)
        expected_classes = (200, 100)
        self.assertEquals(expected_ids, ids)
        self.assertEquals(expected_degrees, degrees)
        self.assertEquals(expected_classes, classes)

        # 2 -- 3 -- 4
        id2classes = {2: 200, 3:100, 4:100}
        edges = set([
            (2, 3, 1),
            (2, 4, 1),
            (3, 4, 0),
        ])
        ids, degrees, classes = GraphletMatcher.to_code(id2classes, edges)
        expected_ids= (2, 4, 3)
        expected_degrees = (2, 2, 2)
        expected_classes = (200, 100, 100)
        self.assertEquals(expected_ids, ids)
        self.assertEquals(expected_degrees, degrees)
        self.assertEquals(expected_classes, classes)

        # 2 -- 3 -- 4 -- 5
        id2classes = {2: 200, 3:100, 4:100, 5:200}
        edges = set([
            (2, 3, 1),
            (2, 4, 1),
            (3, 4, 0),
            (4, 5, 1),
        ])
        ids, degrees, classes = GraphletMatcher.to_code(id2classes, edges)
        expected_ids= (4, 2, 3, 5)
        expected_degrees = (3, 2, 2, 1)
        expected_classes = (100, 200, 100, 200)
        self.assertEquals(expected_ids, ids)
        self.assertEquals(expected_degrees, degrees)
        self.assertEquals(expected_classes, classes)

        # 3 -- 4
        id2classes = {3:100, 4:100}
        edges = set([
            (3, 4, 0),
        ])
        ids, degrees, classes = GraphletMatcher.to_code(id2classes, edges)
        expected_ids= (4, 3)
        expected_degrees = (1, 1)
        expected_classes = (100, 100)
        self.assertEquals(expected_ids, ids)
        self.assertEquals(expected_degrees, degrees)
        self.assertEquals(expected_classes, classes)

        # 3 -- 4 -- 5
        id2classes = {3:100, 4:100, 5:100}
        edges = set([
            (3, 4, 0),
            (4, 5, 1),
        ])
        ids, degrees, classes = GraphletMatcher.to_code(id2classes, edges)
        expected_ids= (4, 5, 3)
        expected_degrees = (2, 1, 1)
        expected_classes = (100, 100, 100)
        self.assertEquals(expected_ids, ids)
        self.assertEquals(expected_degrees, degrees)
        self.assertEquals(expected_classes, classes)
