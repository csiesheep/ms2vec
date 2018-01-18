#!/usr/bin/python
# -*- encoding: utf8 -*-

import collections


__author__ = 'sheep'


class TrainingSetGenerator():

    def __init__(self, graph):
        self.graph = graph

    def generate(self, count, length, window, seed=None):
        graph_id2classes = {}
        for class_, ids in self.graph.class_nodes.items():
            for id_ in ids:
                graph_id2classes[id_] = class_

        c = 0
        matcher = GraphletMatcher()
        for walk in self.graph.random_walks(count, length, seed=seed):
            for nodes in TrainingSetGenerator.get_nodes(walk, window):
                for id2degrees in GraphletMatcher.complete_and_count_degrees(self.graph, nodes):

                    gid, role_ids, node_ids, node_classes = matcher.get_graphlet(graph_id2classes, id2degrees)
                    print gid
                    c += 1
#                   yield gid, role_ids, node_ids, node_classes
        print c

    @staticmethod
    def get_nodes(walk, window):
        for i in xrange(len(walk)-1):
            yield walk[i:i+window+1]


#lass GraphletCompleter():

#   @staticmethod
#   #TODO handle multi-graphs that two nodes may have more than one edge.
#   #TODO node_class
#   def complete(g, nodes, prev_edges=None):
#       if prev_edges is None:
#           ext_edges = set([])
#           for i, from_id in enumerate(nodes[:-1]):
#               from_tos = g.graph[from_id]
#               for to_id in nodes[i:]:
#                   if to_id not in from_tos:
#                       continue
#                   for edge_class_id in from_tos[to_id]:
#                       ext_edges.add((from_id, to_id, edge_class_id))
#           return ext_edges
#       else:
#           to_id = nodes[-1]
#           for i, from_id in enumerate(nodes[:-1]):
#               from_tos = g.graph[from_id]
#               if to_id not in from_tos:
#                   continue
#               for edge_class_id in from_tos[to_id]:
#                   prev_edges.add((from_id, to_id, edge_class_id))
#           return prev_edges

#   @staticmethod
#   #TODO implement
#   def incremental_complete():
#       pass


#FIXME currently, only for less than or equal to 4 nodes, and indirected
class GraphletMatcher():

    def __init__(self):
        self.template = { #{(degrees): (roles)}
            (1, 1): (0, 0),
            (1, 1, 2): (0, 0, 1),
            (2, 2, 2): (0, 0, 0),
            (1, 1, 2, 2): (0, 0, 1, 1),
            (1, 1, 1, 3): (0, 0, 0, 1),
            (2, 2, 2, 2): (0, 0, 0, 0),
            (1, 2, 2, 3): (0, 1, 1, 2),
            (2, 2, 3, 3): (0, 0, 1, 1),
            (3, 3, 3, 3): (0, 0, 0, 0),
        }
        self.graphlets = {}
        self.gid_offset = 0
        self.rid_offset = 0

    def __eq__(self, other):
        if not isinstance(other, GraphletMatcher):
            return False
        if self.graphlet != other.graphlets:
            return False
        return True

    @staticmethod
    def complete_and_count_degrees(g, nodes):
        id2count = {nodes[0]: 1, nodes[1]: 1}
        yield id2count

        i = 2
        while i < len(nodes):
            to_id = nodes[i]
            if to_id in id2count:
                break

            id2count[to_id] = 1
            id2count[nodes[i-1]] += 1
            for from_id in nodes[:i-1]:
                if to_id in g.graph[from_id]:
                    id2count[from_id] += 1
                    id2count[to_id] += 1
            i += 1
            yield id2count

    def get_graphlet(self, id2classes, id2degrees):
        '''
            return graphlet_id, role_ids, node_ids, node_classes
        '''
        deg_class_ids = zip(*sorted([(degree, id2classes[id_], id_)
                                     for id_, degree
                                     in id2degrees.items()]))
        degrees = deg_class_ids[0]

        if degrees not in self.template:
            return None, None, None, None

        classes = deg_class_ids[1]
        key = (degrees, classes)
        if key not in self.graphlets:
            roles = [self.rid_offset+rid
                     for rid in self.template[degrees]]
            self.graphlets[key] = (len(self.graphlets), roles)
            self.rid_offset = roles[-1]+1
        gid, role_ids = self.graphlets[key]

        return gid, role_ids, deg_class_ids[2], classes
