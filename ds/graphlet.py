#!/usr/bin/python
# -*- encoding: utf8 -*-

import collections


__author__ = 'sheep'


class TrainingSetGenerator():

    def __init__(self, graph):
        self.graph = graph

    def generate(self, count, length, window, seed=None):
        def complete_and_count_degrees(window, walk):
            for i in xrange(len(walk)-1):
                nodes = walk[i:i+window+1]
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
                        if to_id in self.graph.graph[from_id]:
                            id2count[from_id] += 1
                            id2count[to_id] += 1
                    i += 1
                    yield id2count

        graph_id2classes = {}
        for class_, ids in self.graph.class_nodes.items():
            for id_ in ids:
                graph_id2classes[id_] = class_

        matcher = GraphletMatcher()
        for walk in self.graph.random_walks(count, length, seed=seed):
            for id2degrees in complete_and_count_degrees(window,walk):
#                   matcher.get_graphlet(graph_id2classes, id2degrees)

#                   gid, role_ids, node_ids, node_classes = matcher.get_graphlet(graph_id2classes, id2degrees)
#                   print gid

                    yield matcher.get_graphlet(graph_id2classes, id2degrees)
#                   yield gid, role_ids, node_ids, node_classes



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


    def get_graphlet(self, id2classes, id2degrees):
        '''
            return graphlet_id, role_ids, node_ids, node_classes
        '''
        degrees, classes, ids = zip(*sorted([(id2degrees[id_], id2classes[id_], id_)
                                             for id_
                                             in id2degrees]))

        if degrees not in self.template:
            return None, None, None, None

        key = (degrees, classes)
        try:
            gid, role_ids = self.graphlets[key]
        except KeyError:
            role_ids = [self.rid_offset+rid
                        for rid in self.template[degrees]]
            gid = len(self.graphlets)
            self.graphlets[key] = (gid, role_ids)
            self.rid_offset = role_ids[-1]+1

        return gid, role_ids, ids, classes
