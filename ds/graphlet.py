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

        matcher = GraphletMatcher()
        for walk in self.graph.random_walks(count, length, seed=seed):
            for nodes in TrainingSetGenerator.get_nodes(walk, window):
                prev_edges = None
                id2classes = {nodes[0]: graph_id2classes[nodes[0]]}
                for w in range(window):
                    if w+2 > len(nodes):
                        break
                    id2classes[nodes[w+1]] = graph_id2classes[nodes[w+1]]
                    ext_edges = GraphletCompleter.complete(self.graph,
                                                     nodes[:w+2],
                                                     prev_edges=prev_edges)
                    prev_edges = ext_edges
                    gid, role_ids, node_ids, node_classes = matcher.get_graphlet_id_and_role_ids(id2classes, ext_edges)
#                   print gid

    @staticmethod
    def get_nodes(walk, window):
        for i in xrange(len(walk)):
            yield walk[i:i+window+1]


class GraphletCompleter():

    @staticmethod
    #TODO handle multi-graphs that two nodes may have more than one edge.
    #TODO node_class
    def complete(g, nodes, prev_edges=None):
        if prev_edges is None:
            ext_edges = set([])
            for i, from_id in enumerate(nodes[:-1]):
                from_tos = g.graph[from_id]
                for to_id in nodes[i:]:
                    if to_id not in from_tos:
                        continue
                    for edge_class_id in from_tos[to_id]:
                        ext_edges.add((from_id, to_id, edge_class_id))
            return ext_edges
        else:
            to_id = nodes[-1]
            for i, from_id in enumerate(nodes[:-1]):
                from_tos = g.graph[from_id]
                if to_id not in from_tos:
                    continue
                for edge_class_id in from_tos[to_id]:
                    prev_edges.add((from_id, to_id, edge_class_id))
            return prev_edges

    @staticmethod
    #TODO implement
    def incremental_complete():
        pass


#FIXME currently, only for less than or equal to 4 nodes, and indirected
class GraphletMatcher():

    def __init__(self):
        self.template = { #{(degrees): (roles)}
            (1, 1): (0, 0),
            (2, 1, 1): (0, 1, 1),
            (2, 2, 2): (0, 0, 0),
            (2, 2, 1, 1): (0, 0, 1, 1),
            (3, 1, 1, 1): (0, 1, 1, 1),
            (2, 2, 2, 2): (0, 0, 0, 0),
            (3, 2, 2, 1): (0, 1, 1, 2),
            (3, 2, 2, 1): (0, 1, 1, 2),
            (3, 3, 2, 2): (0, 0, 1, 1),
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
    def to_code(id2class, edges):
        id2count = dict([(id_, 0) for id_ in id2class])
        for edge in edges:
            id2count[edge[0]] += 1
            id2count[edge[1]] += 1
        deg_class_ids = sorted([(degree, id2class[id_], id_)
                                 for id_, degree
                                 in id2count.items()],
                                reverse=True)
        ids, degrees, classes = [], [], []
        for d, c, id_ in deg_class_ids:
            ids.append(id_)
            degrees.append(d)
            classes.append(c)
        return tuple(ids), tuple(degrees), tuple(classes)

    def get_graphlet_id_and_role_ids(self, id2classes, edges):
        '''
            return graphlet_id, role_ids, node_ids, node_classes
        '''
        ids, degrees, classes = GraphletMatcher.to_code(id2classes,
                                                        edges)
        #TODO cycles in the graphlet
        if degrees not in self.template:
            return None, None, None, None

        key = (degrees, classes)
        if key not in self.graphlets:
            roles = [self.rid_offset+rid
                     for rid in self.template[degrees]]
            self.graphlets[key] = (len(self.graphlets), roles)
            self.rid_offset += self.template[degrees][-1]+1
        gid, role_ids = self.graphlets[key]
        return gid, role_ids, ids, classes
