#!/usr/bin/python
# -*- encoding: utf8 -*-

import sys
import collections
import cPickle
from multiprocessing import Process, Pool, Value, Array, Queue, Pipe
import os
import time
import tempfile


__author__ = 'sheep'


graph = None

def generate_training_set_pipe(g, count, length, window, batch_size,
                                                         seed=None,
                                                         num_processes=1):
    global graph
    graph = g

    cs = [count/num_processes] * num_processes
    if count % num_processes != 0:
        for i in range(count % num_processes):
            cs[-i-1] += 1

    id2classes = {}
    for class_, ids in graph.class_nodes.items():
        for id_ in ids:
            id2classes[id_] = class_

    processes = []
    pipes = []
    for i in range(num_processes):
        out_pipe, in_pipe = Pipe()
        pipes.append([in_pipe, True])

        p = Process(target=sub_generate_pipe,
                    args=(i, id2classes, cs[i], length, window, seed, out_pipe))
        processes.append(p)

    for p in processes:
        p.start()

    finish_count = 0
    while finish_count < num_processes:
        for pipe in pipes:
            if not pipe[1]:
                continue

            dataset = pipe[0].recv()

            if dataset == 'DONE':
                pipe[1] = False
                finish_count += 1
                continue

            i = 0
            while i < len(dataset):
                yield dataset[i:i+batch_size]
                i += batch_size

    for p in processes:
        p.join()

def sub_generate_pipe(ith, id2classes, count, length, window, seed, pipe):
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
                    if to_id in graph.graph[from_id]:
                        id2count[from_id] += 1
                        id2count[to_id] += 1
                i += 1
                yield id2count

    global graph

    max_size = 500
    size = 0
    buffered = [0] * max_size
    index = 0

    matcher = GraphletMatcher()
    for walk in graph.random_walks(count, length, seed=seed):
        for id2degrees in complete_and_count_degrees(window,walk):
            data = matcher.get_graphlet(id2classes, id2degrees)
            buffered[index] = data
            index += 1

            if index == max_size:
                pipe.send(buffered)
                buffered = [0] * max_size
                index = 0

    pipe.send(buffered)
    pipe.send('DONE')


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
