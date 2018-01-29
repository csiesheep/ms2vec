#!/usr/bin/python
# -*- encoding: utf8 -*-

import math
from multiprocessing import Process, Pool, Value, Array
import numpy as np
import optparse
import os
import random
from scipy.special import expit
import sys
import time
import warnings

from ds import graphlet


__author__ = 'sheep'


graph = None
matcher = None


class MP2Vec():

    def __init__(self, size=100, window=3, neg=5,
                       alpha=0.025, num_processes=1, iterations=1,
                       is_no_circle_path=False):
        '''
            size:      Dimensionality of word embeddings
            window:    Max window length
            neg:       Number of negative examples (>0) for
                       negative sampling, 0 for hierarchical softmax
            alpha:     Starting learning rate
            num_processes: Number of processes
            iterations: Number of iterations
            is_no_circle_path: Generate training data without circle in the path
        '''
        self.size = size
        self.window = window
        self.neg = neg
        self.alpha = alpha
        self.num_processes = num_processes
        self.vocab = None
        self.node2vec = None
        self.role2vec = None
        self.is_no_circle_path = is_no_circle_path

    def train(self, g, walk_num, walk_length, seed=None,
                                              id2vec_fname=None,
                                              path2vec_fname=None):
        if seed is not None:
            np.random.seed(seed)

        global graph
        graph = g

        #FIXME matcher
        global matcher
        matcher = graphlet.GraphletMatcher()

        id2classes = {}
        for class_, ids in graph.class_nodes.items():
            for id_ in ids:
                id2classes[id_] = class_

        self.id2node = {}
        for node, id_ in graph.node2id.items():
            self.id2node[id_] = node

        training_size = self.window * (len(graph.graph) * walk_num * walk_length)
        print 'training data size: %d' % training_size
        print 'distinct node count: %d' % len(graph.graph)

        #initialize vectors
        Wx, Wr = MP2Vec.init_net(self.size,
                                 len(graph.graph),
                                 role_size=100)

        counter = Value('i', 0)
        neg_sampler = NegSampler(graph)

        print 'start training'
        if self.num_processes > 1:
            cs = [walk_num/self.num_processes] * self.num_processes
            if walk_num % self.num_processes != 0:
                for i in range(walk_num % self.num_processes):
                    cs[-i-1] += 1

            processes = []
            for i in range(self.num_processes):
                p = Process(target=train_process,
                                   args=(i, Wx, Wr, cs[i], walk_length,
                                         neg_sampler, id2classes,
                                         self.neg, self.alpha,
                                         self.window, counter,
                                         self.is_no_circle_path,
                                         seed, training_size))
                processes.append(p)

            start = time.time()
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            end = time.time()
        else:
            start = time.time()
            train_process(0, Wx, Wr, walk_num, walk_length, neg_sampler,
                          id2classes,
                          self.neg, self.alpha,
                          self.window, counter,
                          self.is_no_circle_path, seed, training_size)
            end = time.time()

        self.node2vec = []
        for vec in Wx:
            self.node2vec.append(np.array(list(vec)))

        self.role_count = matcher.rid_offset
        self.role2vec = []
        for vec in Wr[:matcher.rid_offset]:
            self.role2vec.append(np.array(list(vec)))

        print
        print 'Finished. Total time: %.2f minutes' %  ((end-start)/60)

    @staticmethod
    def init_net(dim, node_size, role_size=1000):
        '''
            return
                Wx: a |V|*d matrix for input layer to hidden layer
                Wr: a |roles|*d matrix for hidden layer to output layer
        '''
        tmp = np.random.uniform(low=-0.5/dim,
                                high=0.5/dim,
                                size=(node_size, dim)).astype(np.float64)
        Wx = np.ctypeslib.as_ctypes(tmp)
        Wx = Array(Wx._type_, Wx, lock=False)

        tmp = np.random.uniform(low=0.0,
                                high=1.0/dim,
                                size=(node_size, dim)).astype(np.float64)
        Wr = np.ctypeslib.as_ctypes(tmp)
        Wr = Array(Wr._type_, Wr, lock=False)
        return Wx, Wr

    def dump_to_file(self, output_fname, type_='node'):
        '''
            input:
                type_: 'node' or 'role'
        '''
        with open(output_fname, 'w') as f:
            if type_ == 'node':
                f.write('%d %d\n' % (len(self.node2vec), self.size))
                for id_, vector in enumerate(self.node2vec):
                    line = ' '.join([str(v) for v in vector])
                    f.write('%s %s\n' % (self.id2node[id_], line))
            else:
                f.write('%d %d\n' % (len(self.role2vec), self.size))
                for id_, vector in enumerate(self.role2vec):
                    line = ' '.join([str(v) for v in vector])
                    f.write('%d %s\n' % (id_, line))


class NegSampler(object):
    '''
        For negative sampling.
        A list of indices of words in the vocab
        following a power law distribution.
    '''
    def __init__(self, g, seed=None, size=1000000, times=1):
        id2freq = dict(zip(g.graph.keys(), [0]*len(g.graph)))
        for walk in graph.random_walks(1, 1000, seed=seed):
            for id_ in walk:
                id2freq[id_] += 1

        self.table = NegSampler.generate_table(id2freq)
        random.shuffle(self.table)
        self.index = 0

    @staticmethod
    def generate_table(id2freq):
        power = 0.75
        total = sum([math.pow(count, power) for count in id2freq.values()])

        table_size = len(id2freq) * 10
        table = []
        p = 0
        current = 0
        for id_, count in id2freq.items():
            p = float(math.pow(count, power))/total

            to_ = int(table_size * p)
            if to_ == 0:
                to_ = 1
            table.extend([id_] * to_)
        return table

    def sample(self, count):
        if count == 0:
            return []

        if self.index + count < len(self.table):
            samples = self.table[self.index:self.index+count]
            self.index += count
            return samples

        if self.index + count == len(self.table):
            samples = self.table[self.index:]
            self.index = 0
            return samples

        new_index = self.index+count-len(self.table)
        samples = (self.table[self.index:] + self.table[:new_index])
        self.index = new_index
        return samples

def train_process(pid, Wx, Wr, walk_num, walk_length,
                  neg_sampler, id2classes, neg, starting_alpha,
                  window, counter,
                  is_no_circle_path, seed, training_size):

    def get_wp2_wp3(wp):
        wp2 = np.zeros(dim)
        wp3 = np.zeros(dim)
        for i, v in enumerate(wp):
#           v *= 4
            if v > 0:
                wp2[i] = 1
            if -6 <= v <= 6:
                s = 1 / (1 + math.exp(-v))

#               s = exp_table[int((v + max_exp)*(exp_table_size/max_exp/2))]
#               s = expit(v)

                wp3[i] = s * (1-s)
#               wp3[i] = dev_sigmoid(v)
        return wp2, wp3

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

    def to_x_y(data):
        i = random.randint(0, len(data[1])-1)
        xs = data[2][0:i] + data[2][i+1:]
        xrs = data[1][0:i] + data[1][i+1:]
        pos_y = data[2][i]
        yr = data[1][i]
        return xs, xrs, pos_y, yr

    np.seterr(invalid='raise', over ='raise', under='raise')

    #ignore the PEP 3118 buffer warning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        Wx = np.ctypeslib.as_array(Wx)
        Wr = np.ctypeslib.as_array(Wr)

    global graph
    global matcher

    step = 10000
    dim = len(Wx[0])
    alpha = starting_alpha
    data_count = 0

    exp_table_size = 1000
    max_exp = 6
    exp_table = []
    for i in range(exp_table_size):
        tmp = math.exp((float(i)/exp_table_size * 2 - 1) * 6)
        exp_table.append(tmp / (tmp+1))
#   for f in range(-10, 10):
#       f = float(f)/10
#       print f, sigmoid(f), exp_table[int((f + max_exp)*(exp_table_size/max_exp/2))]
#   raw_input()


    for walk in graph.random_walks(walk_num, walk_length, seed=seed):
#       print walk
        for id2degrees in complete_and_count_degrees(window, walk):
            data = matcher.get_graphlet(id2classes, id2degrees)
            if data[0] is None:
                continue

#           print data[2], data[1], matcher.graphlets, matcher.rid_offset

            xs, xrs, pos_y, yr = to_x_y(data)
#           i = random.randint(0, len(data[1])-1)
#           xs = data[2][0:i] + data[2][i+1:]
#           xrs = data[1][0:i] + data[1][i+1:]
#           pos_y = data[2][i]
#           yr = data[1][i]

            neg_ys = neg_sampler.sample(neg)
            wyr = Wr[yr]
            wyr2, wyr3 = get_wp2_wp3(wyr)

#           print xs, xrs, pos_y, yr, neg_ys

            #SGD learning
            #TODO speed up
            #TODO context predicts target
            for i in range(len(xs)):
                wx = Wx[xs[i]]
                if xrs[i] == yr:
                    wxr = wyr
                    wxr2 = wyr2
                    wxr3 = wyr3
                else:
                    wxr = Wr[xrs[i]]
                    wxr2, wxr3 = get_wp2_wp3(wxr)

                for y, label in ([(pos_y, 1)] + [(y, 0) for y in neg_ys]):
                    #TODO check
                    if xrs[i] == y:
                        continue
                    if label == 0 and y == pos_y:
                        continue

                    wy = Wx[y]

                    wxr2yr2 = wxr2 * wyr2
                    wxy = wx * wy

#                   p = sigmoid(dot)
#                   p = expit(np.dot(wxr2yr2, wxy))
#                   p = exp_table[int((np.dot(wxr2yr2, wxy) + max_exp)*(exp_table_size/max_exp/2))]
                    p = 1 / (1 + math.exp(-np.dot(wxr2yr2, wxy)))
                    g = alpha * (label - p)
                    if g == 0:
                        continue

#                   print x, wx, y, wy
#                   print xr, wxr, wxr2, wxr3
#                   print yr, wyr, wyr2, wyr3
#                   print dot, p, g

                    exr = g * wxr3 * wyr2 * wxy
                    eyr = g * wxr2 * wyr3 * wxy
                    wxr2yr2 = g * wxr2yr2
                    ex = wxr2yr2 * wy
#                   print 'ex', ex
#                   print 'ey', g * wxr2 * wyr2 * wx
#                   print 'exr', exr
#                   print 'eyr', eyr
                    wy += wxr2yr2 * wx
                    wx += ex
                    wxr += exr
                    wyr += eyr
#                   print 'wx', wx
#                   print 'wy', wy
#                   print 'wxr', wxr
#                   print 'wyr', wyr
#                   raw_input()

            data_count += 1
            if data_count % step == 0:
                counter.value += step
                ratio = float(counter.value)/training_size

                alpha = starting_alpha * (1-ratio)
                if alpha < starting_alpha * 0.0001:
                    alpha = starting_alpha * 0.0001

                sys.stdout.write(("\r%f "
                                  "%d/%d (%.2f%%) "
                                  "" % (alpha,
                                       counter.value,
                                       training_size,
                                       ratio*100,
                                       )))
                sys.stdout.flush()

    counter.value += (data_count % step)
    ratio = float(counter.value)/training_size
