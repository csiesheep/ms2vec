#!/usr/bin/python
# -*- encoding: utf8 -*-

import cPickle
import math
import numpy as np
import optparse
import os
import sys
import tempfile

from ds import loader
from ds import graphlet
from model.ms2vec import MP2Vec


__author__ = 'sheep'


def main(graph_fname, node_vec_fname, role_vec_fname, graphlet_vec_fname, options):
    '''\
    %prog [options] <graph_fname> <node_vec_fname> <path_vec_fname> <graphlet_vec_fname>

    graph_fname: the graph file
        It can be a file contained edges per line (e.g., res/karate_club_edges.txt)
        or a pickled graph file.
    node_vec_fname: the output file for nodes' vectors
    path_vec_fname: the output file for meta-paths' vectors
    '''

    print 'Load a HIN...'
    g = loader.load_a_HIN(graph_fname)
    print len(g.graph)
    for c in g.class_nodes:
        print c, len(g.class_nodes[c])
    print g.edge_class2id
    g.create_node_choices()

    id2classes = {}
    for class_, ids in g.class_nodes.items():
        for id_ in ids:
            id2classes[id_] = class_

    print 'Preprocess graphlet matcher...'
    to_freq = False
    if options.freq_fname is None or not os.path.exists(options.freq_fname):
        to_freq = True
        id2freq = dict(zip(g.graph.keys(), [0]*len(g.graph)))
    else:
        print 'load id2freq', options.freq_fname
        id2freq = cPickle.load(open(options.freq_fname))

    to_matcher = False
    if options.matcher_fname is None or not os.path.exists(options.matcher_fname):
        to_matcher = True
        matcher = graphlet.GraphletMatcher()
    else:
        print 'load matcher ', options.matcher_fname
        matcher = cPickle.load(open(options.matcher_fname))

    if to_freq or to_matcher:
        for walk in g.random_walks(1, 100):
            if to_matcher:
                for id2degrees in graphlet.complete_and_count_degrees(g,
                                                                   options.window,
                                                                   walk):
                    matcher.get_graphlet(id2classes, id2degrees)
            if to_freq:
                for id_ in walk:
                    id2freq[id_] += 1

        if to_freq and options.freq_fname is not None:
            print 'dump id2freq', options.freq_fname
            cPickle.dump(id2freq, open(options.freq_fname, 'w'))
        if to_matcher and options.matcher_fname is not None:
            print 'dump matcher', options.matcher_fname
            cPickle.dump(matcher, open(options.matcher_fname, 'w'))

    print matcher.graphlets
    print 'graphlet:', len(matcher.graphlets)
    print 'roles:', matcher.rid_offset
    tmp_freq_fname = '/tmp/ms_freq.txt'
    with open(tmp_freq_fname, 'w') as f:
        for id_, freq in sorted(id2freq.items()):
            f.write('%d\n' % (freq))

    print 'Generate training set'
    _, tmp_node_vec_fname = tempfile.mkstemp()
    _, tmp_role_vec_fname = tempfile.mkstemp()

    tmp_data_fname = options.training_fname
    to_generate = False
    if options.training_fname is None:
        _, tmp_data_fname = tempfile.mkstemp()
        print tmp_data_fname
        to_generate = True
    elif not os.path.exists(options.training_fname):
        to_generate = True

    if to_generate:
        graphlet.generate_training_set_to_file(g,
                                      matcher,
                                      id2classes,
                                      options.walk_length,
                                      options.window,
                                      tmp_data_fname,
                                      num_processes=options.num_processes)

    print 'Learn representations...'
    model = 'ms2vec'
    if options.mode:
        model = 'ms2vec_c2t'
    statement = ("model_c/bin/%s -size %d -node_count %d "
                 "-role_count %d -graphlet_count %d -role_ratio %f "
                 "-train %s -freq %s -alpha %f "
                 "-output %s -output_role %s -output_graphlet %s "
                 "-window %d -negative %d "
                 "-threads %d -sigmoid_reg %d -iteration %d -equal %d"
                 "" % (model,
                       options.dim,
                       max(g.graph)+1,
                       matcher.rid_offset,
                       len(matcher.graphlets),
                       options.role_ratio,
                       tmp_data_fname,
                       tmp_freq_fname,
                       options.alpha,
                       tmp_node_vec_fname,
                       tmp_role_vec_fname,
                       graphlet_vec_fname,
                       options.window,
                       options.neg,
                       options.num_processes,
                       options.sigmoid_reg * 1,
                       options.iter,
                       options.equal * 1))
    print statement
    os.system(statement)

    output_node2vec(g, tmp_node_vec_fname, node_vec_fname)
    output_role2vec(matcher, tmp_role_vec_fname, role_vec_fname)

def output_node2vec(g, tmp_node_vec_fname, node_vec_fname):
    with open(tmp_node_vec_fname) as f:
        with open(node_vec_fname, 'w') as fo:
            id2node = dict([(v, k) for k, v in g.node2id.items()])
            first = True
            for line in f:
                if first:
                    first = False
                    fo.write(line)
                    continue

                id_, vectors = line.strip().split(' ', 1)
                line = '%s %s\n' % (id2node[int(id_)], vectors)
                fo.write(line)

def output_role2vec(matcher, tmp_role_vec_fname, role_vec_fname):
    with open(tmp_role_vec_fname) as f:
        with open(role_vec_fname, 'w') as fo:
            for key, g_rs in matcher.graphlets.items():
                gid, roles = g_rs
                fo.write("#gid:%d\tkey:%s\troles:%s\n" % (gid,
                                                          str(key),
                                                          str(roles)))

            for line in f:
                fo.write(line)


if __name__ == '__main__':
    parser = optparse.OptionParser(usage=main.__doc__)
    parser.add_option('-l', '--walk-length', action='store',
                      dest='walk_length', default=100, type='int',
                      help=('The length of each random walk '
                            '(default: 100)'))
#   parser.add_option('-k', '--walk-num', action='store',
#                     dest='walk_num', default=10, type='int',
#                     help=('The number of random walks starting from '
#                           'each node (default: 10)'))
    parser.add_option('-n', '--negative', action='store', dest='neg',
                      default=5, type='int',
                      help=('Number of negative examples (>0) for '
                            'negative sampling, 0 for hierarchical '
                            'softmax (default: 5)'))
    parser.add_option('-d', '--dim', action='store', dest='dim',
                      default=100, type='int',
                      help=('Dimensionality of word embeddings '
                            '(default: 100)'))
    parser.add_option('-a', '--alpha', action='store', dest='alpha',
                      default=0.025, type='float',
                      help='Starting learning rate (default: 0.025)')
    parser.add_option('-r', '--role_ratio', action='store',
                      dest='role_ratio',
                      default=1.0, type='float',
                      help='Learning ratio of roles (default: 1.0)')
    parser.add_option('-w', '--window', action='store', dest='window',
                      default=1, type='int',
                      help='Max window length (default: 3)')
    parser.add_option('-p', '--num_processes', action='store',
                      dest='num_processes', default=1, type='int',
                      help='Number of processes (default: 1)')
    parser.add_option('-i', '--iter', action='store', dest='iter',
                      default=1, type='int',
                      help='Training iterations (default: 1)')
    parser.add_option('-c', '--allow-circle', action='store_true',
                      dest='allow_circle', default=False,
                      help=('Set to all circles in relationships between '
                            'nodes (Default: not allow)'))
    parser.add_option('-s', '--sigmoid_regularization',
                      action='store_true', dest='sigmoid_reg',
                      default=False,
                      help=('Use sigmoid function for regularization '
                            'for meta-path vectors '
                            '(Default: binary-step function)'))
    parser.add_option('-t', '--correct-negs',
                      action='store_true', dest='correct_neg',
                      default=False,
                      help=('Select correct negative data '
                            '(Default: false)'))
    parser.add_option('-e', '--equal',
                      action='store_true', dest='equal',
                      default=False,
                      help=('Use sigmoid function for regularization '
                            'for meta-path vectors '
                            '(Default: binary-step function)'))
    parser.add_option('-m', '--mode',
                      action='store_true', dest='mode',
                      default=False,
                      help=('Change to c2t model '
                            '(Default: t2c model)'))
    parser.add_option('-f', '--training_fname',
                      action='store', dest='training_fname',
                      default=None,
                      help=('Training file name '
                            '(Default: None)'))
    parser.add_option('-g', '--matcher_fname',
                      action='store', dest='matcher_fname',
                      default=None,
                      help=('Matcher file name '
                            '(Default: None)'))
    parser.add_option('-q', '--freq_fname',
                      action='store', dest='freq_fname',
                      default=None,
                      help=('Node frequency file name '
                            '(Default: None)'))
    options, args = parser.parse_args()

    if len(args) != 4:
        parser.print_help()
        sys.exit()

    sys.exit(main(args[0], args[1], args[2], args[3], options))

