#!/usr/bin/python
# -*- encoding: utf8 -*-

import cPickle
import network


__author__ = 'sheep'


def load_a_HIN(fname):
    '''
        Load a HIN from either an edge file or a pickled file
    '''
    try:
        g = load_a_HIN_from_pickle_file(fname)
    except cPickle.UnpicklingError:
        g = load_a_HIN_from_edge_file(fname)
    return g

def load_a_HIN_from_pickle_file(fname):
    '''
        Load a HIN from a pickled file
        The pickled file can be generated by tools/pickle_a_HIN.py
    '''
    g = cPickle.load(open(fname))
#   g.print_statistics()
    return g

def load_a_HIN_from_edge_file(fname):
    '''
        Load a HIN from a file which contains edges of the HIN

        In the file, each line is an edge, formated as
        <source_node> <source_class> <dest_node> <dest_class> <edge_class>
        An example file: res/karate_club_edges.txt

        It is assumed that the HIN is directed
    '''
    g = network.HIN()
    with open(fname) as f:
        for line in f:
            if line.startswith('#'):
                continue
            line = line.strip()
            src, src_class, dst, dst_class, edge_class = line.split('\t')
            g.add_edge(src, src_class, dst, dst_class, edge_class)
#   g.print_statistics()
    return g

