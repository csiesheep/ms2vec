#!/usr/bin/python
# -*- encoding: utf8 -*-

import sys
import optparse

from ds import loader
from ds import graphlet


__author__ = 'sheep'


def main(graph_fname):
    '''\
    %prog [options]
    '''
    walk_num = 10
    walk_length = 1280
    window = 3
    batch_size = 100
    g = loader.load_a_HIN(graph_fname)

    c = 0
    for dataset in graphlet.generate_training_set_pipe(g,
                                                       walk_num,
                                                       walk_length,
                                                       window,
                                                       batch_size,
                                                       num_processes=4):
#       c += len(dataset)
#       if c % 1000000 == 0:
#           print c
        pass
    return 0


if __name__ == '__main__':
    parser = optparse.OptionParser(usage=main.__doc__)
    options, args = parser.parse_args()

    if len(args) != 1:
        parser.print_help()
        sys.exit()

    sys.exit(main(*args))

