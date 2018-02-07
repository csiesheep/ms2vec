#!/usr/bin/python
# -*- encoding: utf8 -*-

import sys
import optparse

from ds import loader


__author__ = 'sheep'


def main(graph_fname, vec_fname, output_fname):
    '''\
    %prog [options]
    '''
    g = loader.load_a_HIN(graph_fname)
    id2name = dict([(id_, name) for name, id_ in g.node2id.items()])

    with open(vec_fname) as f:
        with open(output_fname, 'w') as fo:
            first = True
            for line in f:
                if first:
                    first = False
                    fo.write(line)
                    continue
                tokens = line.split(' ', 1)
                name = id2name[int(tokens[0])]
                fo.write('%s %s' % (name, tokens[1]))

    return 0


if __name__ == '__main__':
    parser = optparse.OptionParser(usage=main.__doc__)
    options, args = parser.parse_args()

    if len(args) != 3:
        parser.print_help()
        sys.exit()

    sys.exit(main(*args))

