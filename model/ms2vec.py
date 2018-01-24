from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import optparse
import sys
import tensorflow as tf
import timeit

from ds import loader
from ds import graphlet


__author__ = 'sheep'


def main(graph_fname, output_fname, options):
    '''\
    %prog [options] <graph_fname> <output_fname>
    '''

    g = loader.load_a_HIN(graph_fname)
    node_count = len(g.graph)
    role_count = 1000

    graph = tf.Graph()
    with graph.as_default():
        xs = tf.placeholder(tf.int32, [None, None])
        xrs = tf.placeholder(tf.int32, [None, None])
        y = tf.placeholder(tf.int32, [None, 1])
        yr = tf.placeholder(tf.int32, [None])

        with tf.device('/cpu:0'):
            Wx = tf.Variable(tf.random_uniform([node_count, options.d],
                                                -1.0/options.d,
                                                1.0/options.d))
#           print(xs)
#           print(Wx)
            hxs = tf.nn.embedding_lookup(Wx, xs)
#           print(hxs)

            Wxr = tf.Variable(tf.random_uniform([role_count, options.d],
                                                 -1.0/options.d,
                                                 1.0/options.d))
#           print(xrs)
            hxrs = tf.sigmoid(tf.nn.embedding_lookup(Wxr, xrs))
#           print(hxrs)

            weighted_hxs = tf.multiply(hxs, hxrs)
            c = tf.reduce_mean(weighted_hxs, 1)
#           print(c)

            hyr = tf.sigmoid(tf.nn.embedding_lookup(Wxr, yr))
#           print(hyr)
            h = tf.multiply(c, hyr)
#           print(h)

#           print(y)

            #TODO
#           by = tf.Variable(tf.zeros([options.d]))
            by = tf.Variable(tf.zeros([node_count]))
#           raw_input()

            error = tf.reduce_mean(tf.nn.nce_loss(weights=Wx,
                                                  biases=by,
                                                  labels=y,
                                                  inputs=h,
                                                  num_sampled=options.neg_size,
                                                  num_classes=node_count,
                                                  remove_accidental_hits=True,
                                                  ))

            train = tf.train.GradientDescentOptimizer(1.0).minimize(error)
            init = tf.initialize_all_variables()


#   xs_data = [(0, 1, 0), (0, 1, 2)]
#   xrs_data = [(0, 0, 0), (1, 1, 2)]
#   y_data = [[6], [7]]
#   yr_data = [0, 3]

    with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as session:
        init.run()

        # Train
        start_time = timeit.default_timer()
        ith = 0
        total_error_r = 0.0

#       total = sum(range(2, options.window+2)) * (len(g.graph) * options.walk_num * options.walk_length)
        total = options.window * (len(g.graph) * options.walk_num * options.walk_length)
        print(total)
        for dataset in graphlet.generate_training_set(g,
                                                      options.walk_num,
                                                      options.walk_length,
                                                      options.window,
                                                      options.batch_size):
            gid, xs_data, xrs_data, xcs_data, y_data, yr_data, yc_data = dataset
#           print(xs_data)
#           print(xs_data)
#           print(xrs_data)
#           print(y_data)
#           print(yr_data)
#           print
#           raw_input()
            _, error_r_val = session.run([train, error],
                                          feed_dict={
                                            xs: xs_data,
                                            xrs: xrs_data,
                                            y: y_data,
                                            yr: yr_data,
                                          })
            total_error_r += error_r_val

            ith += len(gid)
            if ith % 100000 == 0:
                diff_time = timeit.default_timer() - start_time
                print('ith:%d (%.2f%%) error: %.6f time:%.2f'
                      '' % (ith,
                            float(ith)/total*100,
                            total_error_r / ith,
                            diff_time))

        final_vectors = Wx.eval()


        id2node = {}
        for node, id_ in g.node2id.items():
            id2node[id_] = node
        with open(output_fname, 'w') as f:
            f.write('%d %d\n' % (len(id2node), options.d))
            for ith, vec in enumerate(final_vectors):
                f.write('%s %s\n' % (id2node[ith], ' '.join(map(str, vec))))


if __name__ == '__main__':
    parser = optparse.OptionParser(usage=main.__doc__)
    parser.add_option("-d", "--dimension", dest="d", default=128,
                      type=int, help="the number of dimension")
    parser.add_option("-w", "--window", dest="window", default=3,
                      type=int, help="the window size")
    parser.add_option("-c", "--walk_num", dest="walk_num", default=9,
                      type=int, help="the number of random walks per node")
    parser.add_option("-l", "--walk_length", dest="walk_length", default=320,
                      type=int, help="the length of each random walk")
    parser.add_option("-b", "--batch_size", dest="batch_size", default=100,
                      type=int, help="the size of each batch for learning")
    parser.add_option("-n", "--neg_size", dest="neg_size", default=5,
                      type=int, help="the number of negative samples")
    parser.add_option("-s", "--seed", dest="seed", default=None,
                      type=int, help="the seed for random")
    options, args = parser.parse_args()

    if len(args) != 2:
        parser.print_help()
        sys.exit()

    sys.exit(main(args[0], args[1], options))

