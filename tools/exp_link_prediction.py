#!/usr/bin/python
# -*- encoding: utf8 -*-

import optparse
from sklearn.svm import LinearSVC
import sys

from tools import exp_classification as exp


__author__ = 'sheep'


def abs_minus(vec1, vec2):
    return [abs(v1-v2) for v1, v2 in zip(vec1, vec2)]

def minus(vec1, vec2):
    return [(v1-v2) for v1, v2 in zip(vec1, vec2)]

def average(vec1, vec2):
    return [(v1+v2)/2 for v1, v2 in zip(vec1, vec2)]

def hadamard(vec1, vec2):
    return [v1*v2 for v1, v2 in zip(vec1, vec2)]


def main(node_vec_fname, training_fname, test_fname):
    '''\
    %prog [options] <node_vec_fname> <groundtruth_fname>

    groundtruth_fname example: res/karate_club_groups.txt
    '''
    node2vec = exp.load_node2vec(node_vec_fname)
    exp_link_ranking(node2vec, training_fname, test_fname, vec_func=hadamard)
    return 0

def exp_link_ranking(node2vec, training_fname, test_fname,
                     to_predict=False,
                     vec_func=hadamard):

    def rank_by_prediction(id_, cands, node2vec, vec_func, k):
        vec1 = node2vec[id_]
        scores = []
        for cid in cands:
            v = get_vec(vec1, node2vec[cid], vec_func=vec_func)
            s = model.decision_function([v])[0]
            scores.append((s, cid))
        ranked = sorted(scores, reverse=True)[:k]
        return ranked

    def MAP(ranked, business_ids):
        score = 0.0
        hits = 0.0
        for i, (s, b_id) in enumerate(ranked):
            if b_id in business_ids:
                hits += 1
                score += hits/(i+1)
        return score/len(business_ids)

    def hits_at_k(ranked, business_ids, step=50):
        recalls = []
        hits = 0.0
        for i, (s, b_id) in enumerate(ranked):
            if b_id in business_ids:
                hits += 1
            if (i+1) % 50 == 0:
                recalls.append(hits)
            if i == 500:
                break
        return recalls

    print 'training model...'
    model = train_a_model(node2vec,
                          training_fname,
                          vec_func=vec_func)

    data = {}
    for id1, id2, label in parse(test_fname):
        if id1 not in data:
            data[id1] = [set(), []]
        if label == 1:
            data[id1][0].add(id2)
        data[id1][1].append(id2)
    print len(data)
    print sum([len(d[0]) for d in data.values()])

    print 'ranking ...'
    total_map = 0.0
    total_hits = [0.0] * 10
    total_checkin_count = 0
    for ith, id1 in enumerate(data.keys()):
        pos = data[id1][0]
        if len(pos) == 0:
            continue
        cands = data[id1][1]

        ranked = rank_by_prediction(id1, cands, node2vec, vec_func,500)

        map_ = MAP(ranked, pos)
        total_map += map_
        hits = hits_at_k(ranked, pos)
        for jth, hit in enumerate(hits):
            total_hits[jth] += hit

        total_checkin_count += len(pos)
        if ith % 10 == 0:
            print ith
            print total_map/(ith+1), [h/total_checkin_count for h in total_hits]
            print [h/((ith+1) * (i+1)*50)for i, h in enumerate(total_hits)]
            print

def train_a_model(node2vec, training_fname, vec_func=hadamard):
    def cv(X, y):
        model = LinearSVC()
        return sum(cross_validation.cross_val_score(model, X, y, cv=5,
                                                    scoring='f1',
                                                    n_jobs=5))/5

    X = []
    y = []
    for id1, id2, label in parse(training_fname):
        v = get_vec(node2vec[id1], node2vec[id2], vec_func=vec_func)
        X.append(v)
        y.append(label)

    model = LinearSVC()
    model.fit(X, y)

    return model

def get_vec(v1, v2, vec_func=hadamard):
    vec = vec_func(v1, v2)
    return vec

def parse(fname):
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            id1, id2, label = line.split('\t')
            label = int(label)
            yield id1, id2, label


if __name__ == '__main__':
    parser = optparse.OptionParser(usage=main.__doc__)
    options, args = parser.parse_args()

    if len(args) != 3:
        parser.print_help()
        sys.exit()

    sys.exit(main(*args))

