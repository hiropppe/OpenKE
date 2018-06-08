#!/usr/bin/env python

import sys


def to_raw(t='./train2id.txt', e='./entity2id.txt', r='./relation2id.txt'):
    entity_dict, relation_dict = {}, {}
    with open(e) as f:
        f.next()
        for l in f:
            fbid, idx = l.split()
            entity_dict[idx] = fbid
    with open(r) as f:
        f.next()
        for l in f:
            relid, idx = l.split()
            relation_dict[idx] = relid
    with open(t) as f:
        f.next()
        for l in f:
            e1, e2, r = l.split()
            sys.stdout.write('{} {} {}\n'.format(
                entity_dict[e1], entity_dict[e2], relation_dict[r]))


if __name__ == '__main__':
    to_raw()
