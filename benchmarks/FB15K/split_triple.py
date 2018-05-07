#!/usr/bin/env python

import sys


def split_triple(n_triples=100000):
    relation2id = dict()
    relation2id = dict()
    with open('./relation2id.txt', 'r') as f:
        f.next()
        for l in f:
            r, r_idx = l.split()
            relation2id[r] = int(r_idx)

    idx = 0
    dataset_idx = 0
    entity_dict = dict()
    entity_out = None
    triple_out = None
    for i, l in enumerate(sys.stdin):
        if i % n_triples == 0:
            if i != 0:
                entity_out.close()
                triple_out.close()
                idx = 0
                dataset_idx += 1
                entity_dict = dict()
            entity_out = open('entity2id_{:d}.txt'.format(dataset_idx), 'w')
            triple_out = open('train2id_{:d}.txt'.format(dataset_idx), 'w')

        triple = l.split()
        for e in triple[:-1]:
            if e in entity_dict:
                e_idx = entity_dict[e]
            else:
                e_idx = idx
                entity_dict[e] = idx
                entity_out.write('{:s}\t{:d}\n'.format(e, e_idx))
                idx += 1
            triple_out.write('{:d} '.format(e_idx))

        triple_out.write('{:d}\n'.format(int(relation2id[triple[-1]])))

    entity_out.close()
    triple_out.close()


if __name__ == '__main__':
    split_triple()
