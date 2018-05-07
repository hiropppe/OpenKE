#!/usr/bin/env python

import sys


def split_triple(n_triples=100000):
    relation2id = dict()
    with open('./relation2id.txt', 'r') as f:
        f.next()
        for l in f:
            r, r_idx = l.split()
            relation2id[r] = int(r_idx)

    idx = 0
    dataset_idx = 0
    entity_dict = dict()
    entity_out, triple_out = None, None
    entities, triples = list(), list()
    for i, l in enumerate(sys.stdin):
        if i % n_triples == 0:
            if i != 0:
                with open('entity2id_{:d}.txt'.format(dataset_idx), 'w') as entity_out:
                    if entities:
                        entity_out.write('{:d}\n'.format(len(entities)))
                        for e in entities:
                            entity_out.write(e)
                with open('train2id_{:d}.txt'.format(dataset_idx), 'w') as triple_out:
                    if triples:
                        triple_out.write('{:d}\n'.format(len(triples)))
                        for t in triples:
                            triple_out.write(t)
                idx = 0
                dataset_idx += 1
                entity_dict = dict()
                del entities[:]
                del triples[:]

        triple = l.split()
        head, tail = None, None
        for e in triple[:-1]:
            if e in entity_dict:
                e_idx = entity_dict[e]
            else:
                e_idx = idx
                entity_dict[e] = idx
                entities.append('{:s}\t{:d}\n'.format(e, e_idx))
                idx += 1
            if head is None:
                head = e_idx
            else:
                tail = e_idx

        triples.append('{:d} {:d} {:d}\n'.format(head, tail, int(relation2id[triple[-1]])))

    with open('entity2id_{:d}.txt'.format(dataset_idx), 'w') as entity_out:
        if entities:
            entity_out.write('{:d}\n'.format(len(entities)))
            for e in entities:
                entity_out.write(e)
    with open('train2id_{:d}.txt'.format(dataset_idx), 'w') as triple_out:
        if triples:
            triple_out.write('{:d}\n'.format(len(triples)))
            for t in triples:
                triple_out.write(t)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        split_triple(int(sys.argv[1]))
    else:
        split_triple()
