#!/usr/bin/env python
# -*- coding:utf8 -*-
import json
import sys

from smart_open import smart_open
from tqdm import tqdm


relation_set = None


class WikidataDumpReader(object):

    def __init__(self, dump_path, n_items, filtered=False):
        self.dump_path = dump_path
        self.n_items = n_items
        self.filtered = filtered

    def __iter__(self):
        with smart_open(self.dump_path) as f:
            if not self.filtered:
                f.next()
            pbar = tqdm(total=self.n_items)
            for i, line in enumerate(f):
                if i < self.n_items:
                    try:
                        if self.filtered:
                            yield json.loads(line)
                        else:
                            yield json.loads(line[:-2])
                    except ValueError:
                        pass
                else:
                    break

                pbar.update(1)


def entity2id_func(args):
    seq = 0
    for data in WikidataDumpReader(args.dump, args.items, args.filtered):
        if data['type'] != 'item':
            continue

        sys.stdout.write('{:s}\t{:d}\n'.format(data['id'], seq))
        seq += 1


def relation2id_func(args):
    relation_set = set()
    seq = 0
    for data in WikidataDumpReader(args.dump, args.items, args.filtered):
        if data['type'] != 'item':
            continue

        if 'claims' not in data:
            continue

        for r in data['claims'].keys():
            if r not in relation_set:
                sys.stdout.write('{:s}\t{:d}\n'.format(r, seq))
                relation_set.add(r)
                seq += 1


def triple2id_func(args):
    sys.stderr.write('Reading entity ids.\n')
    entity2id, relation2id = {}, {}
    with open(args.entity, 'r') as f:
        f.next()
        for l in tqdm(f):
            e, n = l.split()
            entity2id[e] = int(n)

    sys.stderr.write('Reading relation ids.\n')
    with open(args.relation, 'r') as f:
        f.next()
        for l in tqdm(f):
            r, n = l.split()
            relation2id[r] = int(n)

    for data in WikidataDumpReader(args.dump, args.items, args.filtered):
        if data['type'] != 'item':
            continue

        e1 = data['id']

        if 'claims' not in data:
            continue

        for r, claims in data['claims'].items():
            for claim in claims:
                mainsnak = claim['mainsnak']
                if 'datatype' in mainsnak:
                    datatype = mainsnak['datatype']
                    snaktype = mainsnak['snaktype']

                    if datatype == 'wikibase-item' and snaktype == 'value':
                        e2 = mainsnak['datavalue']['value']['id']
                        if e1 in entity2id and e2 in entity2id and r in relation2id:
                            sys.stdout.write('{:d} {:d} {:d}\n'.format(
                                entity2id[e1], entity2id[e2], relation2id[r]))


def poplang_func(args):
    if args.mpack:
        import msgpack

    #f = open('100.mpack', 'wb')

    for data in WikidataDumpReader(args.dump, args.items):
        nolang = True
        for prop in ('sitelinks', 'labels', 'descriptions', 'aliases'):
            if prop in data:
                by_lang = data[prop]
                for lang in by_lang.keys():
                    if prop == 'sitelinks':
                        if not any(lang == l + 'wiki' for l in args.langs):
                            by_lang.pop(lang)
                    elif lang not in args.langs:
                        by_lang.pop(lang)

                if nolang and len(by_lang):
                    nolang = False

        if nolang:
            continue

        if args.mpack:
            packed = msgpack.packb(data)
            #import pdb; pdb.set_trace()
            #f.write(packed + '\n')
            #sys.stdout.write(packed + '\n')
        else:
            sys.stdout.write(json.dumps(data) + '\n')

    #f.close()


def main(cmd_line_args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dump', '-d', type=str, default=None, required=True,
        help='Wikidata dump file.')
    parser.add_argument(
        '--items', '-n', type=int, default=48248472, required=True,
        help='Number of items to process.')
    parser.add_argument(
        '--filtered', '-f', default=False, action='store_true', required=False,
        help='Set if use filtered dump file.')

    subparsers = parser.add_subparsers(help='sub-command help')
    entity = subparsers.add_parser('entity2id', help='Output entity2id.')
    entity.set_defaults(func=entity2id_func)
    relation = subparsers.add_parser('relation2id', help='Output relation2id.')
    relation.set_defaults(func=relation2id_func)
    triple = subparsers.add_parser('triple2id', help='Output triple2id.')
    triple.set_defaults(func=triple2id_func)
    triple.add_argument(
        '--entity', '-e', type=str, default='./entity2id.txt', required=False,
        help='Path to entiity2id.txt')
    triple.add_argument(
        '--relation', '-r', type=str, default='./relation2id.txt', required=False,
        help='Path to relation2id.txt')
    poplang = subparsers.add_parser('poplang', help='Pop unnecessary langs.')
    poplang.set_defaults(func=poplang_func)
    poplang.add_argument(
        '--langs', nargs='+', default=['en', 'ja'], required=False,
        help='Remove other than specified langs data from dump.')
    poplang.add_argument(
        '--mpack', action='store_true', default=False, required=False,
        help='Serialize using msgpack.')

    args = parser.parse_args()

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    args.func(args)


if __name__ == '__main__':
    main()
