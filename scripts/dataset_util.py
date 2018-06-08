import numpy as np
import os

from tqdm import tqdm


def split(in_path, n_split, random=True):
    print('Splitting dataset (n_split={:d}, random={:s}) .'.format(n_split, str(random)))
    rel2id = dict()
    with open(os.path.join(in_path, 'relation2id.txt'), 'r') as f:
        f.next()
        for l in f:
            r, r_idx = l.split()
            rel2id[r] = int(r_idx)
    print('#Relations {:d}'.format(int(r_idx)+1))

    raw_triple = {}
    with open(os.path.join(in_path, 'train2rawid.txt'), 'r') as f:
        for i, l in enumerate(f):
            raw_triple[i] = l
    print('#Triples {:d}'.format(i+1))

    triple_ids = np.arange(i+1)
    if random:
        np.random.shuffle(triple_ids)
    subset_length = (i+1)/n_split
    print('#Max subset triples {:d}'.format(subset_length))

    splits = []
    for i in range(n_split):
        print('[Subset] {:d}'.format(i))
        ent2id = dict()
        ents = list()
        tris = list()
        e_cnt = 0
        split = 'train_{:d}'.format(i)
        splits.append(split)
        subset_path = os.path.join(in_path, split)
        entity_path = os.path.join(subset_path, 'entity2id.txt')
        triple_path = os.path.join(subset_path, 'train2id.txt')
        if not os.path.exists(subset_path):
            os.makedirs(subset_path)
        if i < n_split - 1:
            subset_ids = triple_ids[i*subset_length: (i+1)*subset_length]
        else:
            subset_ids = triple_ids[i*subset_length:]

        print('Sampling triples ...')
        for subset_id in tqdm(subset_ids):
            each_triple = raw_triple[subset_id]
            triple = each_triple.split()
            head_id, tail_id, rel_id = None, None, rel2id[triple[-1]]
            for e in triple[:-1]:
                if e in ent2id:
                    e_idx = ent2id[e]
                else:
                    e_idx = e_cnt
                    ent2id[e] = e_cnt
                    e_cnt += 1
                    ents.append('{:s}\t{:d}\n'.format(e, e_idx))

                if head_id is None:
                    head_id = e_idx
                else:
                    tail_id = e_idx

            tris.append('{:d} {:d} {:d}\n'.format(head_id, tail_id, rel_id))

        print('Write entities to {:s}'.format(entity_path))
        with open(entity_path, 'w') as ent_out:
            ent_out.write('{:d}\n'.format(len(ents)))
            for e in tqdm(ents):
                ent_out.write(e)

        print('Write triples to {:s}'.format(triple_path))
        with open(triple_path, 'w') as tri_out:
            tri_out.write('{:d}\n'.format(len(tris)))
            for t in tqdm(tris):
                tri_out.write(t)

    return splits
