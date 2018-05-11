#!/usr/bin/env python
# -*- coding:utf8 -*-

import tensorflow as tf

from tqdm import tqdm


def to_tfrecord(args):
    out = args.triple[:-3] + 'tfrecord'
    opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    with tf.python_io.TFRecordWriter(out, options=opt) as writer:
        with open(args.triple) as f:
            f.next()
            for l in tqdm(f):
                h, t, r = l.split()
                example = tf.train.Example(features=tf.train.Features(feature={
                        "h": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(h)])),
                        "t": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(t)])),
                        "r": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(r)]))
                    }))
                writer.write(example.SerializeToString())


def main(cmd_line_args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--triple', '-t', type=str, default=None, required=True,
        help='Path to triple2id.txt')

    args = parser.parse_args()

    to_tfrecord(args)


if __name__ == '__main__':
    main()
