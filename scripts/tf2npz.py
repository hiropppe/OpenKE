#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import config
import models
import numpy as np


def tf2npz(args):
    con = config.Config()

    con.set_in_path(args.in_path)
    con.set_dimension(args.dimention)

    con.init()
    con.set_model(models.DistMult)
    con.set_import_files(args.import_path)
    con.restore_tensorflow()

    params = con.get_parameters()
    np.savez_compressed(args.out_path, rel=params['rel_embeddings'], ent=params['ent_embeddings'])


def main(cmd_line_args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in_path', '-i', type=str, default=None, required=True,
        help='Path to training dataset.')
    parser.add_argument(
        '--out_path', '-o', type=str, default=None, required=True,
        help='Path to npz file to export.')
    parser.add_argument(
        '--import_path', '-m', type=str, default='./res/model.vec.tf', required=False,
        help='Path to TF model to restore.')
    parser.add_argument(
        '--dimention', '-d', type=int, default=50, required=False,
        help='Embedding dimention size.')

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    tf2npz(args)


if __name__ == '__main__':
    main()
