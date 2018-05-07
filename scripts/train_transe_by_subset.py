import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import config
import models


def train(args):
    con = config.Config()
    con.set_in_path(args.in_path)

    con.set_work_threads(10)
    con.set_train_times(10000)
    con.set_nbatches(20)
    con.set_alpha(0.01)
    con.set_margin(1.0)
    con.set_bern(0)
    con.set_dimension(50)
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)
    con.set_opt_method("Adagrad")
    con.set_early_stopping_rounds(50)

    con.set_export_files(os.path.join(args.out_path, "model.vec.tf"), 0)
    con.set_out_files(os.path.join(args.out_path, "embedding.vec.json"))

    for i in range(1):
        for j in range(args.subset):
            con.set_train_subset("train" + str(j))
            con.init()
            con.set_model(models.TransE)
            con.load_parameters()
            con.run()


def main(cmd_line_args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in_path', '-i', type=str, default='./benchmarks/FB15K/', required=True,
        help='Path to training dataset.')
    parser.add_argument(
        '--out_path', '-o', type=str, default='./res/', required=True,
        help='Path to directory where export model and parameters.')
    parser.add_argument(
        '--subset', '-s', type=int, default=None, required=True,
        help='Number of sub dataset.')

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    train(args)


if __name__ == '__main__':
    main()
