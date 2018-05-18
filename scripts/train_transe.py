import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import config
import models
import multiprocessing


def train(args):
    if args.num_gpus > 1:
        if args.sync_grads:
            con = config.MultiGPUConfig()
        else:
            con = config.AsyncMultiGPUConfig()
    else:
        con = config.Config()
    con.set_in_path(args.in_path)

    con.set_test_link_prediction(args.test_link_prediction)
    con.set_test_triple_classification(args.test_triple_classification)
    con.set_work_threads(args.threads)
    con.set_train_times(args.epochs)
    con.set_nbatches(args.batches)
    con.set_alpha(0.01)
    con.set_margin(1.0)
    con.set_bern(0)
    con.set_dimension(50)
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)
    con.set_opt_method("Adagrad")
    con.set_early_stopping_rounds(50)

    con.set_export_files(os.path.join(args.out_path, "model.vec.tf"), 0)
    con.set_export_steps(100)
    con.set_out_files(os.path.join(args.out_path, "embedding.vec.json"))
    con.init()
    con.set_model(models.TransE)
    con.run()

    if args.test_link_prediction or args.test_triple_classification:
        con.test()


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
        '--epochs', '-e', type=int, default=1000, required=False,
        help='Max epoch size.')
    parser.add_argument(
        '--batches', '-b', type=int, default=1000, required=False,
        help='Number of batches in each epoch.')
    parser.add_argument(
        '--threads', '-t', type=int, default=max(multiprocessing.cpu_count()/2, 1), required=False,
        help='Thread size for sampling input batch.')
    parser.add_argument(
        '--test_link_prediction', '-tl', default=False, action='store_true', required=False,
        help='Test link prediction.')
    parser.add_argument(
        '--test_triple_classification', '-tc', default=False, action='store_true', required=False,
        help='Test triple classification.')
    parser.add_argument(
        '--export_steps', type=int, default=100, required=False,
        help='Save model and parameter at specified step interval.')
    parser.add_argument(
        '--num_gpus', type=int, default=1, required=False,
        help='')
    parser.add_argument(
        '--sync_grads', default=False, action="store_true", required=False,
        help='')

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    train(args)


if __name__ == '__main__':
    main()
