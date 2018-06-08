import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import config
import models
import multiprocessing


def train(args):
    if args.grad_operation == 'sync':
        con = config.MultiGPUConfig()
        con.set_num_gpus(args.num_gpus)
    elif args.grad_operation == 'async':
        con = config.AsyncMultiGPUConfig()
        con.set_num_gpus(args.num_gpus)
        con.set_num_train_threads(args.num_train_threads)
    else:
        con = config.Config()

    con.set_in_path(args.in_path)

    con.set_work_threads(args.threads)
    con.set_train_times(args.epochs)
    con.set_nbatches(args.batches)
    con.set_alpha(args.learning_rate)
    con.set_bern(0)
    con.set_dimension(args.dimention)
    con.set_ent_neg_rate(args.ent_neg_rate)
    con.set_rel_neg_rate(args.rel_neg_rate)
    con.set_opt_method("Adam")
    con.set_early_stopping_rounds(50)
    con.set_per_process_gpu_memory_fraction(args.per_process_gpu_memory_fraction)

    con.set_export_files(args.export_path, args.export_steps)
    con.set_out_files(os.path.join(args.out_path, "embedding.vec.json"))

    for i in range(1):
        for j in range(args.subset):
            con.set_train_subset("train_" + str(j))
            con.init()
            con.set_model(models.DistMult)
            con.load_parameters()
            con.run()

    if args.test_link_prediction or args.test_triple_classification:
        con = config.Config()
        con.set_train_subset(None)
        con.set_in_path(args.in_path)
        con.set_per_process_gpu_memory_fraction(args.per_process_gpu_memory_fraction)
        con.set_test_link_prediction(args.test_link_prediction)
        con.set_test_triple_classification(args.test_triple_classification)
        con.set_dimension(args.dimention)
        con.init()
        con.set_model(models.DistMult)
        con.load_parameters('./res/embedding.vec.h5')
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
        '--subset', '-s', type=int, default=None, required=True,
        help='Number of sub dataset.')
    parser.add_argument(
        '--learning_rate', '-lr', type=float, default=0.001, required=False,
        help='Learning rate.')
    parser.add_argument(
        '--dimention', '-d', type=int, default=512, required=False,
        help='Embedding dimention size.')
    parser.add_argument(
        '--ent_neg_rate', type=int, default=2, required=False,
        help='Negative sampling entity rate.')
    parser.add_argument(
        '--rel_neg_rate', type=int, default=2, required=False,
        help='Negative sampling relation rate.')
    parser.add_argument(
        '--epochs', '-e', type=int, default=10000, required=False,
        help='Max epoch size.')
    parser.add_argument(
        '--batches', '-b', type=int, default=10, required=False,
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
        '--export_steps', type=int, default=10, required=False,
        help='Save model and parameter at specified step interval.')
    parser.add_argument(
        '--num_gpus', type=int, default=1, required=False,
        help='')
    parser.add_argument(
        '--per_process_gpu_memory_fraction', type=float, default=None, required=False,
        help='')
    parser.add_argument(
        '--num_train_threads', type=int, default=1, required=False,
        help='')
    parser.add_argument(
        '--grad_operation', default=None, choices=['sync', 'async'], required=False,
        help='')
    parser.add_argument(
        '--import_path', type=str, default=None, required=False,
        help='')
    parser.add_argument(
        '--export_path', type=str, default='./res/model.vec.tf', required=False,
        help='')

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    train(args)


if __name__ == '__main__':
    main()
