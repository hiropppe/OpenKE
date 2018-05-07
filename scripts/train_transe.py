import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import config
import models


def train(args):
    # Input training files from benchmarks/FB15K/ folder.
    con = config.Config()
    # True: Input test files from the same folder.
    con.set_in_path(args.in_path)

    con.set_test_link_prediction(False)
    con.set_test_triple_classification(False)
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

    # Models will be exported via tf.Saver() automatically.
    con.set_export_files(os.path.join(args.out_path, "model.vec.tf"), 0)
    # Model parameters will be exported to json files automatically.
    con.set_out_files(os.path.join(args.out_path, "./res/embedding.vec.json"))
    # Initialize experimental settings.
    con.init()
    # Set the knowledge embedding model
    con.set_model(models.TransE)
    # Train the model.
    con.run()
    # To test models after training needs "set_test_flag(True)".
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

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    train(args)


if __name__ == '__main__':
    main()
