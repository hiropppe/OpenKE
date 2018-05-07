import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import config
import models


def test(args):
    con = config.Config()
    con.set_in_path(args.in_path)

    con.set_test_link_prediction(True)
    con.set_test_triple_classification(True)
    con.set_work_threads(10)
    con.set_margin(1.0)
    con.set_dimension(50)

    con.init()
    con.set_model(models.TransE)
    con.load_parameters(os.path.join(args.out_path, 'embedding.vec.h5'))

    # con.predict(453, None, 37, n=10)
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

    test(args)


if __name__ == '__main__':
    main()
