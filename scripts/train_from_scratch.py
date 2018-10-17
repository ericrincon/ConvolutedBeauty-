import argparse

from convolutedbeauty.data.dataset import BeautyDataSet


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data-path")

    args = arg_parser.parse_args()

    dataset = BeautyDataSet(args.data_path)


if __name__ == '__main__':
    main()
