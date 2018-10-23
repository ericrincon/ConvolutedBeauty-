import argparse

from convolutedbeauty.data.dataset import BeautyDataSet
from convolutedbeauty.training.fit import Train
from convolutedbeauty.models.factory import get_model

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data-path", default="/Volumes/Portable Drive/Backup2015 dec/Documents/faces")
    arg_parser.add_argument("--model", default="se-dense")
    arg_parser.add_argument("--epochs", default=30, type=int)
    arg_parser.add_argument("--minibatch", default=64, type=int)
    args = arg_parser.parse_args()

    dataset = BeautyDataSet(args.data_path)
    model = get_model(args.model_name)
    train_model = Train(model, dataset)
    train_model.fit(args.epochs, args.minibatch)





if __name__ == '__main__':
    main()
