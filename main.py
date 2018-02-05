import argparse

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn

from data.dataloader import *
from train import *
from models.vgg import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debug models?')
    parser.add_argument('--debug', help='Debug the model', action='store_true', default=False)
    parser.add_argument('--datasetpath', help='path to dataset', action='store', dest='path')
    args = parser.parse_args()
    print(args.path)
    if args.path is None:
        raise Exception('Need a path to parent directory of the dataset. Use --datasetpath <path_to_data>')
    if args.debug:
        r = 0.025
        train_ratio = 0.0001
        val_ratio = 0.001
        print_every=1

    else:
        r = 1.0
        train_ratio = 1.0
        val_ratio = 1.0
        print_every = 1000
    train_set = DatasetADK20(dataset_path=args.path, train=True, ratio=train_ratio, resolution_reduction = r)

    val_set = DatasetADK20(dataset_path=args.path, train=False, ratio=val_ratio, resolution_reduction = r)
    model = VGGFCN16(num_classes = train_set.num_classes)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
            model,
            train_set,
            val_set,
            optimizer,
            criterion,
            scheduler,
            use_gpu = True
        )
    trainer.train(print_every=print_every)
