# -*- coding: utf-8 -*-

import os.path
import argparse

import torch.cuda
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from model import SimVAE


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 256))
        # transforms.Normalize((0.2, 0.2, 0.3), (0.4, 0.4, 1))
    ])
    train_dataset = ImageFolder(root=args.train, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True, drop_last=True)
    test_dataset = {os.path.basename(x): ImageFolder(root=x, transform=transform) for x in args.test}
    test_loader = {name: DataLoader(dataset=dataset, batch_size=len(dataset)) for name, dataset in test_dataset.items()}
    model = SimVAE(device=args.device, train=True, config=args.output, features=args.features, T_max=args.T_max,
                   temperature=args.temperature, lr=args.lr, weight_decay=args.weight_decay)
    # model.train(train_loader=train_loader, test_loader=test_loader, env=args.__dict__)
    model.run(user=test_loader['user'], attack=test_loader['attack'], env=args.__dict__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the variation verify model by contrastive learning')

    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--batch', '-b', default=64, type=int, help='Batch size')
    parser.add_argument('--features', '-f', default=128, type=int, help='Net features')
    parser.add_argument('--temperature', '-t', default=0.07, type=float, help='Reserved')
    parser.add_argument('--max-epoch', '-e', default=300, type=int, help='Max training epoch')
    parser.add_argument('--T-max', '-T', default=400, type=int, help='The period of scheduler')
    parser.add_argument('--weight-decay', '-wd', default=1e-4, type=float, help='Weight decay of Optimizer(Adam)')

    parser.add_argument('--output', '-o', default='./model/SimCLR.pth', type=str,
                        help='The output filename of trained weight and bias')
    parser.add_argument('--log-dir', default='./logs', type=str, help='The log-dir of tensorboard')
    parser.add_argument('--log-step', default=10, type=int, help='The log step of train')
    parser.add_argument('--test-mode', default='counter', type=str, help='The mode of test on train',
                        choices=['epoch', 'counter', 'none'])
    parser.add_argument('--test-step', default=30, type=int, help='The test step of train')
    parser.add_argument('--save-step', default=4, type=int, help='The checkpoint per step epoch of train')
    parser.add_argument('--fp16-precision', action='store_true',
                        help='Whether or not to use 16-bit precision GPU training')

    parser.add_argument('--train', default='./raw3/train', type=str, help='The folder of train dataset')
    parser.add_argument('--test', nargs='+', type=list, help='The list folder of test dataset',
                        default=[
                            # './raw3/test',
                            # './raw3/time0', './raw3/time1',
                            # './raw3/position_b', './raw3/position_l', './raw3/position_r', './raw3/position_y',
                            # './raw3/walk', './raw3/no_case', './raw3/paper_cover', './raw3/play_music',
                            # './raw3/heavy_force', './raw3/light_force',
                            './raw4/user', './raw4/attack'
                        ])

    parser.add_argument('--device', default='cuda', type=str, help='The device used to train')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(args)
    main()
