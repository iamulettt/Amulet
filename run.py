
import os

import utils
import argparse
from io import StringIO
from flask import Flask, request, abort

import torch
from torchvision.transforms import transforms

from model import SimVAE

app = Flask(__name__)

net: SimVAE
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 256))
])

register: torch.Tensor = torch.Tensor()


def init():
    return SimVAE(device=args.device, train=False, config=args.output, features=args.features, temperature=args.temperature)


@torch.no_grad()
@app.route('/verify', methods=['POST'])
def solve_verify():
    pass

@torch.no_grad()
@app.route('/register', methods=['POST'])
def solve_register():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Amulet')

    parser.add_argument('--features', '-f', default=128, type=int, help='Net features')
    parser.add_argument('--temperature', '-t', default=0.07, type=float, help='Reserved')

    parser.add_argument('--output', '-o', default='./model3-1/SimCLR.pth-1.5311.7', type=str,
                        help='The output filename of trained weight and bias')
    parser.add_argument('--fp16-precision', action='store_true',
                        help='Whether or not to use 16-bit precision GPU training')

    parser.add_argument('--device', default='cuda', type=str, help='The device used to train')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(args)
    net = init()
    app.run(port=xxxx, host='0.0.0.0')
