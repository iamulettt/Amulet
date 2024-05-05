
import time

import utils
import shutil
import logging
import os.path
import numpy as np
import torch

from torch import nn, autocast
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import models
from datetime import datetime
from tqdm import tqdm


class _BaseModel(object):
    def __init__(self, device, config, train):
        self.device = device
        self.config = config
        self.best_loss = None
        self._train_ = train

    def save(self, epoch, loss):
        filename = self.config + '-{:.4f}.'.format(loss) + str(epoch)
        torch.save(self.state_dict(), filename)
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss:
            self.best_loss = loss
            shutil.copyfile(filename, self.config)

    def load_state_dict(self, state_dict):
        ...

    def state_dict(self):
        ...

    def train(self, train_loader, test_loader, log_config):
        ...

    def run(self, user, attack, env):
        ...
