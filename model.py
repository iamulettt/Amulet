# -*- coding: utf-8 -*-

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

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR

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


class VAEEncoder(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.latent = latent
        self.net = models.resnet18(num_classes=latent * 2)
        dim_mlp = self.net.fc.in_features
        self.net.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            self.net.fc,
        )

    def forward(self, x):
        x = self.net(x)
        mu, log_var = x.split(self.latent, dim=1)
        return mu, log_var


class InvConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.05),
        )

    def forward(self, x):
        return self.net(x)


class VAEDecoder(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=latent, out_features=512 * 8),
            nn.Unflatten(dim=1, unflattened_size=(512, 2, 4)),
            InvConvBlock(in_channels=512, out_channels=256,
                         kernel_size=3, stride=2, padding=1, output_padding=1),
            InvConvBlock(in_channels=256, out_channels=128,
                         kernel_size=3, stride=2, padding=1, output_padding=1),
            InvConvBlock(in_channels=128, out_channels=64,
                         kernel_size=3, stride=2, padding=1, output_padding=1),
            InvConvBlock(in_channels=64, out_channels=32,
                         kernel_size=3, stride=2, padding=1, output_padding=1),
            InvConvBlock(in_channels=32, out_channels=16,
                         kernel_size=3, stride=2, padding=1, output_padding=1),
            InvConvBlock(in_channels=16, out_channels=3,
                         kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        return self.net(x)


class SimVaeNet(nn.Module):
    __name__ = 'SimVae'

    def __init__(self, latent, temperature):
        super().__init__()
        self.latent = latent
        self.temperature = temperature
        self.encoder = VAEEncoder(latent=latent)
        self.decoder = VAEDecoder(latent=latent)
        self.contrast = nn.Sequential(
            nn.Linear(in_features=self.latent, out_features=latent),
        )

    @staticmethod
    def reparameterize(mu, log_var, alpha=1.):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) * alpha
        return eps * std + mu

    def loss_function(self, recon, inputs, labels, mu, log_var, z):
        recon_loss = F.mse_loss(recon, inputs)
        kld_loss = torch.mean(-0.5 * torch.sum(1. + log_var - mu.pow(2) - log_var.exp()), dim=0)

        labels = torch.Tensor(labels.unsqueeze(0) == labels.unsqueeze(1))
        labels = labels.to(mu.dtype)
        features = F.normalize(self.contrast(mu), dim=1)
        logits = torch.matmul(features, features.T)

        # logits = F.sigmoid(logits / self.temperature)
        # original = logits
        # contrast_loss = F.binary_cross_entropy(logits, labels)

        # logits = F.relu(logits)
        logits = (logits + 1) / 2
        original = logits
        contrast_loss = F.mse_loss(logits, labels)

        return [recon_loss, kld_loss, contrast_loss], recon, logits, labels, original

    def forward_loss(self, inputs, labels):
        mu, log_var = self.encoder(inputs)
        z = self.reparameterize(mu, log_var)
        return self.loss_function(self.decoder(z), inputs, labels, mu, log_var, z)

    def forward(self, inputs):
        mu, log_var = self.encoder(inputs)
        return F.normalize(self.contrast(mu), dim=1)


class SimVAE(_BaseModel):
    __name__ = 'SimVAE'
    default = {
        'lr': 0.001,
        'T_max': 1000,
        'features': 128,
        'temperature': 0.07,
        'weight_decay': 1e-4
    }

    def __init__(self, device, config=None, train=True, **kwargs):
        super().__init__(device, config, train)
        self.features: int = 128
        self.temperature: float = 0.07
        self.net: nn.Module
        self.optimizer: Adam
        # self.scheduler: CosineAnnealingLR
        self.scheduler: ExponentialLR
        if config is not None and os.path.isfile(config):
            self.load_state_dict(torch.load(config))
        else:
            if train:
                self.load_train_dict(kwargs)
            else:
                raise Exception("Run mode should load .pth file")

    def load_train_dict(self, train_dict):
        for key, value in self.default.items():
            train_dict.setdefault(key, value)
        self._get_net_init(train_dict)

    def _get_net_init(self, train_dict):
        self.features = train_dict['features']
        self.temperature = train_dict['temperature']
        self.net = SimVaeNet(latent=self.features, temperature=self.temperature)
        self.net.to(device=self.device).train(mode=self._train_)
        self.optimizer = Adam(self.net.parameters(), lr=train_dict['lr'], weight_decay=train_dict['weight_decay'])
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=train_dict['T_max'])
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)

    def load_state_dict(self, state_dict):
        self._get_net_init(self.default)
        self.features = state_dict['features']
        self.temperature = state_dict['temperature']
        self.net.load_state_dict(state_dict['weights'])
        if self._train_:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.scheduler.load_state_dict(state_dict['scheduler'])

    def state_dict(self):
        return {
            'features': self.features,
            'temperature': self.temperature,
            'weights': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

    env_default = {
        'log_dir': './logs',
        'log_step': 10,
        'test_mode': 'epoch',
        'test_step': 1,
        'save_step': 1,
        'max_epoch': 120,
        'fp16_precision': False
    }
    loss_weight0 = [30, 1/32, 10]
    loss_weight1 = [20, 1/128, 1]
    loss_weight = loss_weight0

    def train(self, train_loader, test_loader, env):
        for key, value in self.env_default.items():
            env.setdefault(key, value)
        log_dir = env.get('log_dir')
        log_step = env.get('log_step')
        test_mode = env.get('test_mode')
        test_step = env.get('test_step')
        save_step = env.get('save_step')
        max_epoch = env.get('max_epoch')
        fp16_precision = env.get('fp16_precision')
        test_mode = 0 if test_mode == 'epoch' else 1 if test_mode == 'counter' else 2

        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        env['log_dir'] = log_dir = os.path.join(log_dir, current_time)
        writer = SummaryWriter(log_dir=log_dir)
        logging.basicConfig(filename=self.config + '.log', level=logging.INFO)
        utils.save_config_file(self.config + '.yaml', env)

        def _verify_test_loader():
            if isinstance(test_loader, DataLoader):
                return {'test': test_loader}
            elif isinstance(test_loader, list):
                return {f'test{i}': j for i, j in enumerate(test_loader)}
            elif isinstance(test_loader, dict):
                return test_loader
            else:
                return {}

        test_loader = _verify_test_loader()

        import matplotlib.pyplot as plt

        def _log_function(name, divide, matrix, feature, labels, logits):
            length = max(divide) + 1
            div = [divide == i for i in range(length)]
            num = [x.sum() for x in div]
            res = torch.zeros(length, length)
            for i in range(length):
                for j in range(length):
                    res[i, j] = matrix[div[i], :][:, div[j]].sum()
            T = sum(x * x for x in num)
            F = sum(num) * sum(num) - T
            TP = sum(res[i, i] for i in range(length))
            FN = T - TP
            TN = res.sum() - TP
            FP = F - TN
            for i in range(length):
                for j in range(length):
                    res[i, j] /= num[i] * num[j]
            logging.info(f"{name}_acc_matrix_{log_step}:\n{res}")
            logging.info(f"REC: {TP / T}, PRE: {TP / (TP + FP)}")
            logging.info(f"FPR: {FP / F}, FNR: {FN / T}")

            plt.figure()
            tt = time.time()
            for i in range(length):
                x = torch.nonzero(div[i])
                y = feature[div[i], :]
                y1 = y[:, div[i]]
                x1 = x.broadcast_to(y1.shape)
                plt.scatter(x1, y1, s=2)
                y2 = y[:, ~div[i]]
                x2 = x.broadcast_to(y2.shape)
                plt.scatter(x2, y2, s=2)
            plt.savefig(f'./imgs/{tt}_{name}.png')
            plt.close()

            alt = 20
            plt.figure()
            loo = []
            fpr = [1]
            fnr = [0]
            for threshold in range(1, alt, 1):
                threshold /= alt
                matrix = (labels >= threshold).eq(logits >= threshold)
                for i in range(length):
                    for j in range(length):
                        res[i, j] = matrix[div[i], :][:, div[j]].sum()
                TP = sum(res[i, i] for i in range(length))
                FN = T - TP
                TN = res.sum() - TP
                FP = F - TN
                loo.append([FN / T, FP / F, TP / (TP + FP)])
                fnr.append(FN / T)
                fpr.append(FP / F)
            fpr.append(0)
            fnr.append(1)
            plt.plot(torch.linspace(0, 1, alt + 1), fpr)
            plt.plot(torch.linspace(0, 1, alt + 1), fnr)
            plt.savefig(f'./imgs/{tt}_{name}1.png')
            plt.close()
            logging.info(f"{name}_FNR_FPR_PRE_{log_step}:\n{loo}")

        @torch.no_grad()
        def _test(loader, name='test'):
            test_ = utils.LossAcc()
            self.net.eval()
            for images, targets in loader:
                images = images.to(self.device)
                with autocast(enabled=fp16_precision, device_type=self.device):
                    loss_s, recon, logits, labels, feature = self.net.forward_loss(images, targets)
                    loss_s = [x * y for x, y in zip(loss_s, self.loss_weight)]
                    loss = sum(loss_s)
                    cnt = labels.shape[0] * labels.shape[1]
                    # acc_matrix = (labels >= 0.4).eq(logits >= 0.4)
                    acc_matrix = labels.round().eq(logits.round())
                    _log_function(name, targets, acc_matrix, feature, labels, logits)
                    acc = torch.sum(acc_matrix).item() / cnt
                    loss = np.array([loss.item(), loss_s[0].item(), loss_s[1].item(), loss_s[2].item()])
                    test_.push(loss=loss, cnt=1, acc=acc)
                    writer.add_images(name, torch.cat((images[0:1], recon[0:1]), dim=0))
            self.net.train()
            loss = test_.get_loss()
            return {name + '_loss': loss[0],
                    name + '_loss_recon': loss[1],
                    name + '_loss_kld': loss[2],
                    name + '_loss_contrast': loss[3],
                    name + '_acc': test_.get_acc()}

        def _train():
            counter = 0

            def test():
                for name, loader in test_loader.items():
                    _ = _test(loader, name)
                    writer.add_scalars('train', _, global_step=counter)
                    logging.debug(f"Counter: {counter / test_step}\t{_}")

            logging.info(f"Start {self.__name__} training for {max_epoch} epochs.")
            logging.info(f"Training with {self.device}.")

            log_ = utils.LossAcc()
            epoch_ = utils.LossAcc()

            scaler = GradScaler(enabled=fp16_precision)
            for epoch in range(max_epoch):
                # if epoch <= 20:
                #     self.loss_weight = self.loss_weight1
                # else:
                #     self.loss_weight = self.loss_weight0
                for images, targets in tqdm(train_loader, desc=f"Training epoch {epoch} "):
                    images = images.to(self.device)
                    with autocast(enabled=fp16_precision, device_type=self.device):
                        loss_s, recon, logits, labels, feature = self.net.forward_loss(images, targets)
                        loss_s = [x * y for x, y in zip(loss_s, self.loss_weight)]
                        loss = sum(loss_s)

                        self.optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()

                    cnt = labels.shape[0] * labels.shape[1]
                    acc = torch.sum(labels.round().eq(logits.round())).item() / cnt
                    loss = np.array([loss.item(), loss_s[0].item(), loss_s[1].item(), loss_s[2].item()])
                    log_.push(loss=loss, cnt=1, acc=acc)
                    epoch_.push(loss=loss, cnt=1, acc=acc)

                    counter += 1
                    if counter % log_step == 0:
                        loss = log_.get_loss()
                        writer.add_scalars('train', {
                            'train_loss': loss[0],
                            'train_loss_recon': loss[1],
                            'train_loss_kld': loss[2],
                            'train_loss_contrast': loss[3],
                            'train_acc': log_.get_acc(),
                            'lr': self.scheduler.get_last_lr()[0]
                        }, global_step=counter)
                        log_.clear()

                    if test_mode == 1 and counter % test_step == 0:
                        test()

                if test_mode == 0 and (epoch + 1) % test_step == 0:
                    test()

                if (epoch + 1) % save_step == 0:
                    self.save(epoch, epoch_.get_loss()[0])
                    logging.info(f"Epoch: {epoch}\tLoss: {epoch_.get_loss()}\tAccuracy: {epoch_.get_acc()}")
                    epoch_.clear()

                if self.scheduler.last_epoch != -1 or epoch > 10:
                    self.scheduler.step()
            logging.info("Training has finished.")

        _train()

    @torch.no_grad()
    def run(self, user, attack, env):
        fp16_precision = env.get('fp16_precision', False)
        logging.basicConfig(filename=self.config + '.log', level=logging.INFO)

        self.net.eval()

        def _test(loader):
            features = []
            for images, targets in loader:
                images = images.to(self.device)
                with autocast(enabled=fp16_precision, device_type=self.device):
                    features.append(self.net.forward(images))
            return torch.concat(features, dim=0)

        user = _test(user)
        st = time.time()
        attack = _test(attack)
        logits = torch.relu(torch.matmul(user, attack.T))
        acc = ((logits >= 0.4545).sum(dim=0) >= 1).int()
        ed = time.time()
        print(ed - st)
        logging.info(acc.detach().tolist())
        self.net.train()

    @torch.no_grad()
    def run(self, images, env):
        fp16_precision = env.get('fp16_precision', False)
        self.net.eval()
        images = images.to(self.device)
        with autocast(enabled=fp16_precision, device_type=self.device):
            return self.net.forward(images)

