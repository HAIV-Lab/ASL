import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

from .lr_scheduler import cosine_annealing


class ASLTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:
        """This trainer is for ViT model."""

        self.net = net
        self.train_loader = train_loader
        self.config = config

        print(f"lr: {config.optimizer.lr}")
        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )

        self.loss_criterion = nn.CrossEntropyLoss()
        print(f"self.config.trainer.use_amp: {self.config.trainer.use_amp}")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.trainer.use_amp)

        if isinstance(net, nn.parallel.DistributedDataParallel):
            net.module.set_scaling(config.trainer.trainer_args.scaling)
        else:
            net.set_scaling(config.trainer.trainer_args.scaling)


    def train_epoch(self, epoch_idx):
        self.net.train()
        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        total_epochs = self.config.optimizer.num_epochs
        bar_length = 20  # 进度条长度
        bar_format = "{desc}: {percentage:3.0f}%|{bar:" + str(bar_length) + "}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

        for train_step in tqdm(range(1, len(train_dataiter) + 1),
                            desc='Epoch {:03d}: '.format(epoch_idx),
                            position=0,
                            leave=True,
                            disable=not comm.is_main_process(),
                            bar_format=bar_format):
            batch = next(train_dataiter)
            data = batch['data']
            target = batch['label']

            data = data.cuda()
            target = target.cuda()
            
            # forward
            with torch.cuda.amp.autocast(enabled=self.config.trainer.use_amp):
                logit, feature = self.net(data, return_feature=True)
                loss = F.cross_entropy(logit, target)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            # exponential moving average, show smooth values
            if train_step % 500 == 0:
                print(f"Epoch [{epoch_idx}/{total_epochs}], Step [{train_step}/{len(train_dataiter)}], Loss: {loss.item():.4f}")
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced

