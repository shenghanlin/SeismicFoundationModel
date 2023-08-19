# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
import util.tools as tools
from models_mae_Denoise import forward_loss
from util.msssim import MSSSIM
from util.msssim import PSNR
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,task=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            if task=='Interpolation':
                loss, _, _ = model(samples)
                # loss_value = loss.item()
            else:
                outputs = model(samples)
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    
@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc, miou = tools.accuracy(output, target)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc'].update(acc.item(), n=batch_size)
        metric_logger.meters['miou'].update(miou.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} MIOU {miou.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc, miou=metric_logger.miou, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluateRegressionold(data_loader, model, device):
    # MSEcriterion = torch.nn.MSE()
    criterion = forward_loss

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # acc, miou = tools.accuracy(output, target)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        # metric_logger.meters['acc'].update(acc.item(), n=batch_size)
        # metric_logger.meters['miou'].update(miou.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluateRegression(data_loader, model, device,task=''):
    MSEcriterion = torch.nn.MSELoss()
    MSSSIMcriterion = MSSSIM()
    PSNRcriterion = PSNR()
    # criterion = forward_loss

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            # output = model(images)
            if task=='Denoise':
                output = model(images)
                output = images - output
            elif task == 'Interpolation':
                loss, output, mask = model(images)
                output = output*(1-mask)+images*mask
            else:
                output = model(images)
            mseloss = MSEcriterion(output, target)
            msssimloss = MSSSIMcriterion(output, target)
            psnrloss = PSNRcriterion(output, target)
            
        # acc, miou = tools.accuracy(output, target)

        batch_size = images.shape[0]
        metric_logger.update(loss=mseloss.item())
        metric_logger.meters['mse'].update(mseloss.item(), n=batch_size)
        metric_logger.meters['msssim'].update(msssimloss.item(), n=batch_size)
        metric_logger.meters['psnr'].update(psnrloss.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    print('* MSE {mse.global_avg:.3f} MSSSIM {msssim.global_avg:.3f} PSNR {psnr.global_avg:.3f}'
          .format(mse=metric_logger.mse, msssim=metric_logger.msssim, psnr=metric_logger.psnr))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
