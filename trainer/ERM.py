import time
import torch
import numpy as np

from utils.general_utils import AverageMeter, ProgressMeter
from utils.training_utils import criterion, penalty_v1, penalty_stationary, mean_accuracy, calc_ece_ace, get_maxprob_and_onehot, init_config


def train(
        model, args, device, train_loaders, optimizer, lr_scheduler, epoch
):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    penalty_v1_losses = AverageMeter("Penalty v1", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    eces = AverageMeter("ECE", ":6.2f")
    aces = AverageMeter("ACE", ":6.2f")

    
    model.train()
    end = time.time()

    args.penalty_weight = 0.0
    args.warm_start = 0

    data_iters = [iter(loader) for loader in train_loaders]

    stop_flag = False

    total_num = 0

    iter_num = 0

    while not stop_flag:
        for data_iter in data_iters:
            iter_num += 1
            try:
                images, labels = next(data_iter)
            except StopIteration:
                stop_flag = True
                break

            total_num += images.shape[0]

            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            penalty_v1_loss = penalty_v1(logits, labels)
            acc1 = mean_accuracy(logits, labels)

            #calc calibration metirics
            ece_config = init_config()
            ece_config['num_reps'] = 100
            ece_config['norm'] = 1
            ece_config['ce_type'] = 'em_ece_bin'
            ece_config['num_bins'] = 10
            ece, ace = calc_ece_ace(ece_config, logits, labels)

            losses.update(loss.item(), images.shape[0])
            top1.update(acc1.item(), images.shape[0])
            penalty_v1_losses.update(penalty_v1_loss.item(), images.shape[0])
            eces.update(ece, images.shape[0])
            aces.update(ace, images.shape[0])

            weight_norm = torch.tensor(0.).to(device)
            for w in model.parameters():
                weight_norm += w.norm().pow(2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            batch_time.update(time.time() - end)
            end = time.time()

    return [top1.avg, losses.avg, None, penalty_v1_losses.avg, None, None, eces.avg, aces.avg]




