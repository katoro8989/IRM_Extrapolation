import time
import torch
import numpy as np

from utils.general_utils import AverageMeter, ProgressMeter
from utils.training_utils import criterion, penalty_v1, mean_accuracy, penalty_stationary, penalty_v0, calc_ece_ace, get_maxprob_and_onehot, init_config


def train(
        model, args, device, train_loaders, optimizer, scheduler, epoch
):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    penalty_v1_losses = AverageMeter("Lv1", ":.4f")
    penalty_v0_losses = AverageMeter("Lv0", ":.4f")
    penalty_stationary_losses = AverageMeter("LS", ":.4f")
    reg_losses = AverageMeter("Reg", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    eces = AverageMeter("ECE", ":6.2f")
    aces = AverageMeter("ACE", ":6.2f")


    batch_total = torch.sum(torch.tensor([len(loader) for loader in train_loaders])).item()
    progress = ProgressMeter(
        batch_total,
        [batch_time, top1, losses, penalty_v0_losses, penalty_v1_losses, penalty_stationary_losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()

    data_iters = [iter(loader) for loader in train_loaders]

    stop_flag = False

    iter_num = 0

    penalty_weight = (args.penalty_weight
                      if epoch >= args.warm_start else 1.0)

    while 1:
        envs = [{} for _ in data_iters]
        iter_num += 1
        batch_num = 0

        images_full = []
        labels_full = []

        torch.autograd.set_detect_anomaly(True)

        for env_num, data_iter in enumerate(data_iters):
            try:
                images, labels = next(data_iter)
            except StopIteration:
                stop_flag = True
                break

            batch_num += images.shape[0]

            images = images.to(device)
            labels = labels.to(device)

            images_full.append(images)
            labels_full.append(labels)

            omega_logits = model(images, env_num=env_num)
            omega_loss = criterion(omega_logits, labels)

            optimizer[env_num].zero_grad()
            omega_loss.backward()
            optimizer[env_num].step()
            scheduler[env_num].step()

        if stop_flag:
            break

        for env_num, (images, labels) in enumerate(zip(images_full, labels_full)):

            # model.average_omega()
            model.clear_phi_grad()
            logits = model(images, env_num=env_num)
            loss = criterion(logits, labels)

            envs[env_num]["loss"] = loss
            envs[env_num]["penalty_v1"] = penalty_v1(logits, labels)
            envs[env_num]["penalty_v0"] = penalty_v0(model, images, labels, env_num)
            envs[env_num]["penalty_stationary"] = penalty_stationary(model, images, labels, 0)
            envs[env_num]["acc"] = mean_accuracy(logits, labels)

            #calc calibration metirics
            ece_config = init_config()
            ece_config['num_reps'] = 100
            ece_config['norm'] = 1
            ece_config['ce_type'] = 'em_ece_bin'
            ece_config['num_bins'] = 10
            envs[env_num]["ece"], envs[env_num]["ace"] = calc_ece_ace(ece_config, logits, labels)

        training_loss = torch.stack([env["loss"] for env in envs]).mean()
        training_acc = torch.stack([env["acc"] for env in envs]).mean()
        penalty_v1_loss = torch.stack([env["penalty_v1"] for env in envs]).mean()
        penalty_v0_loss = torch.stack([env["penalty_v0"] for env in envs]).mean()
        penalty_stationary_tensor = torch.stack([env["penalty_stationary"] for env in envs])
        penalty_stationary_loss = penalty_stationary_tensor.mean()
        penalty_v_v1 = (1 - args.var_beta) * penalty_stationary_loss + args.var_beta * penalty_stationary_tensor.var()
        ece = torch.stack([env["ece"] for env in envs]).mean()
        ace = torch.stack([env["ace"] for env in envs]).mean()

        loss = torch.tensor(0.0).to(device)

        # Training loss
        loss += training_loss
        # acc1 = mean_accuracy(logits, labels)

        model.clear_phi_grad()

        weight_norm = torch.tensor(0.).to(device)
        for w in model.parameters():
            weight_norm += w.norm().pow(2)
        weight_norm = args.wd * weight_norm

        loss += weight_norm

        loss += penalty_weight * penalty_v_v1
        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight

        optimizer[-1].zero_grad()
        loss.backward()
        optimizer[-1].step()
        scheduler[-1].step()

        losses.update(loss.item(), batch_num)
        top1.update(training_acc.item(), batch_num)
        penalty_v1_losses.update(penalty_v1_loss.item() * penalty_weight, batch_num)
        penalty_v0_losses.update(penalty_v0_loss.item() * penalty_weight, batch_num)
        penalty_stationary_losses.update(penalty_stationary_loss.item() * penalty_weight, batch_num)
        reg_losses.update(weight_norm.item(), batch_num)
        eces.update(ece.item(), batch_num)
        aces.update(ace.item(), batch_num)
        batch_time.update(time.time() - end)
        end = time.time()

        if iter_num % args.print_freq == 0:
            progress.display(iter_num)

        model.average_omega()

    return [top1.avg, losses.avg, penalty_v0_losses.avg, penalty_v1_losses.avg, penalty_stationary_losses.avg, reg_losses.avg, ece.avg, ace.avg]
