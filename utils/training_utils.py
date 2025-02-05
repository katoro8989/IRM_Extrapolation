from collections import OrderedDict

import torch
import torch.autograd as autograd
import torch.nn as nn

from typing import Tuple
import numpy as np

from utils.optim_utils import LARS


def get_optimizer_scheduler(model, args):
    if args.trainer != "BLO":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.optim == "lars":
            optimizer = LARS(optimizer=optimizer)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[int(0.5 * args.epochs), int(0.75 * args.epochs)],
                                                         gamma=0.1
                                                         )
    else:
        optimizer = []
        for i in range(len(args.training_env)):
            base_optimizer_omega = torch.optim.Adam(model.omega_list[i].parameters(), lr=args.omega_lr)
            if args.optim == "lars":
                base_optimizer_omega = LARS(optimizer=base_optimizer_omega)
            optimizer.append(base_optimizer_omega)
        base_optimizer_phi = torch.optim.Adam(model.phi.parameters(), lr=args.lr)
        if args.optim == "lars":
            base_optimizer_phi = LARS(optimizer=base_optimizer_phi)
        optimizer.append(base_optimizer_phi)

        scheduler = []

        for i in range(len(args.training_env)):
            scheduler.append(torch.optim.lr_scheduler.MultiStepLR(optimizer[i],
                                                                  milestones=[int(0.5 * args.epochs),
                                                                              int(0.75 * args.epochs)],
                                                                  gamma=0.1
                                                                  ))
        scheduler.append(torch.optim.lr_scheduler.MultiStepLR(optimizer[-1],
                                                              milestones=[int(0.5 * args.epochs),
                                                                          int(0.75 * args.epochs)],
                                                              gamma=0.1
                                                              ))

    return optimizer, scheduler


def get_device(tensor):
    device_idx = tensor.get_device()
    device = "cuda:" + str(device_idx) if device_idx > 0 else "cpu"
    return device


def criterion(logits, y):
    return torch.nn.CrossEntropyLoss()(logits, y.view(-1))


def mean_accuracy(logits, y):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == y.view(-1)).sum()
    return correct / y.shape[0]


def penalty_v1(logits, y):
    device = get_device(logits)
    scale = torch.tensor(1.).to(device).requires_grad_()
    loss = criterion(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


def penalty_v0(model, x, y, env_num=0):
    z = model(x, env_num=-1)
    for param in model.omega_list[env_num].parameters():
        param.requires_grad_(True)
    for param in model.phi.parameters():
        param.requires_grad_(False)
    logit = model.omega_list[env_num](z)
    loss = criterion(logit, y)
    grad = torch.autograd.grad(loss, model.omega_list[env_num].parameters(), retain_graph=True, create_graph=True)[0]
    for param in model.phi.parameters():
        param.requires_grad_(True)
    return torch.sum(grad.view(-1) ** 2)


def penalty_stationary(model, x, y, env_num):
    z = model(x, env_num=-1)
    for param in model.omega_list[env_num].parameters():
        param.requires_grad_(True)
    for param in model.phi.parameters():
        param.requires_grad_(False)
    logit = model.omega_list[env_num](z)
    loss = criterion(logit, y)
    grad = torch.autograd.grad(loss, model.omega_list[env_num].parameters(), retain_graph=True, create_graph=True)[0]
    for param in model.phi.parameters():
        param.requires_grad_(True)
    return torch.sum(grad.view(-1) ** 2)


def get_maxprob_and_onehot(probs: numpy.ndarray, labels: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:

    maxprob_list = []
    idx_list = []

    for i in range(len(probs)):
        maxprob_list.append(numpy.max(probs[i]))
        idx_list.append(numpy.argmax(probs[i]))

    maxprob_list = numpy.array(maxprob_list)
    idx_list = numpy.array(idx_list)
    one_hot_labels = labels == idx_list

    return  maxprob_list, one_hot_labels

def calc_ece(config, preds, labels_oneh):
    """ A helper function to compute ECE.
        Args:
            config: configration dict
            pred: prediction score (fx)
            labels_oneh: one hot label (y)
        Return:
            ece: estimated (mean) calibration error by using config['ce_type'] strategy
    """

    saved_ece = []
    for _ in range(config['num_reps']):

        ce = calibrate(config, preds, labels_oneh)
        saved_ece.append(ce)
    ece = np.mean(saved_ece)
    return ece



def l2_between_grads_variance(cov_1, cov_2):
    assert len(cov_1) == len(cov_2)
    cov_1_values = [cov_1[key] for key in sorted(cov_1.keys())]
    cov_2_values = [cov_2[key] for key in sorted(cov_2.keys())]
    return (
            torch.cat(tuple([t.view(-1) for t in cov_1_values])) -
            torch.cat(tuple([t.view(-1) for t in cov_2_values]))
    ).pow(2).sum()


def compute_grads_variance(features, labels, classifier, alg):

    from backpack import backpack, extend
    from backpack.extensions import BatchGrad

    bce_extended = extend(nn.BCEWithLogitsLoss())

    logits = classifier(features)
    loss = bce_extended(logits.sum(dim=-1).unsqueeze(-1), labels.float())
    with backpack(BatchGrad()):
        loss.backward(
            inputs=list(classifier.parameters()), retain_graph=True, create_graph=True
        )

    dict_grads = OrderedDict(
        [
            (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
            for name, weights in classifier.named_parameters()
        ]
    )
    dict_grads_variance = {}
    for name, _grads in dict_grads.items():
        grads = _grads * labels.size(0)  # multiply by batch size
        env_mean = grads.mean(dim=0, keepdim=True)
        if alg != "NC":  # NotCentered
            grads = grads - env_mean
        if alg == "OD":  # OffDiagonal
            dict_grads_variance[name] = torch.einsum("na,nb->ab", grads,
                                                     grads) / (grads.size(0) * grads.size(1))
        else:
            dict_grads_variance[name] = (grads).pow(2).mean(dim=0)

    return dict_grads_variance


def set_new_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def update_penalty(epochs, epoch, max_lr):
    return (epoch + 1) / epochs * max_lr


def get_test_acc(model, test_env_loaders, device):
    accuracy = []
    for test_ld in test_env_loaders:

        total = 0
        correct = 0

        for (images, labels) in test_ld:
            images = images.to(device)
            labels = labels.to(device)
            total += images.shape[0]
            _, predicted = torch.max(model(images), 1)
            correct += (predicted == labels.view(-1)).sum()

        accuracy.append(correct / total)
    accuracy = torch.tensor(accuracy)

    return accuracy

def get_test_acc_ece_ace(model, test_env_loader, device):
    model.eval()

    total = 0
    correct = 0

    labels_all = []
    logits_all = []

    for (images, labels) in test_env_loader:
        images = images.to(device)
        labels = labels.to(device)
        total += images.shape[0]
        logits = model(images)
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels.view(-1)).sum()
        labels_all.append(labels)
        logits_all.append(logits)

    #calc calibration metirics
    ece_config = {}
    ece_config['num_reps'] = 100
    ece_config['norm'] = 1
    ece_config['ce_type'] = 'em_ece_bin'
    ece_config['num_bins'] = 10

    #ensure lavel is one hot
    all_probs = np.array(logits)
    all_labels = np.array(labels)
    maxprob_list, one_hot_labels = get_maxprob_and_onehot(all_probs, all_labels)
    ece_config['num_samples'] = int(len(one_hot_labels))

    #calc ece
    ece_config['ce_type'] = 'ew_ece_bin'
    ece = calc_ece(ece_config, maxprob_list, one_hot_labels)

    #calc ace
    ece_config['ce_type'] = 'em_ece_bin'
    ace = calc_ece(ece_config, maxprob_list, one_hot_labels)

    accuracy = correct / total

    return accuracy, ece, ace


def analyze_acc(acc_all: torch.Tensor):
    print(f"There are {acc_all.shape[0]} test environments.")
    worst_acc = torch.min(acc_all).detach().cpu().item()
    best_acc = torch.max(acc_all).detach().cpu().item()
    avg_acc = torch.mean(acc_all).detach().cpu().item()
    print("The best test accuracy is {:.4f}".format(best_acc))
    print("The worst test accuracy is {:.4f}".format(worst_acc))
    print("The accuracy difference is {:.4f}".format(best_acc - worst_acc))
    print("The average accuracy is {:.4f}".format(avg_acc))

    return [best_acc, worst_acc, avg_acc]
