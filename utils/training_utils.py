from collections import OrderedDict

import torch
import torch.autograd as autograd
import torch.nn as nn

from typing import Tuple
import numpy as np

from backpack import backpack, extend
from backpack.extensions import BatchGrad

from utils.optim_utils import LARS
from utils.calibration_metrics import CalibrationMetric
from models.EBD import EBD


def get_optimizer_scheduler(model, args):
    if "BLO" not in args.trainer:
        print("lr: ", args.lr)
        if args.optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif args.optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        elif args.optim == "lars":
            optimizer = LARS(optimizer=optimizer)
        
        if args.dataset == "CFMNIST" or args.dataset == "CMNIST":
            gamma = 1.0
        else:
            gamma = 0.1
            # gamma = 1.0

        if args.optim == "sgd":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(0.5 * args.epochs)],
                                                            gamma=gamma
                                                            )
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(1 * args.epochs)],
                                                            gamma=gamma
                                                            )
        print("lr: ", optimizer.param_groups[0]["lr"])
    else:
        optimizer = []
        for i in range(len(args.training_env)):
            if args.optim == "adam":
                base_optimizer_omega = torch.optim.Adam(model.omega_list[i].parameters(), lr=args.omega_lr)
            elif args.optim == "sgd":
                base_optimizer_omega = torch.optim.SGD(model.omega_list[i].parameters(), lr=args.omega_lr, momentum=0.9)
            elif args.optim == "lars":
                base_optimizer_omega = LARS(optimizer=base_optimizer_omega)
            optimizer.append(base_optimizer_omega)
        base_optimizer_phi = torch.optim.Adam(model.phi.parameters(), lr=args.lr)
        if args.optim == "sgd":
            base_optimizer_phi = torch.optim.SGD(model.phi.parameters(), lr=args.lr, momentum=0.9)
        elif args.optim == "lars":
            base_optimizer_phi = LARS(optimizer=base_optimizer_phi)
        optimizer.append(base_optimizer_phi)

        scheduler = []

        if args.dataset == "CFMNIST" or args.dataset == "CMNIST":
            gamma = 1.0
        else:
            gamma = 0.1

        for i in range(len(args.training_env)):
            if args.optim == "sgd":
                scheduler.append(torch.optim.lr_scheduler.MultiStepLR(optimizer[i],
                                                                  milestones=[int(0.5 * args.epochs)],
                                                                  gamma=gamma
                                                                  ))
            else:
                scheduler.append(torch.optim.lr_scheduler.MultiStepLR(optimizer[i],
                                                                #   milestones=[int(0.5 * args.epochs),
                                                                #               int(0.75 * args.epochs)],
                                                                  milestones=[int(0.75 * args.epochs)],
                                                                  gamma=gamma
                                                                  ))

        if args.optim == "sgd":
            scheduler.append(torch.optim.lr_scheduler.MultiStepLR(optimizer[-1],
                                                              milestones=[int(0.5 * args.epochs)],
                                                              gamma=gamma
                                                              ))
        else:
            scheduler.append(torch.optim.lr_scheduler.MultiStepLR(optimizer[-1],
                                                                # milestones=[int(0.5 * args.epochs),
                                                                            # int(0.75 * args.epochs)],
                                                                milestones=[int(0.75 * args.epochs)],
                                                                gamma=gamma
                                                                ))

    return optimizer, scheduler


def get_device(tensor):
    device_idx = tensor.get_device()
    device = "cuda:" + str(device_idx) if device_idx > 0 else "cpu"
    return device


def criterion(logits, y):
    return torch.nn.CrossEntropyLoss()(logits, y.view(-1))
    # return torch.nn.functional.binary_cross_entropy_with_logits(logits, y.float())


def mean_accuracy(logits, y):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == y.view(-1)).sum()
    return correct / y.shape[0]
    # preds = (logits > 0.).float()
    # return ((preds - y).abs() < 1e-2).float().mean()

def penalty_v1(logits, y):
    device = get_device(logits)
    scale = torch.tensor(1.).to(device).requires_grad_()
    loss = criterion(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)

def penalty_v1b(logits, y):
    device = get_device(logits)
    scale = torch.tensor(1.).to(device).requires_grad_()
    loss_1 = criterion(logits[::2] * scale, y[::2])
    loss_2 = criterion(logits[1::2] * scale, y[1::2])
    grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
    grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
    result = torch.sum(grad_1 * grad_2)
    return result


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

def penalty_fishr(logits, y, len_minibatches, optimizer, model, num_domains, ema_per_domain):
    dict_grads = _get_grads(logits, y, optimizer, model)
    grads_var_per_domain = _get_grads_var_per_domain(dict_grads, len_minibatches, num_domains, ema_per_domain)
    return _compute_distance_grads_var(grads_var_per_domain, num_domains)

def _get_grads(logits, y, optimizer, model):
    bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
    classifier = model.omega_list[0]
    optimizer.zero_grad()
    loss = bce_extended(logits, y).sum()
    with backpack(BatchGrad()):
        loss.backward(
            inputs=list(classifier.parameters()), retain_graph=True, create_graph=True
        )

    # compute individual grads for all samples across all domains simultaneously
    dict_grads = OrderedDict(
        [
            (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
            for name, weights in classifier.named_parameters()
        ]
    )
    return dict_grads

def _get_grads_var_per_domain(dict_grads, len_minibatches, num_domains, ema_per_domain):
    # grads var per domain
    grads_var_per_domain = [{} for _ in range(num_domains)]
    for name, _grads in dict_grads.items():
        all_idx = 0
        for domain_id, bsize in enumerate(len_minibatches):
            env_grads = _grads[all_idx:all_idx + bsize]
            all_idx += bsize
            env_mean = env_grads.mean(dim=0, keepdim=True)
            env_grads_centered = env_grads - env_mean
            grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

    # moving average
    for domain_id in range(num_domains):
        grads_var_per_domain[domain_id] = ema_per_domain[domain_id].update(
            grads_var_per_domain[domain_id]
        )

    return grads_var_per_domain

def _compute_distance_grads_var(grads_var_per_domain, num_domains):

    # compute gradient variances averaged across domains
    grads_var = OrderedDict(
        [
            (
                name,
                torch.stack(
                    [
                        grads_var_per_domain[domain_id][name]
                        for domain_id in range(num_domains)
                    ],
                    dim=0
                ).mean(dim=0)
            )
            for name in grads_var_per_domain[0].keys()
        ]
    )

    penalty = 0
    for domain_id in range(num_domains):
        penalty += l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
    return penalty / num_domains

def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()

class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.named_parameters = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.named_parameters[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.named_parameters[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data
    

def init_config():
    """Initializer of Configration for CalibrationMetric."""

    n_bins = 10
    alpha = 1.0
    beta = 1.0
    config = {}
    config['num_reps'] = 100
    config['num_bins'] = n_bins
    config['split'] = ''
    config['norm'] = 1
    config['calibration_method'] = 'no_calibration'
    config['bin_method'] = ''
    config['d'] = 1
    config['alpha'] = alpha
    config['beta'] = beta
    config['a'] = alpha
    config['b'] = beta
    config['dataset'] = 'polynomial'
    config['ce_type'] = 'ew_ece_bin'
    config['num_samples'] = 5
    
    return config


def get_maxprob_and_onehot(probs: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    maxprob_list = []
    idx_list = []

    for i in range(len(probs)):
        maxprob_list.append(np.max(probs[i]))
        idx_list.append(np.argmax(probs[i]))

    maxprob_list = np.array(maxprob_list)
    idx_list = np.array(idx_list)
    labels = labels.reshape(-1)
    one_hot_labels = labels == idx_list

    return  maxprob_list, one_hot_labels

def build_calibration_metric(config):
    """Builder of CalibrationMetric."""

    if config == None:
        config = init_config()

    ce_type = config['ce_type']
    num_bins = config['num_bins']
    bin_method = config['bin_method']
    norm = config['norm']

    # [4] Call CalibrationMetric constructor
    cm = CalibrationMetric(ce_type, num_bins, bin_method, norm)
    return cm

def calibrate(config, preds, labels_oneh):
    """Compute estimated calibration error.
        Args:
            config: configration dict
            pred: prediction score (fx)
            labels_oneh: one hot label (y)
        Return:
            ce: calibration error by using config['ce_type'] strategy
    """

    num_samples = config['num_samples']
    scores = preds.reshape((num_samples, 1))
    raw_labels = labels_oneh.reshape((num_samples, 1))

    # [3] Call build_calibration_metric function
    cm = build_calibration_metric(config)
    ce = cm.compute_error(scores, raw_labels)

    return 100 * ce

def calc_ece_ace(config, tensor_logits, tensor_labels):
    if tensor_logits.dim() == 1 or (tensor_logits.dim() == 2 and tensor_logits.shape[1] == 1):
        # 0 で埋めたテンソルと連結
        tensor_logits = torch.cat([torch.zeros_like(tensor_logits), tensor_logits], dim=1)


    probs = torch.nn.functional.softmax(tensor_logits, dim=1)
    labels = tensor_labels.detach().cpu().numpy()
    probs = probs.detach().cpu().numpy()

    maxprobs, one_hot_labels = get_maxprob_and_onehot(probs, labels)

    config['num_samples'] = int(len(one_hot_labels))

    def _ece(config, probs, labels_oneh):
        saved_ece = []
        for _ in range(config['num_reps']):
            ce = calibrate(config, probs, labels_oneh)
            saved_ece.append(ce)
        ece = np.mean(saved_ece)

        return ece

    config['ce_type'] = 'ew_ece_bin'
    ece = _ece(config, maxprobs, one_hot_labels)
    ece = torch.tensor(ece)

    config['ce_type'] = 'em_ece_bin'
    ace = _ece(config, maxprobs, one_hot_labels)
    ace = torch.tensor(ace)


    return ece, ace



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

        batch_acc = []

        for (images, labels) in test_ld:
            images = images.to(device)
            labels = labels.to(device)
            batch_acc.append(mean_accuracy(model(images), labels))

        env_acc = torch.tensor(batch_acc).mean()

        accuracy.append(env_acc)
    accuracy = torch.tensor(accuracy)

    return accuracy

def get_test_acc_ece_ace(model, test_env_loader, device):
    model.eval()

    total = 0
    correct = 0

    ece_list = []
    ace_list = []

    batch_acc = []

    for (images, labels) in test_env_loader:
        images = images.to(device)
        labels = labels.to(device)
        total += images.shape[0]
        logits = model(images)
        batch_acc.append(mean_accuracy(logits, labels))

        #calc calibration metirics
        ece_config = init_config()
        ece_config['num_reps'] = 100
        ece_config['norm'] = 1
        ece_config['ce_type'] = 'em_ece_bin'
        ece_config['num_bins'] = 10
        ece, ace = calc_ece_ace(ece_config, logits, labels)
        ece_list.append(ece)
        ace_list.append(ace)

    ece = torch.tensor(ece_list).mean()
    ace = torch.tensor(ace_list).mean()

    accuracy = torch.tensor(batch_acc).mean()

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
