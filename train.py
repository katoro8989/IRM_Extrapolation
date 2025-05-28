import logging
import importlib
import os
import time

import torch
import wandb

from args import parse_args

from utils.training_utils import get_test_acc, analyze_acc, get_optimizer_scheduler, get_test_acc_ece_ace
from utils.general_utils import parse_configs_file, setup_seed, save_checkpoint, get_exp_name, get_data_per_epoch, plot_trajectory
import datasets
import models


def main():
    args = parse_args()
    parse_configs_file(args)

    setup_seed(args.seed)

    # All kinds of dir names
    exp_name = get_exp_name(args)
    result_sub_dir = os.path.join(
        args.result_dir, exp_name
    )
    graph_dir = os.path.join(
        args.result_dir, "graphs"
    )

    os.makedirs(result_sub_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(result_sub_dir, "setup.log"), "a")
    )
    logger.info(args)

    # wandb
    wandb.init(config=vars(args), 
            project=args.wandb_project_name,
            entity=args.wandb_entity_name,
            )

    # Select GPUs
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    device = torch.device(f"cuda:{gpu_list[0]}" if use_cuda else "cpu")

    env_num = 1 if "BLO" not in args.trainer else len(args.training_env)

    if "MNIST" in args.dataset:
        print("Using MLP")
        model = models.__dict__[args.arch](
            env_num=env_num,
            use_color=True if len(args.training_color_env) > 0 else False
        )
    elif args.dataset == "PACS_FROM_DOMAINBED":
        print("Using ResNet18")
        model = models.ResNet.resnet18_sepfc_us(
                pretrained=False,
                num_classes=7,
                env_num=env_num)
    elif args.dataset == "VLCS_FROM_DOMAINBED":
        print("Using ResNet18")
        model = models.ResNet.resnet18_sepfc_us(
                pretrained=False,
                num_classes=5,
                env_num=env_num)
    elif args.dataset == "DomainNet_FROM_DOMAINBED":
        print("Using ResNet50")
        model = models.ResNet.resnet50_sepfc_us(
                pretrained=True,
                num_classes=345,
                env_num=env_num)
    elif args.dataset == "TerraIncognita_FROM_DOMAINBED":
        print("Using ResNet50")
        model = models.ResNet.resnet50_sepfc_us(
                pretrained=True,
                num_classes=10,
                env_num=env_num)   
    else:
        print("Using ResNet18")
        model = models.ResNet.resnet18_sepfc_us(
                pretrained=False,
                num_classes=1,
                env_num=env_num)

    model.load_device(device)

    # Dataloader
    D = datasets.__dict__[args.dataset](args)
    train_loader, test_loader = D.data_loaders()

    

    # setup_seed(args.seed)
    # # del model from gpu
    # del model
    # torch.cuda.empty_cache()
    # model = models.__dict__[args.arch](
    #     env_num=env_num,
    #     use_color=False
    # )
    # model.load_device(device)




    # Trainer
    trainer = importlib.import_module(f"trainer.{args.trainer}").train

    # Optimizer & Scheduler
    optimizer, scheduler = get_optimizer_scheduler(model, args)

    # Train loop
    best_epoch = 0
    best_diff = 100.0
    best_acc = 0.0
    args.start_epoch = 0
    data_per_epoch = [[] for _ in range(10)]
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        train_stat = trainer(
            model, args, device, train_loader, optimizer, scheduler, epoch
        )
        
        if isinstance(scheduler, list):
            for schedule in scheduler:
                schedule.step()
        else:
            scheduler.step()

        test_accuracy, test_ece, test_ace = get_test_acc_ece_ace(model, test_loader, device)

        wandb_log_dist = {
            "epoch": epoch,
            "train_loss": train_stat[1],
            "train_acc": train_stat[0],
            "train_ece": train_stat[6],
            "train_ace": train_stat[7],
            "penalty_v0": train_stat[2],
            "penalty_v1": train_stat[3],
            "penalty_stationary": train_stat[4],
            "test_acc": test_accuracy,
            "test_ece": test_ece,
            "test_ace": test_ace,
        }


        # if optimizer is list
        if isinstance(optimizer, list):
            phi_lr = optimizer[-1].param_groups[0]["lr"]
            omega_lr = optimizer[-2].param_groups[0]["lr"]
            step = optimizer[-1]._step_count
            wandb_log_dist["phi_lr"] = phi_lr
            wandb_log_dist["omega_lr"] = omega_lr
            wandb_log_dist["step"] = step
        else:
            lr = optimizer.param_groups[0]["lr"]
            wandb_log_dist["lr"] = lr
            step = optimizer._step_count
            wandb_log_dist["step"] = step


        wandb.log(wandb_log_dist)


if __name__ == "__main__":
    main()

