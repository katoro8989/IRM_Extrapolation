import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")

    # primary
    parser.add_argument(
        "--configs", type=str, default=None, help="configs file",
    )

    # General Training Settings

    parser.add_argument('--dataset', type=str, default="CMNIST", choices=["CMNIST", "CFMNIST"])

    parser.add_argument('--epochs', type=int, default=200)

    parser.add_argument('--train_batch_size', type=int, default=128)

    parser.add_argument('--eval_batch_size', type=int, default=128)

    parser.add_argument('--seed', default=37, type=int, help='Random seed (default: 37)')

    parser.add_argument('--arch', default="MLP390", type=str)

    parser.add_argument('--trainer', default="IRM", type=str, help="IRM, ERM, Gray, BLO, IRMV0, VREx")

    parser.add_argument('--data_dir', type=str, default="./data", help="path to datasets")

    parser.add_argument('--hidden-dim', type=int, default=390)

    parser.add_argument('--optim', default="adam", choices=["adam", "lamb", "sam", "lars"])

    parser.add_argument('--save', action="store_true", help="Do you want to save checkpoints?")

    # Training environments

    parser.add_argument('--training_env', default=[0.1, 0.2], nargs="+", type=float)

    parser.add_argument('--training_class_env', default=[], nargs="+", type=float)

    parser.add_argument('--training_color_env', default=[], nargs="+", type=float)

    parser.add_argument('--test_env', default=0.9, type=float)

    parser.add_argument('--label_flip_p', default=0.25, type=float)

    parser.add_argument('--wd', type=float, default=0.00110794568)

    # Training Method Details

    parser.add_argument('--penalty_weight', type=float, default=91257.18613115903)

    parser.add_argument('--lr', type=float, default=0.002)

    parser.add_argument('--warm_start', default=10, type=int)

    ## For BLO only
    parser.add_argument('--omega_lr', type=float, default=0.002)

    # For BIRM only
    parser.add_argument('--prior_sd_coef', type=int, default=1200)
    parser.add_argument('--data_num', type=int, default=50000)

    # Other settings

    parser.add_argument('--print_freq', type=int, default=100)

    parser.add_argument('--result_dir', type=str, default="./results")

    parser.add_argument('--wandb_project_name', type=str, default="BLO")
    parser.add_argument('--wandb_entity_name', type=str, default="katoro13")

    parser.add_argument(
        "--gpu", type=str, default="0", help="Comma separated list of GPU ids"
    )

    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
    )

    return parser.parse_args()

