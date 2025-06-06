import pdb
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
import os, sys, glob, time, subprocess
import h5py
from PIL import Image

from utils.general_utils import setup_seed
from datasets.domainbed_dataset import VLCS

class VLCS_FROM_DOMAINBED:
    def __init__(self, flags):
        self.flags = flags

    def data_loaders(self, **kwargs):
        hparams = {
            "data_augmentation": True,
        }
        test_envs = [self.flags.test_env]
        vlcs_class = VLCS(self.flags.data_dir, test_envs, hparams)

        self.train_loader = []
        for i, env_set in enumerate(vlcs_class):
            if i == self.flags.test_env:
                print("len of test env: ", len(env_set))
                self.test_loader = torch.utils.data.DataLoader(
                    dataset=env_set,
                    batch_size=self.flags.eval_batch_size,
                    shuffle=False,
                    num_workers=4)
                continue
            print("len of train env: ", len(env_set))
            train_ld = torch.utils.data.DataLoader(
                env_set,
                batch_size=self.flags.train_batch_size,
                shuffle=True,
                num_workers=4)

            self.train_loader.append(train_ld)


        
        return self.train_loader, self.test_loader
