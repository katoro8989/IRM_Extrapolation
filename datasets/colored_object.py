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


def get_transform_coco(num_classes):
    if num_classes == 2:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.37493148, 0.21778074, 0.23026027],
                [0.10265636, 0.20582178, 0.21669184])
        ])
    elif num_classes == 10:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.34489644, 0.30505344, 0.3762387 ],
                [0.26109827, 0.2823534, 0.32291284])
        ])
    else:
        raise Exception
    return transform


class COCODataset(object):
    def __init__(self, x_array, y_array, env_array, transform, sp_array=None):
        self.x_array = x_array
        self.y_array = y_array[:, None]
        self.env_array = env_array[:, None]
        self.sp_array = sp_array[:, None]
        self.transform = transform
        assert len(self.x_array) == len(self.y_array)
        assert len(self.y_array) == len(self.env_array)

    def __len__(self):
        return len(self.x_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.env_array[idx]
        if self.sp_array is not None:
            sp = self.sp_array[idx]
        else:
            sp = None
        img = self.x_array[idx]
        img = (img *255).astype(np.uint8)
        img = img.transpose(1, 2, 0)
        img = Image.fromarray(img)
        x = self.transform(img)

        return x,y

def get_coco_handles(num_classes=2, sp_ratio_list=None, noise_ratio=0, dataset='colour', train_test=None, flags=None, grayscale_model=False):
    data_dir = "/gs/bs/tge-24IJ0078/dataset/SPCOCO/coco"
    if dataset == 'places':
        dataset_name = 'cocoplaces_vf_{}_{}'.format(num_classes, confounder_strength)
        original_dirname = os.path.join(data_dir, dataset_name)
    elif dataset == 'colour':

        if grayscale_model:
            dataset_name = 'cocogrey__class_{}_noise_{}_sz_{}'.format(
                num_classes,
                noise_ratio,
                flags.image_scale)
        else:
            dataset_name = 'cococolours_vf_num_class_{}_sp_{}_noise_{}_sz_{}'.format(
                num_classes,
                "_".join([str(x) for x in sp_ratio_list]),
                noise_ratio,
                flags.image_scale)
        original_dirname = os.path.join(data_dir, dataset_name)



    dirname = os.path.join(data_dir,  dataset_name)

    print('Copying data over, this will be worth it, be patient ...', end=' ')
    subprocess.call(['rsync', '-r', original_dirname, data_dir])
    print('Done!')

    if train_test == "train":
        train_file = h5py.File(dirname+'/train.h5py', mode='r')
        # print("what", dirname+'/train.h5py')
        return (train_file, None, None, None, None)
    elif train_test == "test":
        id_test_file = h5py.File(dirname+'/idtest.h5py', mode='r')
        return (id_test_file, None, None, None, None)
    else:
        raise Exception

def get_spcoco_dataset(sp_ratio_list=None, noise_ratio=None, num_classes=None, flags=None):
    coco_transform = get_transform_coco(2)
    train_data_handle, _, _, _, _ = get_coco_handles(
        num_classes=num_classes,
        sp_ratio_list=sp_ratio_list,
        noise_ratio=noise_ratio,
        dataset='colour', train_test="train", flags=flags)
    # shuffle train
    train_x_array = train_data_handle["images"][:]
    train_y_array = train_data_handle["y"][:]
    train_env_array = train_data_handle["e"][:]
    train_sp_array = train_data_handle["g"][:]
    print(train_x_array.shape)
    perm = np.random.permutation(
        range(train_x_array.shape[0]))
    # split train and validation
    train_perm = perm[:int(0.8 * len(perm))]
    val_perm = perm[int(0.8 * len(perm)):]

    coco_dataset_train = COCODataset(
        x_array=train_x_array[train_perm],
        y_array=train_y_array[train_perm],
        env_array=train_env_array[train_perm],
        transform=coco_transform,
        sp_array=train_sp_array[train_perm])

    coco_dataset_val = COCODataset(
        x_array=train_x_array[val_perm],
        y_array=train_y_array[val_perm],
        env_array=train_env_array[val_perm],
        transform=coco_transform,
        sp_array=train_sp_array[val_perm])

    test_data_handle, _, _, _, _ = get_coco_handles(
        num_classes=num_classes,
        sp_ratio_list=sp_ratio_list,
        noise_ratio=noise_ratio,
        dataset='colour',
        train_test="test",
        flags=flags)

    coco_dataset_test = COCODataset(
        x_array=test_data_handle["images"][:],
        y_array=test_data_handle["y"][:],
        env_array=test_data_handle["e"][:],
        transform=coco_transform,
        sp_array=test_data_handle["g"][:])

    return coco_dataset_train, coco_dataset_val, coco_dataset_test



class COCOcolor_LYPD:
    def __init__(self, flags):
        self.flags = flags

    def data_loaders(self, **kwargs):
        setup_seed(1)
        sp_ratio_list = [float(x) for x in "0.999_0.7_0.1".split("_")]
        self.train_dataset, self.val_dataset, self.test_dataset = get_spcoco_dataset(
            sp_ratio_list=sp_ratio_list,
            noise_ratio=0.05,
            num_classes=2,
            flags=self.flags)
        
        env_sets = []
        for i, env_p in enumerate(self.flags.training_env):
            env_sets.append(Subset(self.train_dataset, np.where(self.train_dataset.env_array == i)[0]))

        setup_seed(self.flags.seed)

        self.train_loader = []
        for env_set in env_sets:
            train_ld = torch.utils.data.DataLoader(
                env_set,
                batch_size=self.flags.train_batch_size,
                shuffle=True,
                num_workers=4)

            self.train_loader.append(train_ld)
        
        env_sets = []
        for i, env_p in enumerate(self.flags.training_env):
            env_sets.append(Subset(self.val_dataset, np.where(self.val_dataset.env_array == i)[0]))

        self.val_loader = []
        for env_set in env_sets:
            val_ld = torch.utils.data.DataLoader(
                env_set,
                batch_size=self.flags.eval_batch_size,
                shuffle=False,
                num_workers=4)

            self.val_loader.append(val_ld)

        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.flags.eval_batch_size,
            shuffle=False,
            num_workers=4)

        
        return self.train_loader, self.val_loader, self.test_loader

        # self.train_loader_iter = iter(self.train_loader)
        # self.val_loader = iter(self.val_loader)
        # self.test_loader_iter = iter(self.test_loader)
