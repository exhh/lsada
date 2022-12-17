import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import numpy as np
from torchvision import transforms

class DoubleDataset(Dataset):
    def __init__(self, root, dataA=None, dataB=None, label_type='labels', crop_size=None, split='train', mode='train',
        use_tanh=True):
        self.root = root
        self.split = split
        self.mode = mode
        self.label_type = label_type
        self.crop_size = crop_size
        self.use_tanh = use_tanh
        self.dataA, self.dataB = dataA, dataB
        self.dataA_ids = sorted(self.collect_ids(self.dataA))
        self.dataB_ids = sorted(self.collect_ids(self.dataB))
        self.dataA_size, self.dataB_size = len(self.dataA_ids), len(self.dataB_ids)
        self.permute_data()
        assert self.dataA_size > 0, "Dataset " + dataA + " does not exist!"
        assert self.dataB_size > 0, "Dataset " + dataB + " does not exist!"

    def permute_data(self):
        self.randperm = torch.randperm(self.dataB_size)

    def collect_ids(self, data):
        im_dir = os.path.join(self.root, data, 'images', self.split)
        data_ids = []
        for filename in os.listdir(im_dir):
            if filename.endswith('.png'):
                data_ids.append(filename[:-4])
        return data_ids

    def img_path(self, data, id):
        return os.path.join(self.root, data, 'images', self.split, id+'.png')

    def label_path(self, data, id):
        return os.path.join(self.root, data, self.label_type, self.split, id+'_label.png')

    def joint_transform(self, image, label, image2=None, label2=None):
        if self.mode == 'train':
            # Random crop (due to limited GPU memory)
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
            image = TF.crop(image, i, j, h, w)
            label = TF.crop(label, i, j, h, w)

            if image2 is not None:
                # Random crop (due to limited GPU memory)
                i, j, h, w = transforms.RandomCrop.get_params(image2, output_size=(self.crop_size, self.crop_size)) # image and image2 can have different sizes
                image2 = TF.crop(image2, i, j, h, w)
                label2 = TF.crop(label2, i, j, h, w)

        elif self.mode == 'test' or self.mode == 'validation':
            pass

        image = TF.to_tensor(image)
        if self.use_tanh:
            image = TF.normalize(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        label = torch.from_numpy(np.array(label, np.int64, copy=False)).unsqueeze(0)

        if image2 is not None:
            image2 = TF.to_tensor(image2)
            if self.use_tanh:
                image2 = TF.normalize(image2, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            label2 = torch.from_numpy(np.array(label2, np.int64, copy=False)).unsqueeze(0)
        return image, label, image2, label2

    def __getitem__(self, index):
        id_A = self.dataA_ids[index % self.dataA_size]
        img_A_name = self.img_path(self.dataA, id_A)
        img_A_name_noext = img_A_name.rsplit('/',1)[1].split('.')[0]
        img_A = Image.open(img_A_name)
        label_A = Image.open(self.label_path(self.dataA, id_A)).convert('L')
        id_B = self.dataB_ids[self.randperm[index % self.dataB_size]]
        img_B_name = self.img_path(self.dataB, id_B)
        img_B_name_noext = img_B_name.rsplit('/',1)[1].split('.')[0]
        img_B = Image.open(img_B_name)
        label_B = Image.open(self.label_path(self.dataB, id_B)).convert('L')
        if index == len(self) - 1:
            self.permute_data()

        img_A, label_A, img_B, label_B = self.joint_transform(img_A, label_A, img_B, label_B)
        return img_A, label_A, img_A_name_noext, img_B, label_B, img_B_name_noext

    def __len__(self):
        if self.mode == 'test' and self.split == 'test':
            return min(self.dataA_size, self.dataB_size)
        else:
            return max(self.dataA_size, self.dataB_size)


class SingleDataset(Dataset):
    def __init__(self, root, dataA=None, label_type='labels', crop_size=None, split='train', mode='train',
        use_tanh=False):
        self.root = root
        self.split = split
        self.mode = mode
        self.label_type = label_type
        self.crop_size = crop_size
        self.use_tanh = use_tanh
        self.dataA = dataA
        self.dataA_ids = self.collect_ids(self.dataA)
        self.dataA_size = len(self.dataA_ids)
        assert self.dataA_size > 0, "Dataset " + dataA + " does not exist!"

    def collect_ids(self, data):
        im_dir = os.path.join(self.root, data, 'images', self.split)
        data_ids = []
        for filename in os.listdir(im_dir):
            if filename.endswith('.png'):
                data_ids.append(filename[:-4])
        return data_ids

    def img_path(self, data, id):
        return os.path.join(self.root, data, 'images', self.split, id+'.png')

    def label_path(self, data, id):
        return os.path.join(self.root, data, self.label_type, self.split, id+'_label.png')

    def joint_transform(self, image, label):
        if self.mode == 'train':
            # Random crop (due to limited GPU memory)
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
            image = TF.crop(image, i, j, h, w)
            label = TF.crop(label, i, j, h, w)
        elif self.mode == 'test' or self.mode == 'validation':
            pass

        image = TF.to_tensor(image)
        if self.use_tanh:
            image = TF.normalize(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        label = torch.from_numpy(np.array(label, np.int64, copy=False)).unsqueeze(0)
        return image, label

    def __getitem__(self, index):
        id_A = self.dataA_ids[index % self.dataA_size]
        img_A_name = self.img_path(self.dataA, id_A)
        img_A_name_noext = img_A_name.rsplit('/',1)[1].split('.')[0]
        img_A = Image.open(img_A_name)
        label_A = Image.open(self.label_path(self.dataA, id_A)).convert('L')

        img_A, label_A = self.joint_transform(img_A, label_A)
        return img_A, label_A, img_A_name_noext

    def __len__(self):
        return self.dataA_size
