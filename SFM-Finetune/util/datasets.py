# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

import os, random, glob
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

random.seed(42)

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


## pretrain
class SeismicSet(data.Dataset):

    def __init__(self, path, input_size) -> None:
        super().__init__()
        # self.file_list = os.listdir(path)
        # self.file_list = [os.path.join(path, f) for f in self.file_list]
        self.get_file_list(path)
        self.input_size = input_size
        print(len(self.file_list))

    def __len__(self) -> int:
        return len(self.file_list)
        # return 100000

    def __getitem__(self, index):
        d = np.fromfile(self.file_list[index], dtype=np.float32)
        d = d.reshape(1, self.input_size, self.input_size)
        d = (d - d.mean()) / (d.std()+1e-6)

        # return to_transforms(d, self.input_size)
        return d,torch.tensor([1])

    def get_file_list(self, path):
        dirs = [os.path.join(path, f) for f in os.listdir(path)]
        self.file_list = dirs

        # for ds in dirs:
        #     if os.path.isdir(ds):
        #         self.file_list += [os.path.join(ds, f) for f in os.listdir(ds)]

        return random.shuffle(self.file_list)

class SeismicAugSet(data.Dataset):

    def __init__(self, path, input_size) -> None:
        super().__init__()
        # self.file_list = os.listdir(path)
        # self.file_list = [os.path.join(path, f) for f in self.file_list]
        self.get_file_list(path)
        self.input_size = input_size
        print(len(self.file_list))

    def __len__(self) -> int:
        return len(self.file_list)
        # return 100000

    def __getitem__(self, index):
        d = np.fromfile(self.file_list[index], dtype=np.float32)
        d = d.reshape(self.input_size, self.input_size)
        d = (d - d.mean()) / (d.std()+1e-6)
        d = torch.from_numpy(d).reshape(1,self.input_size,self.input_size)
        d = d.expand(3,self.input_size,self.input_size)
        # print(d.shape)
        # d = (d - d.mean()) / (d.std()+1e-6)
        # d1 = np.flip(d).copy()
        # d = torch.unsqueeze(torch.from_numpy(d),0).reshape(1,1,self.input_size,self.input_size)
        # d1 = torch.unsqueeze(torch.from_numpy(d1),0).reshape(1,1,self.input_size,self.input_size)
        # d2 = torch.cat((d,d1),dim=0)
        # return to_transforms(d, self.input_size)
        return d,torch.tensor([1])

    def get_file_list(self, path):
        dirs = [os.path.join(path, f) for f in os.listdir(path)]
        self.file_list = dirs

        # for ds in dirs:
        #     if os.path.isdir(ds):
        #         self.file_list += [os.path.join(ds, f) for f in os.listdir(ds)]

        return random.shuffle(self.file_list)

class RandomSeismicSet(data.Dataset):

    def __init__(self, path, input_size) -> None:
        super().__init__()
        # self.file_list = os.listdir(path)
        # self.file_list = [os.path.join(path, f) for f in self.file_list]
        self.input_size = input_size
        gx,m1,m2,m3 = np.fromfile('./data/f3_h462x951x651.dat',dtype=np.single),651,951,462
        gx = np.reshape(gx,(m1,m2,m3))
        gx = gx - np.mean(gx)
        gx = gx/np.std(gx)
        self.gx = gx
        # print(len(self.file_list))

    def __len__(self) -> int:
        # return len(self.file_list)
        return 3600

    def __getitem__(self, index):
        m1,m2,m3 = 651,951,462
        rc = np.random.randint(2)
        rt = np.random.randint(m3-self.input_size)
        if rc ==0:
            r1 = np.random.randint(m1-self.input_size)
            r2 = np.random.randint(m2)
            slide = self.gx[r1:r1+self.input_size,r2,rt:rt+self.input_size]
        else:
            r1 = np.random.randint(m1)
            r2 = np.random.randint(m2-self.input_size)
            slide = self.gx[r1,r2:r2+self.input_size,rt:rt+self.input_size]
        slide = np.transpose(slide)
        slide = torch.unsqueeze(torch.from_numpy(slide),0)
        # return to_transforms(d, self.input_size)
        return slide,torch.tensor([1])



def to_transforms(d, input_size):
    t = transforms.Compose([
        transforms.RandomResizedCrop(input_size,
                                     scale=(0.2, 1.0),
                                     interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    return t(d)






### fintune
class FacesSet(data.Dataset):
    # folder/train/data/**.dat, folder/train/label/**.dat
    # folder/test/data/**.dat, folder/test/label/**.dat
    def __init__(self,
                 folder,
                 shape=[224, 224],
                 is_train=True) -> None:
        super().__init__()
        self.shape = shape

        self.data_list = sorted(glob.glob(folder + 'seis/*.dat'))

        
        n = len(self.data_list)
        if is_train:
            self.data_list = self.data_list[:int(n * 0.8)]
            # with open('train_list.txt', 'w') as f:
            #     f.write(str(self.data_list))
        elif not is_train:
            self.data_list = self.data_list[int(n * 0.8):]
            # with open('valid_list.txt', 'w') as f:
            #     f.write(str(self.data_list))
        # else:
        #     self.data_list = sorted(glob.glob(folder + 'data/*.dat'))
        random.shuffle(self.data_list)
        self.label_list = [
            f.replace('/seis/', '/label/') for f in self.data_list
        ]

    def __getitem__(self, index):
        d = np.fromfile(self.data_list[index], np.float32)
        d = d.reshape([1] + self.shape)
        l = np.fromfile(self.label_list[index], np.float32).reshape(self.shape)-1
        l = l.astype(int)
        return d, l


    def __len__(self):
        return len(self.data_list)


class FacesSet2(data.Dataset):
    # folder/train/data/**.dat, folder/train/label/**.dat
    # folder/test/data/**.dat, folder/test/label/**.dat
    def __init__(self,
                 folder,
                 shape=[768, 768],
                 is_train=True) -> None:
        super().__init__()
        self.shape = shape

        # self.data_list = sorted(glob.glob(folder + 'seismic/*.dat'))
        self.data_list = [folder +'seismic/'+ str(f)+'.dat' for f in range(117)]


        
        n = len(self.data_list)
        if is_train:
            self.data_list = self.data_list[:100]
            # with open('train_list.txt', 'w') as f:
            #     f.write(str(self.data_list))
        elif not is_train:
            self.data_list = self.data_list[100:]
            # with open('valid_list.txt', 'w') as f:
            #     f.write(str(self.data_list))
        # else:
        #     self.data_list = sorted(glob.glob(folder + 'data/*.dat'))
        # random.shuffle(self.data_list)
        self.label_list = [
            f.replace('/seismic/', '/label/') for f in self.data_list
        ]

    def __getitem__(self, index):
        d = np.fromfile(self.data_list[index], np.float32)
        d = d.reshape([1] + self.shape)
        l = np.fromfile(self.label_list[index], np.float32).reshape(self.shape)-1
        l = l.astype(int)
        return torch.tensor(d), torch.tensor(l)


    def __len__(self):
        return len(self.data_list)


class FacesSetF3(data.Dataset):
    # folder/train/data/**.dat, folder/train/label/**.dat
    # folder/test/data/**.dat, folder/test/label/**.dat
    def __init__(self,
                 folder,
                 shape=[240, 688],
                 is_train=True) -> None:
        super().__init__()
        self.shape = shape

        # self.data_list = sorted(glob.glob(folder + 'seismic/*.dat'))
        self.data_list = [folder +'seismic/'+ str(f)+'.dat' for f in range(80)]

        
        n = len(self.data_list)
        if is_train:
            self.data_list = self.data_list[:54]
            # with open('train_list.txt', 'w') as f:
            #     f.write(str(self.data_list))
        elif not is_train:
            self.data_list = self.data_list[54:]
            # with open('valid_list.txt', 'w') as f:
            #     f.write(str(self.data_list))
        # else:
        #     self.data_list = sorted(glob.glob(folder + 'data/*.dat'))
        # random.shuffle(self.data_list)
        self.label_list = [
            f.replace('/seismic/', '/label/') for f in self.data_list
        ]

    def __getitem__(self, index):
        d = np.fromfile(self.data_list[index], np.float32)
        d = d.reshape([1] + self.shape)
        l = np.fromfile(self.label_list[index], np.float32).reshape(self.shape)
        l = l.astype(int)
        return torch.tensor(d), torch.tensor(l)


    def __len__(self):
        return len(self.data_list)


class SaltSet(data.Dataset):
    # folder/train/data/**.dat, folder/train/label/**.dat
    # folder/test/data/**.dat, folder/test/label/**.dat
    def __init__(self,
                 folder,
                 shape=[224, 224],
                 is_train=True) -> None:
        super().__init__()
        self.shape = shape

        # self.data_list = sorted(glob.glob(folder + 'images/*.dat'))
        self.data_list = [folder +'seismic/'+ str(f)+'.dat' for f in range(4000)]


        
        n = len(self.data_list)
        if is_train:
            self.data_list = self.data_list[:3500]
            # with open('train_list.txt', 'w') as f:
            #     f.write(str(self.data_list))
        elif not is_train:
            self.data_list = self.data_list[3500:]
            # with open('valid_list.txt', 'w') as f:
            #     f.write(str(self.data_list))
        # else:
        #     self.data_list = sorted(glob.glob(folder + 'data/*.dat'))
        # random.shuffle(self.data_list)
        self.label_list = [
            f.replace('/seismic/', '/label/') for f in self.data_list
        ]

    def __getitem__(self, index):
        d = np.fromfile(self.data_list[index], np.float32)
        d = d.reshape([1] + self.shape)
        l = np.fromfile(self.label_list[index], np.float32).reshape(self.shape)
        l = l.astype(int)
        return torch.tensor(d), torch.tensor(l)



    def __len__(self):
        return len(self.data_list)



class CO2Set(data.Dataset):
    # folder/train/data/**.dat, folder/train/label/**.dat
    # folder/test/data/**.dat, folder/test/label/**.dat
    def __init__(self,
                 folder,
                 shape=[192, 192],
                 is_train=True) -> None:
        super().__init__()
        self.shape = shape
        # self.data_list = sorted(glob.glob(folder + 'images/*.dat'))
        self.data_list = [folder +'train/sx/'+ str(f)+'.dat' for f in range(4576)]

        n = len(self.data_list)
        if is_train:
            self.data_list = self.data_list
        elif not is_train:
            self.data_list = [folder +'Test/sx/'+ str(f)+'.dat' for f in range(290)]
        self.label_list = [
            f.replace('/sx/', '/lx/') for f in self.data_list
        ]

    def __getitem__(self, index):
        d = np.fromfile(self.data_list[index], np.float32)
        d = d.reshape([1] + self.shape)
        d = d - d.mean()
        d = d / d.std()
        l = np.fromfile(self.label_list[index], np.float32).reshape(self.shape)
        l = l.astype(int)
        return torch.tensor(d), torch.tensor(l)



    def __len__(self):
        return len(self.data_list)



class SaltSet3(data.Dataset):
    # folder/train/data/**.dat, folder/train/label/**.dat
    # folder/test/data/**.dat, folder/test/label/**.dat
    def __init__(self,
                 folder,
                 shape=[224, 224],
                 is_train=True) -> None:
        super().__init__()
        self.shape = shape

        # self.data_list = sorted(glob.glob(folder + 'images/*.dat'))
        self.data_list = [folder +'seismic/'+ str(f)+'.dat' for f in range(4000)]


        
        n = len(self.data_list)
        if is_train:
            self.data_list = self.data_list[:3500]
            # with open('train_list.txt', 'w') as f:
            #     f.write(str(self.data_list))
        elif not is_train:
            self.data_list = self.data_list[3500:]
            # with open('valid_list.txt', 'w') as f:
            #     f.write(str(self.data_list))
        # else:
        #     self.data_list = sorted(glob.glob(folder + 'data/*.dat'))
        # random.shuffle(self.data_list)
        self.label_list = [
            f.replace('/seismic/', '/label/') for f in self.data_list
        ]

    def __getitem__(self, index):
        d = np.fromfile(self.data_list[index], np.float32)
        d = d.reshape([1] + self.shape)
        d = d.repeat(3,axis=0)
        
        l = np.fromfile(self.label_list[index], np.float32).reshape(self.shape)
        l = l.astype(int)
        
        return torch.tensor(d), torch.tensor(l)



    def __len__(self):
        return len(self.data_list)


class InterpolationSet(data.Dataset):
    # folder/train/data/**.dat, folder/train/label/**.dat
    # folder/test/data/**.dat, folder/test/label/**.dat
    def __init__(self,
                 folder,
                 shape=[224, 224],
                 is_train=True) -> None:
        super().__init__()
        self.shape = shape
        self.data_list = [folder + str(f)+'.dat' for f in range(6000)]
        n = len(self.data_list)
        if is_train:
            self.data_list = self.data_list
        elif not is_train:
            self.data_list = [folder + str(f)+'.dat' for f in range(8000,8100)]
        self.label_list = self.data_list

    # def generate_mask(self,input_tensor, ratio):
    #     mask = torch.zeros_like(input_tensor)
    #     mask[:, :, torch.randperm(mask.size(2))[:int(mask.size(2) * ratio)]] = 1
    #     return mask
    # def generate_mask(self,input_tensor, ratio):
    #     mask = torch.zeros_like(input_tensor)
    #     indices = torch.randperm(mask.size(2))[:int(mask.size(2) * ratio)]
    #     sorted_indices = torch.sort(indices)[0]  # 对索引进行排序
    #     mask[:, :, sorted_indices] = 1
    #     return mask
    
    def __getitem__(self, index):
        d = np.fromfile(self.data_list[index], np.float32)
        d = d.reshape([1] + self.shape)
        
        # dl = self.generate_mask(torch.tensor(d),0.5)*d
        # l = np.fromfile(self.label_list[index], np.float32).reshape(self.shape)
        # l = l.astype(int)
        return torch.tensor(d), torch.tensor(d)


    def __len__(self):
        return len(self.data_list)
        # return 10000



class DenoiseSet(data.Dataset):
    # folder/train/data/**.dat, folder/train/label/**.dat
    # folder/test/data/**.dat, folder/test/label/**.dat
    def __init__(self,
                 folder,
                 shape=[224, 224],
                 is_train=True) -> None:
        super().__init__()
        self.shape = shape

        # self.data_list = sorted(glob.glob(folder + 'noisenew/*.dat'))
        self.data_list = [folder+'noisenew/'+ str(f)+'.dat' for f in range(2000)]
        n = len(self.data_list)
        if is_train:
            self.data_list = self.data_list
            self.label_list = [f.replace('/noisenew/', '/clean/') for f in self.data_list]
        elif not is_train:
            self.data_list = [folder+'fieldnoise/'+ str(f)+'.dat' for f in range(1600)]
            self.label_list = self.data_list

    def __getitem__(self, index):
        d = np.fromfile(self.data_list[index], np.float32)
        d = d.reshape([1] + self.shape)
        # d = (d - d.mean())/d.std()
        l = np.fromfile(self.label_list[index], np.float32)
        l = l.reshape([1] + self.shape)
        # l = (l - d.mean())/l.std()
        
        

        return torch.tensor(d), torch.tensor(l)


    def __len__(self):
        return len(self.data_list)
        # return 1800
    

class SuperResolutionSet(data.Dataset):
    # folder/train/data/**.dat, folder/train/label/**.dat
    # folder/test/data/**.dat, folder/test/label/**.dat
    def __init__(self,
                 folder,
                 shape=[128, 128],
                 shape2=[256,256],
                 is_train=True) -> None:
        super().__init__()
        self.shape = shape
        self.shape2 = shape2
        

        # self.data_list = sorted(glob.glob(folder + 'low/*.dat'))
        self.data_list = [folder+'low/'+ str(f).zfill(4)+'.dat' for f in range(3200)]
        n = len(self.data_list)
        if is_train:
            self.data_list = [folder+'low/'+ str(f).zfill(4)+'.dat' for f in range(2400)]
        elif not is_train:
            self.data_list = [folder+'low/'+ str(f).zfill(4)+'.dat' for f in range(2400,3200)]
            
        self.label_list = [f.replace('/low/', '/high/') for f in self.data_list]
    def __getitem__(self, index):
        d = np.fromfile(self.data_list[index], np.float32)
        d = d.reshape([1] + self.shape).transpose((0,2,1))
        l = np.fromfile(self.label_list[index], np.float32)
        l = l.reshape([1] + self.shape2).transpose((0,2,1))
        

        return torch.tensor(d), torch.tensor(l)


    def __len__(self):
        return len(self.data_list)
        # return 1800
    
    
class ReflectSet(data.Dataset):
    # folder/train/data/**.dat, folder/train/label/**.dat
    # folder/test/data/**.dat, folder/test/label/**.dat
    def __init__(self,
                 folder,
                 shape=[224, 224],
                 is_train=True) -> None:
        super().__init__()
        self.shape = shape

        # self.data_list = sorted(glob.glob(folder + 'noise/*.dat'))
        self.data_list = [folder+'noise/'+ str(f)+'.dat' for f in range(2200)]


        
        n = len(self.data_list)
        if is_train:
            self.data_list = self.data_list
        elif not is_train:
            self.data_list = [folder+'noise/'+ str(f)+'.dat' for f in range(2200,2400)]
        # else:
        #     self.data_list = sorted(glob.glob(folder + 'data/*.dat'))
        # random.shuffle(self.data_list)
        self.label_list = [
            f.replace('/noise/', '/reflect/') for f in self.data_list
        ]

    def __getitem__(self, index):
        d = np.fromfile(self.data_list[index], np.float32)
        d = d- d.mean()
        d = d/(d.std()+1e-6)
        d = d.reshape([1] + self.shape)
        l = np.fromfile(self.label_list[index], np.float32)
        l = l-l.mean()
        l = l/(l.std()+1e-6)
        l = l.reshape([1] + self.shape)
        

        return torch.tensor(d), torch.tensor(l)


    def __len__(self):
        return len(self.data_list)
        # return 1800


class FacesFake3DSet(data.Dataset):
    # folder/train/data/**.dat, folder/train/label/**.dat
    # folder/test/data/**.dat, folder/test/label/**.dat
    def __init__(self,
                 folder,
                 shape=[224, 224],
                 is_train=True) -> None:
        super().__init__()
        self.shape = shape

        self.data_list = sorted(glob.glob(folder + 'seis/*.dat'))

        
        n = len(self.data_list)
        if is_train:
            self.data_list = self.data_list[:int(n * 0.8)]
            # with open('train_list.txt', 'w') as f:
            #     f.write(str(self.data_list))
        elif not is_train:
            self.data_list = self.data_list[int(n * 0.8):]
            # with open('valid_list.txt', 'w') as f:
            #     f.write(str(self.data_list))
        # else:
        #     self.data_list = sorted(glob.glob(folder + 'data/*.dat'))
        random.shuffle(self.data_list)
        self.label_list = [
            f.replace('/seis/', '/label/') for f in self.data_list
        ]

    def __getitem__(self, index):
        d = np.fromfile(self.data_list[index], np.float32)
        d = d.reshape([1] + self.shape)
        l = np.fromfile(self.label_list[index], np.float32).reshape(self.shape)
        l = l.astype(int)
        return d, l


    def __len__(self):
        return len(self.data_list)

if __name__ == "__main__":
    s = SaltSet('/home/hlsheng/mae_data/finetune/salt/')

    # print(s)

    # d = np.fromfile()
