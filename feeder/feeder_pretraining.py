import time
import torch

import numpy as np
np.set_printoptions(threshold=np.inf)
import random

try:
    from feeder import augmentations
except:
    import augmentations


class Feeder(torch.utils.data.Dataset):
    """ 
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(self,
                 data_path,
                 num_frame_path,
                 l_ratio,
                 input_size,
                 input_representation,
                 mmap=True):

        self.data_path = data_path
        self.num_frame_path = num_frame_path
        self.input_size = input_size
        self.input_representation = input_representation
        self.crop_resize = True
        self.l_ratio = l_ratio

        self.load_data(mmap)

        self.N, self.C, self.T, self.V, self.M = self.data.shape
        self.S = self.V
        self.B = self.V
        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
        
        print(self.data.shape, len(self.number_of_frames))
        print("l_ratio", self.l_ratio)

    def load_data(self, mmap):
        # data: N C T V M

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # load num of valid frame length
        if self.num_frame_path != None:
            self.number_of_frames = np.load(self.num_frame_path)
        else:
            self.number_of_frames = np.ones(self.data.shape[0], dtype=np.int32)*50

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, index):
  
        # get raw input

        # input: C, T, V, M
        data_numpy = np.array(self.data[index])
        
        number_of_frames = self.number_of_frames[index]

        # temporal crop-resize
        data_numpy_v1 = augmentations.temporal_cropresize(data_numpy, number_of_frames, self.l_ratio, self.input_size)
        data_numpy_v2 = augmentations.temporal_cropresize(data_numpy,number_of_frames, self.l_ratio, self.input_size)


        if self.input_representation == "motion":
            # motion
            motion_v1 = np.zeros_like(data_numpy_v1)
            motion_v1[:, :-1, :, :] = data_numpy_v1[:, 1:, :, :] - data_numpy_v1[:, :-1, :, :]
            motion_v2 = np.zeros_like(data_numpy_v2)
            motion_v2[:, :-1, :, :] = data_numpy_v2[:, 1:, :, :] - data_numpy_v2[:, :-1, :, :]

            data_numpy_v1 = motion_v1
            data_numpy_v2 = motion_v2

        elif self.input_representation == "bone":
            # bone
            bone_v1 = np.zeros_like(data_numpy_v1)
            for v1, v2 in self.Bone:
                bone_v1[:, :, v1 - 1, :] = data_numpy_v1[:, :, v1 - 1, :] - data_numpy_v1[:, :, v2 - 1, :]
            bone_v2 = np.zeros_like(data_numpy_v2)
            for v1, v2 in self.Bone:
                bone_v2[:, :, v1 - 1, :] = data_numpy_v2[:, :, v1 - 1, :] - data_numpy_v2[:, :, v2 - 1, :]

            data_numpy_v1 = bone_v1
            data_numpy_v2 = bone_v2

        if random.random() < 0.5:
            data_numpy_v1 = augmentations.Rotate(data_numpy_v1)
        if random.random() < 0.5:
            data_numpy_v1 = augmentations.Flip(data_numpy_v1)
        if random.random() < 0.5:
            data_numpy_v1 = augmentations.Shear(data_numpy_v1)
        if random.random() < 0.5:
            data_numpy_v1 = augmentations.spatial_masking(data_numpy_v1)
        if random.random() < 0.5:
            data_numpy_v1 = augmentations.temporal_masking(data_numpy_v1)

        if random.random() < 0.5:
            data_numpy_v2 = augmentations.Rotate(data_numpy_v2)
        if random.random() < 0.5:
            data_numpy_v2 = augmentations.Flip(data_numpy_v2)
        if random.random() < 0.5:
            data_numpy_v2 = augmentations.Shear(data_numpy_v2)
        if random.random() < 0.5:
            data_numpy_v2 = augmentations.spatial_masking(data_numpy_v2)
        if random.random() < 0.5:
            data_numpy_v2 = augmentations.temporal_masking(data_numpy_v2)

        return data_numpy_v1, data_numpy_v2