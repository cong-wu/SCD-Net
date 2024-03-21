import torch.nn.functional as F
import torch
import random
import numpy as np
from einops import rearrange

import math
from math import sin, cos

ntu_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
)

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = edge2mat(inward, num_node)
    Out = edge2mat(outward, num_node)
    A = In + Out + I
    return A


def spatial_masking(input_data):
    num_node = 25
    self_link = [(i, i) for i in range(num_node)]
    inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                        (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                        (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
    inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
    outward = [(j, i) for (i, j) in inward]
    A = get_spatial_graph(num_node, self_link, inward, outward)
    A = np.matmul(A, A)
    shuffle_index = np.random.randint(low=0, high=25, size=5)
    flag = A[shuffle_index].sum(0)
    joint_indicies = flag.argsort()[-8:]
    out = input_data.copy()
    out[:, :, joint_indicies, :] = 0
    return out


def temporal_masking(input_data):

    input_data = rearrange(input_data, 'c (t d) v m -> c t d v m', d=4)
    temporal_indicies = np.random.choice(16, 6, replace=False)
    out = input_data.copy()
    out[:, temporal_indicies] = 0
    out = rearrange(out, 'c t d v m -> c (t d) v m')
    return out


def Shear(input_data):
    Shear = np.array([
        [1, random.uniform(-1, 1), random.uniform(-1, 1)],
        [random.uniform(-1, 1), 1, random.uniform(-1, 1)],
        [random.uniform(-1, 1), random.uniform(-1, 1), 1]
    ])
    # c t v m
    output = np.dot(input_data.transpose([1, 2, 3, 0]), Shear.transpose())
    output = output.transpose(3, 0, 1, 2)
    return output


def Flip(input_data):
    order = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 20, 23, 24, 21, 22]
    output = input_data[:, :, order, :]
    return output


def Rotate(data):
    def rotate(seq, axis, angle):
        # x
        if axis == 0:
            R = np.array([[1, 0, 0],
                              [0, cos(angle), sin(angle)],
                              [0, -sin(angle), cos(angle)]])
        # y
        if axis == 1:
            R = np.array([[cos(angle), 0, -sin(angle)],
                              [0, 1, 0],
                              [sin(angle), 0, cos(angle)]])

        # z
        if axis == 2:
            R = np.array([[cos(angle), sin(angle), 0],
                              [-sin(angle), cos(angle), 0],
                              [0, 0, 1]])
        R = R.T
        output = np.dot(seq.transpose([1, 2, 3, 0]), R)
        output = output.transpose(3, 0, 1, 2)
        return output

    # c t v m
    new_seq = data.copy()
    total_axis = [0, 1, 2]
    main_axis = random.randint(0, 2)
    for axis in total_axis:
        if axis == main_axis:
            rotate_angle = random.uniform(0, 30)
            rotate_angle = math.radians(rotate_angle)
            new_seq = rotate(new_seq, axis, rotate_angle)
        else:
            rotate_angle = random.uniform(0, 1)
            rotate_angle = math.radians(rotate_angle)
            new_seq = rotate(new_seq, axis, rotate_angle)

    return new_seq


def temporal_cropresize(input_data,num_of_frames,l_ratio,output_size):


    C, T, V, M =input_data.shape

    # Temporal crop
    min_crop_length = 64

    scale = np.random.rand(1)*(l_ratio[1]-l_ratio[0])+l_ratio[0]
    temporal_crop_length = np.minimum(np.maximum(int(np.floor(num_of_frames*scale)),min_crop_length),num_of_frames)

    start = np.random.randint(0,num_of_frames-temporal_crop_length+1)
    temporal_context = input_data[:,start:start+temporal_crop_length, :, :]

    # interpolate
    temporal_context = torch.tensor(temporal_context,dtype=torch.float)
    temporal_context=temporal_context.permute(0, 2, 3, 1).contiguous().view(C * V * M,temporal_crop_length)
    temporal_context=temporal_context[None, :, :, None]
    temporal_context= F.interpolate(temporal_context, size=(output_size, 1), mode='bilinear',align_corners=False)
    temporal_context = temporal_context.squeeze(dim=3).squeeze(dim=0) 
    temporal_context=temporal_context.contiguous().view(C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()

    return temporal_context

def crop_subsequence(input_data,num_of_frames,l_ratio,output_size):


    C, T, V, M =input_data.shape

    if l_ratio[0] == 0.5:
    # if training , sample a random crop

         min_crop_length = 64
         scale = np.random.rand(1)*(l_ratio[1]-l_ratio[0])+l_ratio[0]
         temporal_crop_length = np.minimum(np.maximum(int(np.floor(num_of_frames*scale)),min_crop_length),num_of_frames)

         start = np.random.randint(0,num_of_frames-temporal_crop_length+1)
         temporal_crop = input_data[:,start:start+temporal_crop_length, :, :]

         temporal_crop= torch.tensor(temporal_crop,dtype=torch.float)
         temporal_crop=temporal_crop.permute(0, 2, 3, 1).contiguous().view(C * V * M,temporal_crop_length)
         temporal_crop=temporal_crop[None, :, :, None]
         temporal_crop= F.interpolate(temporal_crop, size=(output_size, 1), mode='bilinear',align_corners=False)
         temporal_crop=temporal_crop.squeeze(dim=3).squeeze(dim=0) 
         temporal_crop=temporal_crop.contiguous().view(C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()

         return temporal_crop

    else:
    # if testing , sample a center crop

        start = int((1-l_ratio[0]) * num_of_frames/2)
        data =input_data[:,start:num_of_frames-start, :, :]
        temporal_crop_length = data.shape[1]

        temporal_crop= torch.tensor(data,dtype=torch.float)
        temporal_crop=temporal_crop.permute(0, 2, 3, 1).contiguous().view(C * V * M,temporal_crop_length)
        temporal_crop=temporal_crop[None, :, :, None]
        temporal_crop= F.interpolate(temporal_crop, size=(output_size, 1), mode='bilinear',align_corners=False)
        temporal_crop=temporal_crop.squeeze(dim=3).squeeze(dim=0) 
        temporal_crop=temporal_crop.contiguous().view(C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()

        return temporal_crop
