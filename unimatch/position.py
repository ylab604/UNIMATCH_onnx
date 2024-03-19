# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py

import torch
import torch.nn as nn
import math



import torch
import torch.nn as nn
import math

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.size = (4, 128, 30, 40)
        self.position_embedding = self._create_fixed_position_embeddings()

    def _create_fixed_position_embeddings(self):

        b, c, h, w = self.size
        ##############3
        # mask = torch.ones((b, h, w), device='cpu')  # [B, H, W]
        mask = torch.ones((b, h, w), device='cuda:0')  # [B, H, W]
        # print("@@@@@@@@@2")
        # print(mask.device)
        # print("@@@@@@@@@2")
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device='cpu')
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device='cuda:0')

        # print("@@@@@@@@@2")
        # print(dim_t.device)
        # print("@@@@@@@@@2")
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # print(pos_x.shape) # torch.Size([4, 30, 40, 64])
        # print(pos_y.shape) # torch.Size([4, 30, 40, 64])
        # print("@@@@@@@@@@@@")
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # print(pos.shape) # torch.Size([4, 128, 30, 40])
        # print("@@@@@@@@@@@@3")


        return pos


    def forward(self, x):
        # x = tensor_list.tensors  # [B, C, H, W]
        # mask = tensor_list.mask  # [B, H, W], input with padding, valid as 0
        # b, c, h, w = x.size()
        # print(x.size()) torch.Size([4, 128, 30, 40])
        # print("@@@@@@@@@@@@@@@@@")
        position_embedding = self.position_embedding

        # print(position_embedding.shape)
        # #torch.Size([16, 128, 30, 40]) 1차시도
        # print("@@@@@@@@@@@@@2check the new one")

        return position_embedding