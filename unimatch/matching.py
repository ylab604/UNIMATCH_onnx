import torch
import torch.nn.functional as F

from .geometry import coords_grid, generate_window_grid, normalize_coords
import numpy as np


def global_correlation_softmax_stereo(feature0, feature1,
                                      ):
    # global correlation on horizontal direction
    b, c, h, w = feature0.shape
    # print("check the w value @@@@@@@@@@@@@@@@@@@")
    # print(w) # 80
    # print("check the w value @@@@@@@@@@@@@@@@@@@")

    # x_grid = torch.linspace(0, w - 1, w, device=feature0.device)  # [W]
    # print(feature0.device)
    # print("check the device")
    x_grid = torch.arange(0,80,1,dtype = torch.float).to(device=feature0.device)

    # print("check the x_grid value @@@@@@@@@@@@@@@@")
    # print(x_grid)
    # # tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.,
    # #         14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27.,
    # #         28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41.,
    # #         42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55.,
    # #         56., 57., 58., 59., 60., 61., 62., 63., 64., 65., 66., 67., 68., 69.,
    # #         70., 71., 72., 73., 74., 75., 76., 77., 78., 79.])
    # print("check the x_grid value @@@@@@@@@@@@@@@@")

    feature0 = feature0.permute(0, 2, 3, 1)  # [B, H, W, C]
    feature1 = feature1.permute(0, 2, 1, 3)  # [B, H, C, W]

    correlation = torch.matmul(feature0, feature1) / (128 ** 0.5)  # [B, H, W, W]

    # mask subsequent positions to make disparity positive
    # mask = torch.triu(torch.ones((w, w)), diagonal=1).type_as(feature0)  # [W, W]
    mask = np.triu(torch.ones((w, w)), k=1)  # [W, W]
    mask = torch.from_numpy(mask)
    # print("check the mask@@@@@@@@@@@@")
    # print(mask.shape) torch.Size([80, 80])
    # print("check the mask@@@@@@@@@@@@")


    # valid_mask = (mask == 0).unsqueeze(0).unsqueeze(0).repeat(b, h, 1, 1)  # [B, H, W, W]
    #
    # correlation[~valid_mask] = -1e9
    valid_mask = mask == 0
    valid_mask = valid_mask.to(correlation.device)
    # correlation에서 valid_mask가 False인 위치를 -1e9로 설정
    # torch.where 사용
    # correlation = torch.where(valid_mask.unsqueeze(0).unsqueeze(0), correlation,
    #                                   torch.tensor(-1e9, device=correlation.device))
    correlation = torch.where(valid_mask.unsqueeze(0).unsqueeze(0), correlation,
                              torch.tensor(-1e9, device=correlation.device))

    prob = F.softmax(correlation, dim=-1)  # [B, H, W, W]

    correspondence = (x_grid.view(1, 1, 1, w) * prob).sum(-1)  # [B, H, W]

    # NOTE: unlike flow, disparity is typically positive
    # disparity = x_grid.view(1, 1, w).repeat(b, h, 1) - correspondence  # [B, H, W]
    disparity = x_grid -correspondence
    return disparity.unsqueeze(1), prob  # feature resolution

# def global_correlation_softmax_stereo(feature0, feature1):
#     b, c, h, w = feature0.shape
#
#     x_grid = torch.linspace(0, w - 1, w, device=feature0.device)  # [W]
#
#     feature0 = feature0.permute(0, 2, 3, 1)  # [B, H, W, C]
#     feature1 = feature1.permute(0, 2, 1, 3)  # [B, H, C, W]
#
#     correlation = torch.matmul(feature0, feature1) / (c ** 0.5)  # [B, H, W, W]
#
#     # Efficient masking to ensure positive disparity
#     # Generate a dynamic mask based on the indices
#     mask = torch.arange(w, device=feature0.device).view(1, 1, 1, w) >= torch.arange(w, device=feature0.device).view(1, 1, w, 1)
#     correlation = torch.where(mask, correlation, torch.tensor(-1e9, device=feature0.device))
#
#     prob = F.softmax(correlation, dim=-1)  # [B, H, W, W]
#
#     correspondence = (x_grid.view(1, 1, 1, w) * prob).sum(-1)  # [B, H, W]
#
#     disparity = x_grid.view(1, 1, w).repeat(b, h, 1) - correspondence  # [B, H, W]
#
#     return disparity.unsqueeze(1), prob  # Feature resolution




