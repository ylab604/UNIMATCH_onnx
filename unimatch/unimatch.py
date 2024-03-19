import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .transformer import FeatureTransformer
# from .matching import (global_correlation_softmax, local_correlation_softmax, local_correlation_with_flow,
#                        global_correlation_softmax_stereo, local_correlation_softmax_stereo,
#                        correlation_softmax_depth)
from .matching import (global_correlation_softmax_stereo)
from .attention import SelfAttnPropagation
from .geometry import flow_warp, compute_flow_with_depth_pose
from .reg_refine import BasicUpdateBlock
from .utils import normalize_img, feature_add_position, upsample_flow_with_mask


class UniMatch(nn.Module):
    def __init__(self,
                 num_scales=1,
                 feature_channels=128,
                 upsample_factor=8,
                 num_head=1,
                 ffn_dim_expansion=4,
                 num_transformer_layers=6,
                 reg_refine=False,  # optional local regression refinement
                 task='stereo',
                 ):
        super(UniMatch, self).__init__()

        #######################
        self.att_split = 2
        self.corr_radius = -1
        self.prop_radius = -1
        self.scale_idx = 0
        #######################

        self.feature_channels = 128
        self.num_scales = 1
        self.upsample_factor = 8
        self.reg_refine = False

        # CNN
        # self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)
        self.backbone = CNNEncoder(output_dim=128, num_output_scales=1)


        # Transformer
        self.transformer = FeatureTransformer(num_layers=6,
                                              d_model=128,
                                              nhead=1,
                                              ffn_dim_expansion=4,att_num_splits =2
                                              )
        # self.transformer = FeatureTransformer(num_layers=6,
        #                                       d_model=128,
        #                                       nhead=1,
        #                                       ffn_dim_expansion=4,
        #                                       )

        # propagation with self-attn
        self.feature_flow_attn = SelfAttnPropagation(in_channels=128)


        # convex upsampling simiar to RAFT
        # concat feature0 and low res flow as input
        self.upsampler = nn.Sequential(nn.Conv2d(2 + 128, 256, 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(256, 8 ** 2 * 9, 1, 1, 0))
            # thus far, all the learnable parameters are task-agnostic



    # def extract_feature(self, img0, img1):
    #     concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
    #     features = self.backbone(concat)  # list of [2B, C, H, W], resolution from high to low
    #
    #     # reverse: resolution from low to high
    #     features = features[::-1]
    #
    #     feature0, feature1 = [], []
    #     print(len(features))
    #     print("check the len features@@@@@@@@@@@@@@@@")
    #
    #     # for i in range(len(features)):
    #     #     feature = features[i]
    #     #     # print(feature.shape) torch.Size([2, 128, 60, 80])
    #     #     chunks = torch.chunk(feature, 2, 0)  # tuple
    #     #     # print(len(chunks)) # 2
    #     #     feature0.append(chunks[0])
    #     #     feature1.append(chunks[1])
    #     feature = features[0]
    #     # print(feature.shape) torch.Size([2, 128, 60, 80])
    #     feature_tmp = feature
    #     chunks = torch.chunk(feature, 2, 0)  # tuple
    #     feature0.append(chunks[0])
    #     feature1.append(chunks[1])
    #
    #     #print(feature0[0].shape) torch.Size([1, 128, 60, 80])
    #     print("check the type of the feature0@@@@@@@@@@@")
    #
    #     return feature0, feature1

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.backbone(concat)  # list of [2B, C, H, W], resolution from high to low
        # reverse: resolution from low to high
        features = features[::-1]
        feature0, feature1 = [], []
        feature = features[0]
        feature0.append(feature[:1])
        feature1.append(feature[1:])
        #print(feature0[0].shape) torch.Size([1, 128, 60, 80])

        # print("check the type of the feature0@@@@@@@@@@@")
        #
        # print(feature0[0].shape) ### old : torch.Size([1, 128, 60, 80]) new : torch.Size([1, 128, 60, 80])
        #
        # print("check the type of the feature0@@@@@@@@@@@")

        return feature0, feature1



    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8,
                      is_depth=False):
        # if bilinear:
        #     multiplier = 1 if is_depth else upsample_factor
        #     up_flow = F.interpolate(flow, scale_factor=upsample_factor,
        #                             mode='bilinear', align_corners=True) * multiplier
        # else:
        concat = torch.cat((flow, feature), dim=1)
        mask = self.upsampler(concat)
        up_flow = upsample_flow_with_mask(flow, mask, upsample_factor=self.upsample_factor,
                                              is_depth=is_depth)

        return up_flow

    def forward(self, img0, img1,

                **kwargs,
                ):
        results_dict = {}
        flow_preds = []


        # list of features, resolution low to high
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features

        flow = None

        # if task != 'depth':
        #     assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales
        # else:
        #     assert len(attn_splits_list) == len(prop_radius_list) == self.num_scales == 1

        # for scale_idx in range(self.num_scales):
        #     feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]
        #     feature0_ori, feature1_ori = feature0, feature1
        #     upsample_factor = 8 * (2 ** (1 - 1 - scale_idx))
        #     attn_splits = self.att_split
        #     corr_radius = self.corr_radius
        #     prop_radius = self.prop_radius
        #
        #     print(feature0.device)
        #     print("@@@@@@@@@@@@@@@@@@@check the device")
        #
        #
        #     # add position to features
        #     feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)
        #
        #     # Transformer
        #     # feature0, feature1 = self.transformer(feature0, feature1,
        #     #                                       attn_type=attn_type,
        #     #                                       attn_num_splits=attn_splits,
        #     #                                       )
        #     # print(feature0.shape)
        #     # print(feature1.shape)
        #     # print("@@@@@@@@@@@@@@@@@check the feture shape")
        #     feature0, feature1 = self.transformer(feature0, feature1,
        #                                           attn_type="self_swin2d_cross_1d",
        #
        #                                           )
        #     # correlation and softmax
        #     flow_pred = global_correlation_softmax_stereo(feature0, feature1)[0]
        #
        #     # flow or residual flow
        #     flow = flow + flow_pred if flow is not None else flow_pred
        #
        #     flow = flow.clamp(min=0)  # positive disparity
        #
        #     flow = self.feature_flow_attn(feature0, flow.detach(),
        #                                   local_window_attn=prop_radius > 0,
        #                                   local_window_radius=prop_radius,
        #                                   )
        #
        #     # upsample to the original image resolution
        #     flow_pad = torch.cat((-flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
        #     flow_up_pad = self.upsample_flow(flow_pad, feature0)
        #     flow_up = -flow_up_pad[:, :1]  # [B, 1, H, W]
        #
        #     flow_preds.append(flow_up)

        # for scale_idx in range(self.num_scales):
        scale_idx = self.scale_idx
        feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]
        # feature0_ori, feature1_ori = feature0, feature1
        upsample_factor = 8 * (2 ** (1 - 1 - scale_idx))
        attn_splits = self.att_split
        corr_radius = self.corr_radius
        prop_radius = self.prop_radius

        print(feature0.device)
        print("@@@@@@@@@@@@@@@@@@@check the device")


        # add position to features
        feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)

        # Transformer
        # feature0, feature1 = self.transformer(feature0, feature1,
        #                                       attn_type=attn_type,
        #                                       attn_num_splits=attn_splits,
        #                                       )
        # print(feature0.shape)
        # print(feature1.shape)
        # print("@@@@@@@@@@@@@@@@@check the feture shape")
        feature0, feature1 = self.transformer(feature0, feature1,
                                              attn_type="self_swin2d_cross_1d",

                                              )
        # correlation and softmax
        flow_pred = global_correlation_softmax_stereo(feature0, feature1)[0]

        # flow or residual flow
        flow = flow + flow_pred if flow is not None else flow_pred

        flow = flow.clamp(min=0)  # positive disparity

        flow = self.feature_flow_attn(feature0, flow.detach(),
                                      local_window_attn=prop_radius > 0,
                                      local_window_radius=prop_radius,
                                      )

        # upsample to the original image resolution
        flow_pad = torch.cat((-flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
        flow_up_pad = self.upsample_flow(flow_pad, feature0)
        flow_up = -flow_up_pad[:, :1]  # [B, 1, H, W]

        flow_preds.append(flow_up)






        for i in range(len(flow_preds)):

            flow_preds[i] = flow_preds[i].squeeze(1)  # [B, H, W]


        results_dict.update({'flow_preds': flow_preds})
        results_dict = results_dict["flow_preds"][-1]

        return results_dict