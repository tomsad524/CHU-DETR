# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss)
from .deformable_transformer import build_deformable_transformer
from .utils import sigmoid_focal_loss, MLP

from ..registry import MODULE_BUILD_FUNCS
from .dn_components import prepare_for_cdn, dn_post_process
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # import ipdb; ipdb.set_trace()
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)



# modified from torchvision to also return the union
def box_iou_pairwise(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou_pairwise(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    Input:
        - boxes1, boxes2: N,4
    Output:
        - giou: N, 4
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    assert boxes1.shape == boxes2.shape
    iou, union = box_iou_pairwise(boxes1, boxes2) # N, 4

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,2]
    area = wh[:, 0] * wh[:, 1]

    return iou - (area - union) / area

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

if __name__ == '__main__':
    x = torch.rand(5, 4)
    y = torch.rand(3, 4)
    iou, union = box_iou(x, y)


def generate_cosine_positional_encoding(d_model, height, width):
    pe = torch.zeros(d_model, height, width)
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
    pos_y, pos_x = torch.meshgrid(torch.arange(0., height), torch.arange(0., width))
    pe[0::2, :, :] = torch.sin(pos_y * div_term[:, None, None])
    pe[1::2, :, :] = torch.cos(pos_x * div_term[:, None, None])
    return pe


def shuffle_merge(output1, output2):
    if output1.shape != output2.shape:
        raise ValueError("same shape for inputs please")

    B, C, H, W = output1.shape
    quarter = C // 4

    new_output1 = torch.cat((
        output1[:, :quarter, :, :],  # Premier quart de output1
        output2[:, quarter:2 * quarter, :, :],
        output1[:, 2 * quarter:3 * quarter, :, :],
        output2[:, 3 * quarter:, :, :]  # Dernier quart de output2
    ), dim=1)

    new_output2 = torch.cat((
        output2[:, :quarter, :, :],  # Premier quart de output2
        output1[:, quarter:2 * quarter, :, :],
        output2[:, 2 * quarter:3 * quarter, :, :],
        output1[:, 3 * quarter:, :, :]  # Dernier quart de output1
    ), dim=1)

    return new_output1, new_output2


def shuffle_merge_space(output1, output2, patch_size):
    if output1.shape != output2.shape:
        raise ValueError("respecte toi ! Respecte toi frere !")

    B, C, H, W = output1.shape
    PH, PW = patch_size

    patches1 = output1.unfold(2, PH, PH).unfold(3, PW, PW)
    patches2 = output2.unfold(2, PH, PH).unfold(3, PW, PW)

    for i in range(0, patches1.size(2)):
        for j in range(0, patches1.size(3)):
            if (i + j) % 2 == 0:  # Alternance des patches
                patches1[:, :, i, j, :, :], patches2[:, :, i, j, :, :] = patches2[:, :, i, j, :, :], patches1[:, :, i,
                                                                                                     j, :, :]

    new_output1 = patches1.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)
    new_output2 = patches2.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)

    return new_output1, new_output2

# Utility modules for the model (Downsampler, CrossSpectrumAttentionWithPosition)
import torch.nn as nn

class Downsampler(nn.Module):
    """
    A learnable downsampling module (D_down in the paper, Section 3.1.1).
    Uses a 3x3 convolution with stride 2 to halve the spatial resolution of the
    input feature map while learning to encode cross-hierarchical information.
    """
    def __init__(self, in_channels, out_channels):
        super(Downsampler, self).__init__()
        # 3x3 conv captures local spatial relationships during downsampling, more effective than simple pooling
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(32, out_channels) # GroupNorm for training stability
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if x is None:
            return None
        return self.relu(self.norm(self.conv(x)))


class CrossSpectrumAttentionWithPosition(nn.Module):
    def __init__(self, in_channels, num_stages):
        super(CrossSpectrumAttentionWithPosition, self).__init__()
        self.num_stages = num_stages
        self.attention_stages_k1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Softmax(dim=-1)
            )
            for _ in range(num_stages)
        ])

        self.attention_stages_k3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=5),
                nn.Conv2d(in_channels, in_channels, kernel_size=5),
                nn.Conv2d(in_channels, in_channels, kernel_size=5),
                nn.Softmax(dim=-1)
            )
            for _ in range(num_stages)
        ])
        """
        self.attention_stages_k3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3),
                nn.Conv2d(in_channels, in_channels, kernel_size=3),
                nn.Conv2d(in_channels, in_channels, kernel_size=3),
                nn.Softmax(dim=-1)
            )
            for _ in range(num_stages)
        ])
        """
        self.attention_stages_k7 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=7),
                nn.Conv2d(in_channels, in_channels, kernel_size=7),
                nn.Conv2d(in_channels, in_channels, kernel_size=7),
                nn.Softmax(dim=-1)
            )
            for _ in range(num_stages)
        ])

        self.compression_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),
            )
            for _ in range(num_stages)
        ])

        self.proj = nn.Conv2d(in_channels, 64, kernel_size=1)

        # (CHAF) Learnable 1x1 projection layer (P_proj in Eq. 1) that processes and injects prior features from the previous hierarchy level
        self.cross_level_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x1, x2, prev_level_feature):

        # (CHAF) If prior information (relay baton) from the previous hierarchy exists, fuse it into the current visible branch (x1)  — Eq. (1)
        if prev_level_feature is not None:
            # Inject through a learnable projection layer, added element-wise to x1: F_vis_hat = F_vis + P_proj(D_down(F_fused^{i-1}))  (Eq. 1)
            x1 = x1 + self.cross_level_proj(prev_level_feature)

        # hack : future implementation with learned pos_encoding ?
        pos_encoding1 = generate_cosine_positional_encoding(x1.size(1), x1.size(2), x1.size(3))
        pos_encoding2 = generate_cosine_positional_encoding(x2.size(1), x2.size(2), x2.size(3))

        # additive (hack ?) fused pos_encoding
        combined_pos_encoding = pos_encoding1 + pos_encoding2
        combined_pos_encoding = combined_pos_encoding.to(x1.device)

        for i in range(self.num_stages):  # loop for hierarchical self/cross attention steps
            # hierarchical attention + learned attention fusion (convolution based procedure)
            query_conv1, key_conv1, value_conv1, softmax1 = self.attention_stages_k1[i]
            query1 = query_conv1(x1) + combined_pos_encoding
            key1 = key_conv1(x2) + combined_pos_encoding
            value1 = value_conv1(x2)
            attn_weights1 = softmax1(torch.matmul(query1, key1.transpose(-1, -2)))
            fused_features_k1 = torch.matmul(attn_weights1, value1)

            query_conv3, key_conv3, value_conv3, softmax3 = self.attention_stages_k3[i]
            query3 = query_conv3(x1)
            key3 = key_conv3(x2)
            value3 = value_conv3(x2)
            attn_weights3 = softmax3(torch.matmul(query3, key3.transpose(-1, -2)))
            fused_features_k3 = torch.matmul(attn_weights3, value3)

            query_conv7, key_conv7, value_conv7, softmax7 = self.attention_stages_k7[i]
            query7 = query_conv7(x1)
            key7 = key_conv7(x2)
            value7 = value_conv7(x2)
            attn_weights7 = softmax7(torch.matmul(query7, key7.transpose(-1, -2)))
            fused_features_k7 = torch.matmul(attn_weights7, value7)

            #  target features shape
            target_height = fused_features_k1.size(2)
            target_width = fused_features_k1.size(3)

            # reshaping the different attention maps before concatenation : interpolation (not-learned)
            k3_resized = F.interpolate(fused_features_k3, size=(target_height, target_width), mode='bilinear',
                                       align_corners=False)
            k7_resized = F.interpolate(fused_features_k7, size=(target_height, target_width), mode='bilinear',
                                       align_corners=False)

            fused_features = torch.cat((fused_features_k1, k3_resized, k7_resized), 1)  # stacking
            fused_features = self.compression_convs[i](
                fused_features)  # learned operation: compress to the standard features shape

            # Update x1 and x2 for the next stage
            x1, x2 = fused_features, fused_features

        # projects to the reduced dimensionnality (legacy)
        # x1_conv = self.vis_proj(x1_value)
        # x2_conv = self.ir_proj(x2_value)

        # features concatenation (legacy)
        # features = torch.cat((x1_conv, x2_conv, fused_proj), 1)

        fused_features = self.proj(fused_features)  # projection : learned reshaping for dimensionality compression
        return fused_features

class DINO(nn.Module):
    """ This is the Cross-Attention Detector module that performs object detection """

    def __init__(self, backbone, backbone_ir, transformer, num_classes, num_queries,
                 aux_loss=False, iter_update=False,
                 query_dim=2,
                 random_refpoints_xy=False,
                 fix_refpoints_hw=-1,
                 num_feature_levels=1,
                 nheads=8,
                 # two stage
                 two_stage_type='no',  # ['no', 'standard']
                 two_stage_add_query_num=0,
                 dec_pred_class_embed_share=True,
                 dec_pred_bbox_embed_share=True,
                 two_stage_class_embed_share=True,
                 two_stage_bbox_embed_share=True,
                 decoder_sa_type='sa',
                 num_patterns=0,
                 dn_number=100,
                 dn_box_noise_scale=0.4,
                 dn_label_noise_ratio=0.5,
                 dn_labelbook_size=100,
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.

            fix_refpoints_hw: -1(default): learn w and h for each box seperately
                                >0 : given fixed number
                                -2 : learn a shared w and h
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)
        self.backbone_visible = backbone
        self.backbone_ir = backbone_ir
        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(self.backbone_visible.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        if num_feature_levels > 1:
            num_backbone_outs = len(self.backbone_visible.num_channels)
            input_proj_ir_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.backbone_ir.num_channels[_]
                input_proj_ir_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_ir_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj_ir = nn.ModuleList(input_proj_ir_list)
        else:
            self.input_proj_ir = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone_visible.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        conv_corr_list = []
        redim_corr_list = []
        downsamplers_list = []  # (CHAF) Initialize downsampler module list for top-down context path
        num_stages = 2  # Replace with the number of stages you want : experiment : shallow cross-attention => no deepen attention extraction but only one H cross attention performed

        # (CHAF) Create one downsampler for each cross-hierarchy connection
        # For N feature levels, we need N-1 downsamplers (D_down modules)
        for i in range(self.num_feature_levels - 1):
            # Each downsampler's input/output channels match the model's hidden dimension
            downsamplers_list.append(Downsampler(hidden_dim, hidden_dim))

        for i in range(self.num_feature_levels):
            redim_corr_list.append(nn.Sequential(
                nn.Conv2d(2 * hidden_dim + 64, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim)))

        for i in range(self.num_feature_levels):
            conv_corr_list.append(CrossSpectrumAttentionWithPosition(in_channels=256, num_stages=num_stages))

        self.conv_corr = nn.ModuleList(conv_corr_list)  # redimensionnate correlation matrix
        self.redim_corr = nn.ModuleList(redim_corr_list)
        # (CHAF) Register downsampler list as a sub-module of the model
        self.downsamplers = nn.ModuleList(downsamplers_list)

        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = nn.Linear(hidden_dim, num_classes)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(two_stage_add_query_num)

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']
        if decoder_sa_type == 'ca_label':
            self.label_embedding = nn.Embedding(num_classes, hidden_dim)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)
        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if self.fix_refpoints_hw > 0:
            print("fix_refpoints_hw: {}".format(self.fix_refpoints_hw))
            assert self.random_refpoints_xy
            self.refpoint_embed.weight.data[:, 2:] = self.fix_refpoints_hw
            self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(self.refpoint_embed.weight.data[:, 2:])
            self.refpoint_embed.weight.data[:, 2:].requires_grad = False
        elif int(self.fix_refpoints_hw) == -1:
            pass
        elif int(self.fix_refpoints_hw) == -2:
            print('learn a shared h and w')
            assert self.random_refpoints_xy
            self.refpoint_embed = nn.Embedding(use_num_queries, 2)
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
            self.hw_embed = nn.Embedding(1, 1)
        else:
            raise NotImplementedError('Unknown fix_refpoints_hw {}'.format(self.fix_refpoints_hw))

    def forward(self, samples: NestedTensor, samples_thermal: NestedTensor, targets: List = None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # Add thermal canal
        if not isinstance(samples_thermal, NestedTensor):
            samples_thermal = nested_tensor_from_tensor_list(samples_thermal)
        features, poss = self.backbone_visible(samples)
        features_thermal, poss_thermal = self.backbone_ir(samples_thermal)
        srcs = []
        srcs_thermal = []
        masks = []
        masks_thermal = []

        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        for l, feat in enumerate(features_thermal):
            src, mask = feat.decompose()
            srcs_thermal.append(self.input_proj_ir[l](src))
            masks_thermal.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                    src_ir = self.input_proj_ir[l](features_thermal[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                    src_ir = self.input_proj[l](srcs_thermal[-1])
                m = samples.mask
                m_thermal = samples_thermal.mask

                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                mask_thermal = F.interpolate(m_thermal[None].float(), size=src_ir.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone_visible[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                srcs_thermal.append(src_ir)
                masks.append(mask)
                masks_thermal.append(mask_thermal)
                poss.append(pos_l)

        # --- Core Fusion Loop (CHAF Top-Down Context Path) ---
        # (CHAF) Initialize a variable to store the fused result from the previous hierarchy — the “relay baton”
        prev_level_fused = None

        # (CHAF) Sequential fusion logic (top-down chain), replacing the original independent per-level fusion
        for i in range(self.num_feature_levels):
            # 1. (CHAF) Prepare input from the previous hierarchy level
            cross_level_input = None
            if prev_level_fused is not None:
                #     Use the corresponding downsampler self.downsamplers[i-1]  — D_down in Eq. (1)
                cross_level_input = self.downsamplers[i - 1](prev_level_fused)

            # 2. (CHAF) Call CrossSpectrumAttentionWithPosition with three inputs: visible, infrared, and previous-level fused result
            src_corr = self.conv_corr[i](srcs[i], srcs_thermal[i], cross_level_input)

            # 3. Subsequent concatenation and projection logic remains unchanged
            corr = self.redim_corr[i](torch.cat((srcs[i], srcs_thermal[i], src_corr), 1))

            # 4. (CHAF) Update current fused features, save the “relay baton” for the next round
            srcs[i] = corr
            prev_level_fused = corr

            # 5. Mask fusion logic remains unchanged
            masks[i] = masks[i] | masks_thermal[i]

        if self.dn_number > 0 or targets is not None:
            input_query_label, input_query_bbox, attn_mask, dn_meta = \
                prepare_for_cdn(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
                                training=self.training, num_queries=self.num_queries, num_classes=self.num_classes,
                                hidden_dim=self.hidden_dim, label_enc=self.label_enc)
        else:
            assert targets is None
            input_query_bbox = input_query_label = attn_mask = dn_meta = None

        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(srcs, masks, input_query_bbox, poss,
                                                                             input_query_label, attn_mask)
        # In case num object=0
        hs[0] += self.label_enc.weight[0, 0] * 0.0

        # deformable-detr-like anchor update
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack([layer_cls_embed(layer_hs) for
                                     layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list = \
                dn_post_process(outputs_class, outputs_coord_list,
                                dn_meta, self.aux_loss, self._set_aux_loss)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)

        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

            # prepare enc outputs
            if hs_enc.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc) in enumerate(
                        zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc[:-1], ref_enc[:-1])):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                    layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                out['enc_outputs'] = [
                    {'pred_logits': a, 'pred_boxes': b} for a, b in zip(enc_outputs_class, enc_outputs_coord)
                ]

        out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.ta_beta = 2.0

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        Classification loss (Binary focal loss) with Position-Supervised Loss (PSL).
        This is the simplified and direct implementation of the Stable-DINO loss.
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        # Get classification prediction results (logits) from model output
        src_logits = outputs["pred_logits"]  # shape: [batch_size, num_queries, num_classes]

        # --- Prepare base data ---
        # Get indices of positive samples in batch and query dimensions from matching results
        idx = self._get_src_permutation_idx(indices)
        # Get ground-truth class labels for all positive samples
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # Create a one-hot encoded target tensor to mark positive/negative sample positions
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        # --- Compute Position-Supervised Loss (PSL) (Eq. 9) ---
        # Prepare Focal Loss hyperparameters and model predicted probabilities
        focal_alpha = self.focal_alpha  # alpha and focal_alpha naming kept consistent
        focal_gamma = 2.0  # gamma is typically fixed to 2.0
        out_prob = src_logits.sigmoid()
        bs, nq = src_logits.shape[:2]

        # 1. Compute IoU between matched predicted boxes and ground-truth boxes
        src_boxes = outputs["pred_boxes"][idx]
        tgt_bbox = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        iou = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_bbox))[0].diag()

        # --- Ablation Config Support: PSL soft label type (Table 9, Section 4.4.4) ---
        # Check if ablation_config is set and contains a psl_soft_label override.
        # If no ablation_config, defaults to quadratic PSL (the proposed method).
        if hasattr(self, 'ablation_config') and self.ablation_config is not None:
            psl_type = self.ablation_config.psl_soft_label  # 'hard', 'mixed', 'linear', 'quadratic'
        else:
            psl_type = 'quadratic'  # default: f(s_i) = s_i^2

        if psl_type == 'hard':
            # Standard Focal Loss — constant target = 1 for positive samples
            # This is the baseline showing the "multi-optimization ambiguity" problem
            _t = target_classes_onehot.float()
        elif psl_type == 'mixed':
            # Mixed Score: 0.5 * 1 + 0.5 * IoU (causes catastrophic degradation, mAP 41.2%)
            _u = torch.zeros_like(out_prob)
            _u[idx[0], idx[1], target_classes_o] = 0.5 + 0.5 * iou
            _t = _u.pow(self.ta_beta)
        elif psl_type == 'linear':
            # Linear PSL: f(s_i) = s_i (mAP 49.9%)
            _u = torch.zeros_like(out_prob)
            _u[idx[0], idx[1], target_classes_o] = iou
            _t = _u.pow(self.ta_beta)
        else:  # 'quadratic' — the proposed method
            # Quadratic PSL: f(s_i) = s_i^2 (mAP 50.1%, optimal)
            # Convex function: severely penalizes mid-to-low IoU, ensures only
            # well-localized boxes receive high classification scores.
            _u = torch.zeros_like(out_prob)
            # Only fill in IoU values at [batch, query, class] positions corresponding to positive samples
            _u[idx[0], idx[1], target_classes_o] = iou
            # self.ta_beta=2.0, self.ta_alpha=0.0 (fixed in __init__)
            _t = _u.pow(self.ta_beta)

        # 3. Normalize dynamic targets for training stability
        norm_t = torch.zeros_like(_t)
        for i in range(bs):
            # Normalization factor: use max_iou = 1.0 directly
            normalizer = max((1.0 / (_t[i].max() + 1e-8)).detach(), 1)
            norm_t[i] = _t[i] * normalizer
        norm_t = norm_t.detach()

        # 4. Compute final loss using dynamic targets norm_t
        # Compute negative sample loss (target is 0)
        neg_loss = (1 - focal_alpha) * (out_prob ** focal_gamma) * (1 - target_classes_onehot) * (
            -(1 - out_prob + 1e-8).log())

        # Compute positive sample loss (target is norm_t)
        pos_loss = target_classes_onehot * (
                focal_alpha * ((norm_t - out_prob).abs() ** focal_gamma) * F.binary_cross_entropy(out_prob, norm_t,
                                                                                                  reduction='none')
        )

        # Aggregate losses
        loss_class = (pos_loss + neg_loss).sum() / num_boxes
        losses = {"loss_ce": loss_class}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device = next(iter(outputs.values())).device
        indices = self.matcher(outputs_without_aux, targets)

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs['dn_meta']

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.range(0, len(targets[i]['labels']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(
                    self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes * scalar, **kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            l_dict = dict()
            l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
            losses.update(l_dict)

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict = {}
                    for loss in self.losses:
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes * scalar,
                                                    **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    l_dict = dict()
                    l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups, pad_size = dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups

        return output_known_lbs_bboxes, single_pad, num_dn_groups


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=100, nms_iou_threshold=-1) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:, :, 2:] = boxes[:, :, 2:] - boxes[:, :, :2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.nms_iou_threshold > 0:
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b, s in zip(boxes, scores)]

            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in
                       zip(scores, labels, boxes, item_indices)]
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


@MODULE_BUILD_FUNCS.registe_with_name(module_name='dino')
def build_dino(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # if args.dataset_file == "coco_panoptic":
    #     # for panoptic, we just add a num_classes that is large enough to hold
    #     # max_obj_id + 1, but the exact value doesn't really matter
    #     num_classes = 250
    # if args.dataset_file == 'o365':
    #     num_classes = 366
    # if args.dataset_file == 'vanke':
    #     num_classes = 51
    # Dataset class counts (max_obj_id + 1):
    #   FLIR:  3 classes (person=1, car=2, bicycle=3)  -> num_classes=4
    #   LLVIP: 1 class  (person=1)                     -> num_classes=2
    #   KAIST: 1 class  (person=1)                     -> num_classes=2
    #   GIR:   5 classes (person, dog, car, bicycle, motorcycle) -> num_classes=6
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)
    backbone_ir = build_backbone(args)
    transformer = build_deformable_transformer(args)

    try:
        match_unstable_error = args.match_unstable_error
        dn_labelbook_size = args.dn_labelbook_size
    except:
        match_unstable_error = True
        dn_labelbook_size = num_classes

    try:
        dec_pred_class_embed_share = args.dec_pred_class_embed_share
    except:
        dec_pred_class_embed_share = True
    try:
        dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    except:
        dec_pred_bbox_embed_share = True

    model = DINO(
        backbone,
        backbone_ir,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        fix_refpoints_hw=args.fix_refpoints_hw,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_class_embed_share=dec_pred_class_embed_share,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        # two stage
        two_stage_type=args.two_stage_type,
        # box_share
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        decoder_sa_type=args.decoder_sa_type,
        num_patterns=args.num_patterns,
        dn_number=args.dn_number if args.use_dn else 0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    # for DN training
    if args.use_dn:
        weight_dict['loss_ce_dn'] = args.cls_loss_coef
        weight_dict['loss_bbox_dn'] = args.bbox_loss_coef
        weight_dict['loss_giou_dn'] = args.giou_loss_coef

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update(
            {k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses,
                             )
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_select=args.num_select, nms_iou_threshold=args.nms_iou_threshold)}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
