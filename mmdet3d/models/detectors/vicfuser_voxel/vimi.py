# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet3d.core import bbox3d2result, build_prior_generator
from mmdet3d.models.fusion_layers.point_fusion import point_sample
from mmdet.models.detectors import BaseDetector
from mmdet3d.models.builder import (
    DETECTORS,
    MODELS,
    build_backbone,
    build_head,
    build_neck,
    build_voxel_encoder,
    build_middle_encoder,
)
from torch import nn
from mmcv.cnn import ConvModule
from torch.nn import functional as F
from mmcv.runner import force_fp32, auto_fp16
import math
from mmdet3d.models.model_utils.naive_compressor import NaiveCompressor_UNet
from mmdet3d.models.detectors.vicfuser_voxel.vicfuser_voxel_ccm import Mlp, CCMNet
from mmdet3d.ops import Voxelization


def attention(query, key, mask=None, dropout=None):
    # from IPython import embed
    # embed()

    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        return tuple([x0, x1, x2, x3])


class DCN_Up_Conv_List(nn.Module):
    def __init__(self, neck_dcn, channels):
        super(DCN_Up_Conv_List, self).__init__()

        self.upconv0 = nn.Sequential(
            double_conv(channels, channels),
        )

        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            double_conv(channels, channels),
        )
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            double_conv(channels, channels),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            double_conv(channels, channels),
        )
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            double_conv(channels, channels),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            double_conv(channels, channels),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            double_conv(channels, channels),
        )

        self.dcn0 = build_neck(neck_dcn)
        self.dcn1 = build_neck(neck_dcn)
        self.dcn2 = build_neck(neck_dcn)
        self.dcn3 = build_neck(neck_dcn)

    def forward(self, x):
        assert x.__len__() == 4
        x0 = self.dcn0(x[0])
        x0 = self.upconv0(x0)

        x1 = self.dcn1(x[1])
        x1 = self.upconv1(x1)

        x2 = self.dcn2(x[2])
        x2 = self.upconv2(x2)

        x3 = self.dcn3(x[3])
        x3 = self.upconv3(x3)

        return [x0, x1, x2, x3]


class SS_NaiveCompressor(nn.Module):
    def __init__(self, input_dim, c_compress_ratio, s_compress_ratio):
        super(SS_NaiveCompressor, self).__init__()

        # from IPython import embed
        # embed(header='MSC')

        assert (s_compress_ratio <= 4) & (s_compress_ratio >= 0)
        s_compress_ratio0 = max(s_compress_ratio, 0)
        self.compressor0 = NaiveCompressor_UNet(
            input_dim, c_compress_ratio, s_compress_ratio0
        )

    def forward(self, x):
        x0 = self.compressor0(x)

        return x0


@DETECTORS.register_module()
class VIMI(BaseDetector):
    r"""`ImVoxelNet <https://arxiv.org/abs/2106.01178>`_."""

    def __init__(
        self,
        backbone,
        neck,
        neck_3d,
        bbox_head,
        n_voxels,
        anchor_generator,
        compress_ratio=1,
        s_compress_ratio=0,
        se_reduction_ratio=1,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
        neck_dcn=None,
    ):
        print("VIMI_0129.__init__")
        super().__init__(init_cfg=init_cfg)
        self.backbone_v = build_backbone(backbone)
        self.neck_v = build_neck(neck)
        self.backbone_i = build_backbone(backbone)
        self.neck_i = build_neck(neck)

        self.neck_3d = build_neck(neck_3d)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.n_voxels = n_voxels
        self.anchor_generator = build_prior_generator(anchor_generator)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.c_compress_ratio = compress_ratio
        self.s_compress_ratio = s_compress_ratio
        self.img_feat_channels = neck.out_channels
        self.img_feat_channels_c = self.img_feat_channels // self.c_compress_ratio
        self.se_reduction_ratio = se_reduction_ratio

        self.dcn_up_conv_v = DCN_Up_Conv_List(neck_dcn, self.img_feat_channels)
        self.dcn_up_conv_i = DCN_Up_Conv_List(neck_dcn, self.img_feat_channels)

        self.inf_compressor = SS_NaiveCompressor(
            self.img_feat_channels, self.c_compress_ratio, self.s_compress_ratio
        )
        self.ms_block_inf = MultiScaleBlock(
            self.img_feat_channels, self.img_feat_channels
        )

        self.ccmnet = CCMNet(
            self.img_feat_channels,
            self.img_feat_channels,
            self.img_feat_channels,
            self.se_reduction_ratio,
        )

    def extract_img_feat(self, img, img_metas):
        """Extract features from images."""
        bs = img.shape[0]
        img_v = img[:, 0, ...]
        img_i = img[:, 1, ...]

        x_v = self.backbone_v(img_v)
        x_v = self.neck_v(x_v)
        x_v = self.dcn_up_conv_v(list(x_v))
        x_v_tensor = torch.stack(x_v).permute(1, 0, 2, 3, 4)
        x_v_out = torch.mean(x_v_tensor, dim=1)

        x_i = self.backbone_i(img_i)
        x_i0 = self.neck_i(x_i)[0]
        # from IPython import embed
        # embed(header='compress')

        # Add compression encoder-decoder
        x_i0 = self.inf_compressor(x_i0)

        # from IPython import embed
        # embed(header='after comp')

        x_i = self.ms_block_inf(x_i0)

        x_i = self.dcn_up_conv_i(list(x_i))
        x_i_tensor = torch.stack(x_i).permute(1, 0, 2, 3, 4)

        # from IPython import embed
        # embed(header='global attention')
        # query.shape[B,C]
        # key.shape[B,N_levels,C]
        query = torch.mean(x_v_out, dim=(-2, -1))[:, None, :]
        key = torch.mean(x_i_tensor, dim=(-2, -1))
        weights_i = attention(query, key).squeeze(1)

        # print('attention_weights',weights_i)

        x_i_out = (weights_i[:, :, None, None, None] * x_i_tensor).sum(dim=1)

        return tuple((x_v_out, x_i_out))

    def extract_feat(self, img, img_metas):
        """Extract 3d features from the backbone -> fpn -> 3d projection.

        Args:
            img (torch.Tensor): Input images of shape (N, Num_Cam, C_in, H, W).
            img_metas (list): Image metas.

        Returns:
            torch.Tensor: of shape (N, C_out, N_x, N_y, N_z)
        """

        batch_size = img.shape[0]
        x_v, x_i = self.extract_img_feat(img, img_metas)

        x_v, x_i = self.ccmnet(x_v, x_i, img_metas)

        points = self.anchor_generator.grid_anchors(
            [self.n_voxels[::-1]], device=img.device
        )[0][:, :3]

        volumes_v = []
        for feature, img_meta in zip(x_v, img_metas):
            proj_mat_ex0 = points.new_tensor(img_meta["lidar2img"]["extrinsic"][0])
            proj_mat_in0 = points.new_tensor(img_meta["lidar2img"]["intrinsic"][0])
            proj_mats0 = proj_mat_in0 @ proj_mat_ex0

            img_scale_factor = (
                points.new_tensor(img_meta["scale_factor"][:2])
                if "scale_factor" in img_meta.keys()
                else 1
            )
            img_flip = img_meta["flip"] if "flip" in img_meta.keys() else False
            img_crop_offset = (
                points.new_tensor(img_meta["img_crop_offset"])
                if "img_crop_offset" in img_meta.keys()
                else 0
            )
            volume_v = point_sample(
                img_meta,
                img_features=feature[None, ...],
                points=points,
                proj_mat=proj_mats0,
                coord_type="LIDAR",
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img.shape[-2:],
                img_shape=img_meta["img_shape"][:2],
                aligned=False,
            )
            volumes_v.append(
                volume_v.reshape(self.n_voxels[::-1] + [-1]).permute(3, 2, 1, 0)
            )
        x_v = torch.stack(volumes_v)

        volumes_i = []
        for feature, img_meta in zip(x_i, img_metas):
            proj_mat_ex1 = points.new_tensor(img_meta["lidar2img"]["extrinsic"][1])
            proj_mat_in1 = points.new_tensor(img_meta["lidar2img"]["intrinsic"][1])
            proj_mats1 = proj_mat_in1 @ proj_mat_ex1

            img_scale_factor = (
                points.new_tensor(img_meta["scale_factor"][:2])
                if "scale_factor" in img_meta.keys()
                else 1
            )
            img_flip = img_meta["flip"] if "flip" in img_meta.keys() else False
            img_crop_offset = (
                points.new_tensor(img_meta["img_crop_offset"])
                if "img_crop_offset" in img_meta.keys()
                else 0
            )
            volume_i = point_sample(
                img_meta,
                img_features=feature[None, ...],
                points=points,
                proj_mat=proj_mats1,
                coord_type="LIDAR",
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img.shape[-2:],
                img_shape=img_meta["img_shape"][:2],
                aligned=False,
            )
            volumes_i.append(
                volume_i.reshape(self.n_voxels[::-1] + [-1]).permute(3, 2, 1, 0)
            )

        x_i = torch.stack(volumes_i)

        assert x_v.shape[0] == batch_size, "x_v shape[0] is not equal bs"
        assert x_i.shape[0] == batch_size, "x_i shape[0] is not equal bs"
        assert x_v.shape == x_i.shape

        # x_concat = torch.cat((x_v,x_i),dim=1)
        # x = self.conv_o(x_concat)

        # from IPython import embed
        # embed(header='xxx')

        x_stack = torch.stack((x_v, x_i), dim=0)
        x = torch.mean(x_stack, dim=0)

        # x [bs,C, X, Y, Z] [2,64,248,288,12]
        x = self.neck_3d(x)
        # x [[2,256,288,248]*1]
        return x

    def forward_train(self, img, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):
        """Forward of training.

        Args:
            img (torch.Tensor): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.

        Returns:
            dict[str, torch.Tensor]: A dictionary of loss components.
        """
        # from IPython import embed
        # embed(header='VICFuser_BEV.forward_train')

        x = self.extract_feat(img, img_metas)
        x = self.bbox_head(x)
        losses = self.bbox_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        """Forward of testing.

        Args:
            img (torch.Tensor): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.

        Returns:
            list[dict]: Predicted 3d boxes.
        """

        # from IPython import embed
        # embed(header='VICFuser_BEV.forward_test')

        # not supporting aug_test for now
        return self.simple_test(img, img_metas)

    def simple_test(self, img, img_metas):
        """Test without augmentations.

        Args:
            img (torch.Tensor): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.

        Returns:
            list[dict]: Predicted 3d boxes.
        """

        # from IPython import embed
        # embed(header='VICFuser_BEV.simple_test')
        x = self.extract_feat(img, img_metas)
        x = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(*x, img_metas)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test with augmentations.

        Args:
            imgs (list[torch.Tensor]): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        raise NotImplementedError


class ReduceInfTC(nn.Module):
    def __init__(self, channel):
        super(ReduceInfTC, self).__init__()
        self.conv1_2 = nn.Conv2d(
            channel // 2, channel // 4, kernel_size=3, stride=2, padding=0
        )
        self.bn1_2 = nn.BatchNorm2d(channel // 4, track_running_stats=True)
        self.conv1_3 = nn.Conv2d(
            channel // 4, channel // 8, kernel_size=3, stride=2, padding=0
        )
        self.bn1_3 = nn.BatchNorm2d(channel // 8, track_running_stats=True)
        self.conv1_4 = nn.Conv2d(
            channel // 8, channel // 64, kernel_size=3, stride=2, padding=1
        )
        self.bn1_4 = nn.BatchNorm2d(channel // 64, track_running_stats=True)

        self.deconv2_1 = nn.ConvTranspose2d(
            channel // 64, channel // 8, kernel_size=3, stride=2, padding=1
        )
        self.bn2_1 = nn.BatchNorm2d(channel // 8, track_running_stats=True)
        self.deconv2_2 = nn.ConvTranspose2d(
            channel // 8, channel // 4, kernel_size=3, stride=2, padding=0
        )
        self.bn2_2 = nn.BatchNorm2d(channel // 4, track_running_stats=True)
        self.deconv2_3 = nn.ConvTranspose2d(
            channel // 4,
            channel // 2,
            kernel_size=3,
            stride=2,
            padding=0,
            output_padding=1,
        )
        self.bn2_3 = nn.BatchNorm2d(channel // 2, track_running_stats=True)

    def forward(self, x):
        outputsize = x.shape
        # out = F.relu(self.bn1_1(self.conv1_1(x)))
        out = F.relu(self.bn1_2(self.conv1_2(x)))
        out = F.relu(self.bn1_3(self.conv1_3(out)))
        out = F.relu(self.bn1_4(self.conv1_4(out)))

        out = F.relu(self.bn2_1(self.deconv2_1(out)))
        out = F.relu(self.bn2_2(self.deconv2_2(out)))
        x_1 = F.relu(self.bn2_3(self.deconv2_3(out)))

        # x_1 = F.relu(self.bn2_4(self.deconv2_4(out)))
        return x_1


class PixelWeightedFusion(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PixelWeightedFusion, self).__init__()
        self.conv1_1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, stride=1, padding=1
        )
        self.bn1_1 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        return x_1


@DETECTORS.register_module()
class VIMI_Fusion(VIMI):
    def __init__(
        self,
        img_backbone,
        img_neck,
        img_neck_3d,
        pts_voxel_layer,
        pts_voxel_encoder,
        pts_middle_encoder,
        pts_backbone,
        pts_neck,
        bbox_head,
        n_voxels,
        anchor_generator,
        compress_ratio=1,
        s_compress_ratio=0,
        se_reduction_ratio=1,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
        img_neck_dcn=None,
    ):
        super().__init__(
            img_backbone,
            img_neck,
            img_neck_3d,
            bbox_head,
            n_voxels,
            anchor_generator,
            compress_ratio,
            s_compress_ratio,
            se_reduction_ratio,
            train_cfg,
            test_cfg,
            pretrained,
            init_cfg,
            img_neck_dcn,
        )
        self.pts_backbone = build_backbone(pts_backbone)
        self.pts_neck = build_neck(pts_neck)
        self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        self.pts_voxel_encoder = build_voxel_encoder(pts_voxel_encoder)
        self.pts_middle_encoder = build_middle_encoder(pts_middle_encoder)

        self.inf_voxel_layer = Voxelization(**pts_voxel_layer)
        self.inf_voxel_encoder = build_voxel_encoder(pts_voxel_encoder)
        self.inf_middle_encoder = build_middle_encoder(pts_middle_encoder)
        self.inf_backbone = build_backbone(pts_backbone)
        self.inf_neck = build_neck(pts_neck)

        self.pts_fusion_weighted = PixelWeightedFusion(768, 384)
        self.pts_encoder = ReduceInfTC(768)

        self.mod_fusion_weighted = PixelWeightedFusion(640, 256)

    def extract_img_feat_vimi(self, img, img_metas):
        return super().extract_feat(img, img_metas)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    @torch.no_grad()
    @force_fp32()
    def inf_voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.inf_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def extract_pts_feat(self, points, img_metas, points_view="vehicle"):
        """Extract features from points."""

        if points_view == "vehicle":
            voxels, num_points, coors = self.voxelize(points)
            voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
            batch_size = coors[-1, 0].item() + 1
            veh_x = self.pts_middle_encoder(voxel_features, coors, batch_size)
            veh_x = self.pts_backbone(veh_x)
            veh_x = self.pts_neck(veh_x)
            return veh_x

        elif points_view == "infrastructure":
            inf_voxels, inf_num_points, inf_coors = self.inf_voxelize(points)
            inf_voxel_features = self.inf_voxel_encoder(
                inf_voxels, inf_num_points, inf_coors
            )
            inf_batch_size = inf_coors[-1, 0].item() + 1
            inf_x = self.inf_middle_encoder(
                inf_voxel_features, inf_coors, inf_batch_size
            )
            inf_x = self.inf_backbone(inf_x)
            inf_x = self.inf_neck(inf_x)

            inf_x[0] = self.pts_encoder(inf_x[0])
            return inf_x

        else:
            raise Exception("Points View is Error: {}".format(points_view))

    def generate_matrix(self, theta, x0, y0):
        import numpy as np

        c = theta[0][0]
        s = theta[1][0]
        matrix = np.zeros((3, 3))
        matrix[0, 0] = c
        matrix[0, 1] = -s
        matrix[1, 0] = s
        matrix[1, 1] = c
        matrix[0, 2] = -c * x0 + s * y0 + x0
        matrix[1, 2] = -c * y0 - s * x0 + y0
        matrix[2, 2] = 1
        return matrix

    def pts_feature_fusion(self, veh_x, inf_x, img_metas, mode="fusion"):
        """Method II: Based on affine transformation."""
        wrap_feats_ii = []

        """
        for ii in range(len(veh_x[0])):
            inf_feature = inf_x[0][ii:ii+1]
            veh_feature = veh_x[0][ii:ii+1]

            calib_inf2veh_rotation = img_metas[ii]['inf2veh']['rotation']
            calib_inf2veh_translation = img_metas[ii]['inf2veh']['translation']
            inf_pointcloud_range = self.inf_voxel_layer.point_cloud_range
            # theta_rot = [[cos(-theta), sin(-theta), 0.0], [cos(-theta), sin(-theta), 0.0]], theta is in the lidar coordinate.
            # according to the relationship between lidar coordinate system and input coordinate system.
            theta_rot = torch.tensor([[calib_inf2veh_rotation[0][0], -calib_inf2veh_rotation[0][1], 0.0],
                                      [-calib_inf2veh_rotation[1][0], calib_inf2veh_rotation[1][1], 0.0]]).type(dtype=torch.float).cuda(next(self.parameters()).device)
            theta_rot = torch.unsqueeze(theta_rot, 0)
            grid_rot = F.affine_grid(theta_rot, size=torch.Size(veh_feature.shape), align_corners=False)
            # range: [-1, 1].
            # Moving right and down is negative.
            x_trans = -2 * calib_inf2veh_translation[0][0] / (inf_pointcloud_range[3] - inf_pointcloud_range[0])
            y_trans = -2 * calib_inf2veh_translation[1][0] / (inf_pointcloud_range[4] - inf_pointcloud_range[1])
            theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).cuda(next(self.parameters()).device)
            theta_trans = torch.unsqueeze(theta_trans, 0)
            grid_trans = F.affine_grid(theta_trans, size=torch.Size(veh_feature.shape), align_corners=False)

            warp_feat_rot = F.grid_sample(inf_feature, grid_rot, mode='bilinear', align_corners=False)
            warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear', align_corners=False)

            wrap_feats_ii.append(warp_feat_trans)
        """
        for ii in range(len(veh_x[0])):
            inf_feature = inf_x[0][ii : ii + 1]
            veh_feature = veh_x[0][ii : ii + 1]

            lidar_i2v = img_metas[ii]["lidar_i2v"]  # 4*4
            lidar_i2v_rot = lidar_i2v[:3, :3]
            lidar_i2v_trans = lidar_i2v[:3, 3]

            # calib_inf2veh_rotation = img_metas[ii]["inf2veh"]["rotation"]
            # calib_inf2veh_translation = img_metas[ii]["inf2veh"]["translation"]
            calib_inf2veh_rotation = lidar_i2v_rot
            calib_inf2veh_translation = lidar_i2v_trans
            inf_pointcloud_range = self.inf_voxel_layer.point_cloud_range
            
            # img_path = img_metas[ii]["img_info"]
            # pts_path = img_metas[ii]["pts_info"]
            # print("img_path: ", img_path)
            # print("pts_path: ", pts_path)
            # print("calib_inf2veh_rotation: ", calib_inf2veh_rotation)
            # print("calib_inf2veh_translation: ", calib_inf2veh_translation)


            theta_rot = (
                torch.tensor(
                    [
                        [
                            calib_inf2veh_rotation[0][0],
                            -calib_inf2veh_rotation[0][1],
                            0.0,
                        ],
                        [
                            -calib_inf2veh_rotation[1][0],
                            calib_inf2veh_rotation[1][1],
                            0.0,
                        ],
                        [0, 0, 1],
                    ]
                )
                .type(dtype=torch.float)
                .cuda(next(self.parameters()).device)
            )
            theta_rot = (
                torch.FloatTensor(self.generate_matrix(theta_rot, -1, 0))
                .type(dtype=torch.float)
                .cuda(next(self.parameters()).device)
            )
            # Moving right and down is negative.
            x_trans = (
                -2
                * calib_inf2veh_translation[0]
                / (inf_pointcloud_range[3] - inf_pointcloud_range[0])
            )
            y_trans = (
                -2
                * calib_inf2veh_translation[1]
                / (inf_pointcloud_range[4] - inf_pointcloud_range[1])
            )
            theta_trans = (
                torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans], [0.0, 0.0, 1]])
                .type(dtype=torch.float)
                .cuda(next(self.parameters()).device)
            )
            theta_r_t = torch.mm(theta_rot, theta_trans, out=None)

            grid_r_t = F.affine_grid(
                theta_r_t[0:2].unsqueeze(0),
                size=torch.Size(veh_feature.shape),
                align_corners=False,
            )
            warp_feat_trans = F.grid_sample(
                inf_feature, grid_r_t, mode="bilinear", align_corners=False
            )
            wrap_feats_ii.append(warp_feat_trans)

        wrap_feats = [torch.cat(wrap_feats_ii, dim=0)]

        if mode not in ["fusion", "inf_only", "veh_only"]:
            raise Exception("Mode is Error: {}".format(mode))
        if mode == "inf_only":
            return wrap_feats
        elif mode == "veh_only":
            return veh_x
        veh_cat_feats = [torch.cat([veh_x[0], wrap_feats[0]], dim=1)]
        veh_cat_feats[0] = self.pts_fusion_weighted(veh_cat_feats[0])

        return veh_cat_feats

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        points,
        infrastructure_points,
        **kwargs
    ):
        """Training forward function.

        Args:
            img (torch.Tensor): Input images of shape (N, C_in, H, W).
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        for ii in range(len(infrastructure_points)):
            infrastructure_points[ii][:, 3] = 255 * infrastructure_points[ii][:, 3]

        pts_feat_veh = self.extract_pts_feat(
            points, img_metas, points_view="vehicle"
        )  # [[2, 384, 248, 288]*1]
        pts_feat_inf = self.extract_pts_feat(
            infrastructure_points, img_metas, points_view="infrastructure"
        )  # [[2, 384, 248, 288]*1]

        pts_feat_fused = self.pts_feature_fusion(
            pts_feat_veh, pts_feat_inf, img_metas, mode="fusion"
        )  # [[2, 384, 248, 288]*1]

        img_feat_fused = self.extract_img_feat_vimi(
            img, img_metas
        )  # [[2, 256, 288, 248]*1]

        feat_fused = torch.cat(
            [img_feat_fused[0], pts_feat_fused[0].transpose(-1, -2)], dim=1
        )  # [2, 640, 288, 248]
        feat_fused = self.mod_fusion_weighted(feat_fused)

        out = self.bbox_head([feat_fused])
        losses = self.bbox_head.loss(*out, gt_bboxes_3d, gt_labels_3d, img_metas)
        return losses

    def forward_test(
        self, img, img_metas, points, infrastructure_points, rescale=False, **kwargs
    ):
        """Forward of testing.

        Args:
            img (torch.Tensor): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.

        Returns:
            list[dict]: Predicted 3d boxes.
        """

        # from IPython import embed
        # embed(header='VICFuser_BEV.forward_test')

        # not supporting aug_test for now
        return self.simple_test(img, points, img_metas, infrastructure_points, rescale)

    def simple_test(self, img, points, img_metas, infrastructure_points, rescale=False):
        """Test function without augmentaiton."""
        for ii in range(len(infrastructure_points)):
            infrastructure_points[ii][:, 3] = 255 * infrastructure_points[ii][:, 3]
        pts_feat_veh = self.extract_pts_feat(
            points, img_metas, points_view="vehicle"
        )  # [[2, 384, 248, 288]*1]
        pts_feat_inf = self.extract_pts_feat(
            infrastructure_points, img_metas, points_view="infrastructure"
        )  # [[2, 384, 248, 288]*1]
        pts_feat_fused = self.pts_feature_fusion(
            pts_feat_veh, pts_feat_inf, img_metas, mode="fusion"
        )  # [[2, 384, 248, 288]*1]

        img_feat_fused = self.extract_img_feat_vimi(
            img, img_metas
        )  # [[2, 256, 288, 248]*1]

        feat_fused = torch.cat(
            [img_feat_fused[0], pts_feat_fused[0].permute(0, 1, 3, 2)], dim=1
        )  # [2, 640, 288, 248]

        feat_fused = self.mod_fusion_weighted(feat_fused)

        out = self.bbox_head([feat_fused])

        bbox_list = self.bbox_head.get_bboxes(*out, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]
        return bbox_results
