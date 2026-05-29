# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2.pointnet2_modules import PointnetSAModuleMSG, PointnetFPModule


class PointNet2MSG(nn.Module):
    def __init__(self, radii_list, dim_in=6, use_xyz=True):
        super(PointNet2MSG, self).__init__()
        self.SA_modules = nn.ModuleList()
        c_in = dim_in
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=radii_list[0],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 16, 16, 32]],
                use_xyz=use_xyz,
                bn=True,
            )
        )
        c_out_0 = 32 + 32

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=radii_list[1],
                nsamples=[16, 32],
                mlps=[[c_in, 32, 32, 64], [c_in, 32, 32, 64]],
                use_xyz=use_xyz,
                bn=True,
            )
        )
        c_out_1 = 64 + 64

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=radii_list[2],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 64, 128]],
                use_xyz=use_xyz,
                bn=True,
            )
        )
        c_out_2 = 128 + 128

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=radii_list[3],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 128, 256], [c_in, 128, 128, 256]],
                use_xyz=use_xyz,
                bn=True,
            )
        )
        c_out_3 = 256 + 256

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + dim_in, 256, 256], bn=True))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + c_out_0, 256, 256], bn=True))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 256, 256], bn=True))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512], bn=True))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud):
        _, N, _ = pointcloud.size()

        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features[0]