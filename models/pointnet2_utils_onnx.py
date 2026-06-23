"""
ONNX-Compatible PointNet++ Utilities
This module provides pure PyTorch implementations of PointNet++ operations
that are fully compatible with ONNX export (no loops, no dynamic control flow).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).unsqueeze(2)
    dist += torch.sum(dst ** 2, -1).unsqueeze(1)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, S, nsample]
    Return:
        new_points: indexed points data, [B, S, C] or [B, S, nsample, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample_onnx(xyz, npoint):
    """
    ONNX-compatible sampling using uniform stride-based sampling.
    Picks every (N // npoint)-th point, giving a spread that approximates FPS
    without any Python loops or random ops that break ONNX tracing.

    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    # Use tensor ops for B so batch dim stays dynamic in the ONNX graph.
    # N and C are fixed (1024 points, 3 coords) so Python ints are fine.
    N = xyz.shape[1]

    stride = N // npoint
    indices = torch.arange(npoint, dtype=torch.long, device=device) * stride  # [npoint]
    indices = indices.clamp(0, N - 1)
    # expand along batch dim using tensor shape, not a Python int
    centroids = indices.unsqueeze(0).expand(xyz.shape[0], npoint)  # [B, npoint]

    return centroids


def query_knn_point(k, xyz, new_xyz):
    """
    ONNX-compatible K-Nearest Neighbors query (replaces ball query).
    Uses fixed k instead of radius-based grouping.

    Input:
        k: number of nearest neighbors
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, k]
    """
    # Calculate pairwise distances
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]

    # Find k nearest neighbors
    _, group_idx = torch.topk(sqrdists, k, dim=-1, largest=False, sorted=True)

    return group_idx


def sample_and_group_onnx(npoint, nsample, xyz, points, use_knn=True):
    """
    ONNX-compatible sampling and grouping.

    Input:
        npoint: number of points to sample
        nsample: number of neighbors per point
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
        use_knn: use k-NN instead of ball query (more ONNX friendly)
    Return:
        new_xyz: sampled points position data, [B, npoint, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    C = xyz.shape[2]
    S = npoint

    # Farthest point sampling
    fps_idx = farthest_point_sample_onnx(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]

    # K-NN grouping
    idx = query_knn_point(nsample, xyz, new_xyz)  # [B, npoint, nsample]

    # Group points
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)  # [B, npoint, nsample, 3]

    if points is not None:
        grouped_points = index_points(points, idx)  # [B, npoint, nsample, D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, 3+D]
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


def sample_and_group_all_onnx(xyz, points):
    """
    ONNX-compatible group all points.

    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    new_xyz = torch.zeros_like(xyz[:, :1, :])        # [B, 1, 3] — dynamic B
    grouped_xyz = xyz.unsqueeze(1)                   # [B, 1, N, 3]
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.unsqueeze(1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstractionONNX(nn.Module):
    """ONNX-compatible Set Abstraction layer"""
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstractionONNX, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all_onnx(xyz, points)
        else:
            new_xyz, new_points = sample_and_group_onnx(self.npoint, self.nsample, xyz, points)

        # new_xyz: [B, npoint, C]
        # new_points: [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsgONNX(nn.Module):
    """ONNX-compatible Multi-Scale Grouping Set Abstraction layer"""
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgONNX, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        S = self.npoint

        # Sample points
        fps_idx = farthest_point_sample_onnx(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)

        new_points_list = []
        for i, nsample in enumerate(self.nsample_list):
            # Group points using k-NN
            idx = query_knn_point(nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)
            grouped_xyz -= new_xyz.unsqueeze(2)  # [B, S, 1, C] broadcast

            if points is not None:
                grouped_points = index_points(points, idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))

            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagationONNX(nn.Module):
    """ONNX-compatible Feature Propagation layer"""
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagationONNX, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.expand(-1, xyz1.shape[1], -1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.unsqueeze(-1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
