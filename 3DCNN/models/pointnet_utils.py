"""
models/pointnet_utils.py — Pure PyTorch PointNet++ building blocks.

No CUDA extensions (pointnet2_ops) — runs on CPU and Colab GPU.
"""

import torch
import torch.nn as nn


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Compute squared Euclidean distance between every pair of points.

    Args:
        src : (B, N, 3)
        dst : (B, M, 3)

    Returns:
        dist : (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2.0 * torch.bmm(src, dst.permute(0, 2, 1))   # (B, N, M)
    dist += (src ** 2).sum(dim=-1, keepdim=True)           # (B, N, 1)
    dist += (dst ** 2).sum(dim=-1, keepdim=True).permute(0, 2, 1)  # (B, 1, M)
    return dist.clamp(min=0.0)


def farthest_point_sample(xyz: torch.Tensor, n_points: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS).

    Args:
        xyz      : (B, N, 3)
        n_points : number of centroids to sample

    Returns:
        idx : (B, n_points) — indices into xyz
    """
    B, N, _ = xyz.shape
    device = xyz.device
    idx = torch.zeros(B, n_points, dtype=torch.long, device=device)
    distance = torch.full((B, N), float("inf"), device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(n_points):
        idx[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest, :].unsqueeze(1)  # (B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)                 # (B, N)
        distance = torch.min(distance, dist)
        farthest = distance.argmax(dim=-1)

    return idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Index into a point set using an index tensor.

    Args:
        points : (B, N, C)
        idx    : (B, S) or (B, S, K)

    Returns:
        indexed: (B, S, C) or (B, S, K, C)
    """
    B = points.shape[0]
    device = points.device
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = (
        torch.arange(B, dtype=torch.long, device=device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    return points[batch_idx, idx, :]


def ball_query(
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
    radius: float,
    n_sample: int,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Ball query: for each centroid, find up to n_sample neighbors within radius.

    Points beyond radius are filled with the centroid index (padding).

    Memory-safe chunked implementation: proses centroid per chunk agar tidak
    mengalokasi distance matrix penuh (B, S, N) sekaligus. Untuk S=512,
    N=16384, B=1024, matrix penuh = ~17 GB. Dengan chunk=256, peak = ~4 GB.

    Args:
        xyz     : (B, N, 3) — all points
        new_xyz : (B, S, 3) — centroids
        radius  : float
        n_sample: int
        chunk_size: proses centroid sebanyak ini per step (default 256)

    Returns:
        idx : (B, S, n_sample) — indices into xyz
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device = xyz.device

    # Pre-compute squared norm dari xyz agar tidak dihitung ulang per chunk
    xyz_norm = (xyz ** 2).sum(dim=-1, keepdim=True)           # (B, N, 1)

    sorted_idx_chunks = []
    dist_sorted_chunks = []

    for start in range(0, S, chunk_size):
        end = min(start + chunk_size, S)
        new_xyz_chunk = new_xyz[:, start:end, :]              # (B, chunk, 3)

        # Distance chunk: (B, chunk, N)
        dist_chunk = -2.0 * torch.bmm(
            new_xyz_chunk, xyz.permute(0, 2, 1)
        )
        dist_chunk += (new_xyz_chunk ** 2).sum(dim=-1, keepdim=True)   # (B, chunk, 1)
        dist_chunk += xyz_norm.permute(0, 2, 1)                        # (B, 1, N)
        dist_chunk = dist_chunk.clamp(min=0.0)

        # topk per chunk — tidak perlu matrix penuh
        d_sorted, s_idx = torch.topk(dist_chunk, n_sample, dim=-1, largest=False)
        sorted_idx_chunks.append(s_idx)
        dist_sorted_chunks.append(d_sorted)

    # Concatenate hasil chunk
    sorted_idx = torch.cat(sorted_idx_chunks, dim=1)      # (B, S, n_sample)
    dist_sorted = torch.cat(dist_sorted_chunks, dim=1)    # (B, S, n_sample)

    mask = dist_sorted > radius ** 2
    # Replace out-of-radius indices with the centroid itself (first neighbor = closest)
    group_first = sorted_idx[:, :, 0:1].expand_as(sorted_idx)
    sorted_idx = sorted_idx.masked_fill(mask, 0)
    # Clamp to centroid's nearest index where masked
    sorted_idx[mask] = group_first[mask]

    return sorted_idx


class SetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction layer.

    Args:
        n_point  : number of centroids to sample
        radius   : ball query radius
        n_sample : max neighbors per ball
        in_ch    : input feature channels (including XYZ if applicable)
        mlp_dims : list of output dims for MLP layers
    """

    def __init__(
        self,
        n_point: int,
        radius: float,
        n_sample: int,
        in_ch: int,
        mlp_dims: list[int],
    ):
        super().__init__()
        self.n_point = n_point
        self.radius = radius
        self.n_sample = n_sample

        layers = []
        last_ch = 3 + in_ch  # relative xyz (3) concatenated with features inside forward
        for dim in mlp_dims:
            layers += [
                nn.Conv2d(last_ch, dim, 1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            ]
            last_ch = dim
        self.mlp = nn.Sequential(*layers)
        self.out_ch = last_ch

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz      : (B, N, 3)  — point coordinates
            features : (B, N, C)  — per-point features

        Returns:
            new_xyz  : (B, S, 3)      — centroid coordinates
            new_feat : (B, S, C_out)  — aggregated features
        """
        B, N, _ = xyz.shape
        S = self.n_point

        # Sample centroids
        fps_idx = farthest_point_sample(xyz, S)         # (B, S)
        new_xyz = index_points(xyz, fps_idx)            # (B, S, 3)

        # Ball query — group neighbors
        group_idx = ball_query(xyz, new_xyz, self.radius, self.n_sample)  # (B, S, K)
        grouped_xyz = index_points(xyz, group_idx)      # (B, S, K, 3)
        grouped_xyz -= new_xyz.unsqueeze(2)             # relative coords

        grouped_feat = index_points(features, group_idx)  # (B, S, K, C)

        # Concatenate relative XYZ with features for first SA layer
        # (for SA2/SA3 the caller already concatenates xyz into features)
        x = torch.cat([grouped_xyz, grouped_feat], dim=-1)  # (B, S, K, 3+C)

        # MLP: operate per-point per-group
        x = x.permute(0, 3, 1, 2)   # (B, 3+C, S, K)
        x = self.mlp(x)              # (B, C_out, S, K)

        # Max pool over neighbors
        x = x.max(dim=-1)[0]        # (B, C_out, S)
        new_feat = x.permute(0, 2, 1)  # (B, S, C_out)

        return new_xyz, new_feat
