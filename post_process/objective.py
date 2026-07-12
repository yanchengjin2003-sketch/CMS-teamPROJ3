import torch
import torch.nn.functional as F
from torch import Tensor


def labeling_cost(C: Tensor, X: Tensor) -> Tensor:
    """Cost for labeling voxel v as label l, for every v and every l.

    :param C: (N, W, H, D, L)
    :param X: (N, W, H, D, L)
    :return:  (N, 1)
    """
    assert C.shape == X.shape, f'Shape mismatch: {C.shape}, {X.shape}'
    assert C.dim() == X.dim() == 5, f'Dimension mismatch: {C.dim()}, {X.dim()}'
    return (C * X).sum(dim=(1, 2, 3, 4), keepdim=True).reshape(-1, 1)


# ─────────────────────────────────────────────
# Adjacency Cost Matrix Builders
# ─────────────────────────────────────────────

def smoothness_penalty(
    num_label: int,
    weight: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: str = None,
) -> Tensor:
    """
    Penalize any adjacent pair of DIFFERENT labels.
    C[l, l'] = weight if l != l' else 0

    :return: (num_label, num_label)
    """
    C = torch.ones((num_label, num_label), dtype=dtype, device=device) * weight
    C.fill_diagonal_(0.0)
    return C


def forbidden_adjacency_penalty(
    num_label: int,
    forbidden_pairs: list[tuple[int, int]],
    weight: float = 1e6,
    dtype: torch.dtype = torch.float32,
    device: str = None,
) -> Tensor:
    """
    Penalize biologically forbidden label adjacencies.

    :return: (num_label, num_label)  (symmetric)
    """
    C = torch.zeros((num_label, num_label), dtype=dtype, device=device)
    for l, l_prime in forbidden_pairs:
        C[l, l_prime] = weight
        C[l_prime, l] = weight
    return C


def combine_adj_costs(*cost_matrices: Tensor) -> Tensor:
    """Additively combine multiple (num_label, num_label) cost matrices."""
    result = torch.zeros_like(cost_matrices[0])
    for C in cost_matrices:
        result = result + C
    return result


# ─────────────────────────────────────────────
# Core Computation
# ─────────────────────────────────────────────

# All 6 axis-aligned neighbor directions in 3D
# Each entry: (axis, shift_direction)
_DIRECTIONS: list[tuple[int, int]] = [
    (0, +1), (0, -1),  # x-axis
    (1, +1), (1, -1),  # y-axis
    (2, +1), (2, -1),  # z-axis
]


def _shift_X(X: Tensor, axis: int, shift: int) -> Tensor:
    """
    Shift X along a spatial axis, padding boundary with zeros.
    Axis 0,1,2 correspond to width, height, depth (spatial dims 1,2,3 in X).

    :param X: (N, W, H, D, L)
    :return:  (N, W, H, D, L)  — neighbor values
    """
    spatial_axis = axis + 1  # offset by 1 due to batch dim
    result = torch.roll(X, shifts=-shift, dims=spatial_axis)

    # zero-out the wrapped boundary (treat boundary as "no neighbor")
    slices = [slice(None)] * X.dim()
    if shift > 0:
        slices[spatial_axis] = slice(-shift, None)   # last `shift` slices
    else:
        slices[spatial_axis] = slice(None, -shift)   # first `|shift|` slices
    result[tuple(slices)] = 0.0
    return result


def adjacency_cost(
    X: Tensor,
    C_adj: Tensor,
    directions: list[tuple[int, int]] = _DIRECTIONS,
) -> Tensor:
    """
    Compute total adjacency cost: Σ_{uv∈E} Σ_{l,l'} C[l,l'] * X[u,l] * X[v,l']

    :param X:     (N, W, H, D, L)
    :param C_adj: (L, L) — shared across all edges
    :param directions: which neighbor directions to include
    :return:      (N, 1)
    """
    assert X.dim() == 5, f'X must be 5D, got {X.dim()}'
    assert C_adj.dim() == 2 and C_adj.shape[0] == C_adj.shape[1] == X.shape[-1]

    C_adj = C_adj.to(dtype=X.dtype, device=X.device)
    total = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)

    for axis, shift in directions:
        X_neighbor = _shift_X(X, axis, shift)  # (N, W, H, D, L)
        cost = torch.einsum('nwhdi,ij,nwhdj->n', X, C_adj, X_neighbor)
        total = total + cost

    return total.reshape(-1, 1)


# ─────────────────────────────────────────────
# Enclosure Cost
# ─────────────────────────────────────────────

def _face_neighbor_kernel(device, dtype) -> Tensor:
    """6-连通（面邻接）conv3d 卷积核，用于统计某个体素性质的面邻居个数。"""
    kernel = torch.zeros(1, 1, 3, 3, 3, device=device, dtype=dtype)
    kernel[0, 0, 1, 1, 0] = kernel[0, 0, 1, 1, 2] = 1
    kernel[0, 0, 1, 0, 1] = kernel[0, 0, 1, 2, 1] = 1
    kernel[0, 0, 0, 1, 1] = kernel[0, 0, 2, 1, 1] = 1
    return kernel


def compute_escape_count(labels: Tensor, inner_layer_id: int, outer_layer_id: int) -> Tensor:
    """(N, W, H, D) -> 每个体素的'逃逸邻居'个数（标签既非膜也非胞质的邻居数）"""
    is_escape = ((labels != inner_layer_id) & (labels != outer_layer_id)).to(torch.float32)
    is_escape = is_escape.unsqueeze(1)  # (N, 1, W, H, D)
    kernel = _face_neighbor_kernel(labels.device, is_escape.dtype)
    return F.conv3d(is_escape, kernel, padding=1).squeeze(1)  # (N, W, H, D)


def compute_self_neighbor_count(labels: Tensor, inner_layer_id: int) -> Tensor:
    """(N, W, H, D) -> 每个体素标签为 inner_layer_id 的面邻居个数（0~6）。"""
    is_self = (labels == inner_layer_id).to(torch.float32).unsqueeze(1)  # (N, 1, W, H, D)
    kernel = _face_neighbor_kernel(labels.device, is_self.dtype)
    return F.conv3d(is_self, kernel, padding=1).squeeze(1)  # (N, W, H, D)


def compute_boundary_mask(labels: Tensor) -> Tensor:
    """(N, W, H, D) bool 掩码，标记体素是否处于图像的空间边界（W/H/D任一维的首末切片）。"""
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, 0, :, :] = True
    mask[:, -1, :, :] = True
    mask[:, :, 0, :] = True
    mask[:, :, -1, :] = True
    mask[:, :, :, 0] = True
    mask[:, :, :, -1] = True
    return mask


def enclosure_cost(labels: Tensor, inner_layer_id: int, outer_layer_id: int, C_enclose) -> Tensor:
    escape_count = compute_escape_count(labels, inner_layer_id, outer_layer_id)
    self_neighbor_count = compute_self_neighbor_count(labels, inner_layer_id)
    is_boundary = compute_boundary_mask(labels)
    # 6 个面邻居全部是自身 label（inner_layer_id）时不计入惩罚：这只是local 逐体素近似，
    # 没有做连通分量级别的全局逃逸判定（那样无法棋盘格并行、耗时不可接受），
    # 因此需要豁免"深埋在合法厚结构内部、邻居全是自身"的体素，否则会被误判为非法包裹。
    all_self = self_neighbor_count == 6
    # 边界体素缺失的邻居会被 conv3d 的零填充误判为"逃逸邻居数为0"，因此边界体素永远不计入被非法包裹的惩罚。
    return C_enclose * ((escape_count == 0) & ~all_self & ~is_boundary).float()


def local_enclosure_bias(local_cost, labels, forbidden_enclosure_pairs, C_enclose) -> Tensor:
    enclosure_bias = torch.zeros_like(local_cost)
    for (inner_layer_id, outer_layer_id) in forbidden_enclosure_pairs:
        enclosure_bias[..., inner_layer_id] += enclosure_cost(labels, inner_layer_id, outer_layer_id, C_enclose)
    local_cost = local_cost + enclosure_bias
    return local_cost

def total_enclosure_cost(
    labels: Tensor,
    forbidden_enclosure_pairs: list[tuple[int, int]],
    C_enclose: float | int = 1e6,
) -> Tensor:
    """Scalar enclosure penalty actually incurred by `labels`.

    Mirrors `local_enclosure_bias`: for each forbidden (inner, outer) pair,
    every voxel currently labeled `inner_layer_id` that has zero non-inner/
    outer neighbors (i.e. is fully enclosed) contributes `C_enclose`.

    :param labels: (N, W, H, D) int64
    :return:       (N, 1)
    """
    N = labels.shape[0]
    total = torch.zeros(N, dtype=torch.float32, device=labels.device)
    for (inner_layer_id, outer_layer_id) in forbidden_enclosure_pairs:
        penalty = enclosure_cost(labels, inner_layer_id, outer_layer_id, C_enclose)  # (N, W, H, D)
        is_inner = (labels == inner_layer_id).to(penalty.dtype)
        total = total + (penalty * is_inner).sum(dim=(1, 2, 3))
    return total.reshape(-1, 1)

# ─────────────────────────────────────────────
# Total Cost Builder
# ─────────────────────────────────────────────

def total_cost(
    C_unary: Tensor,
    X: Tensor,
    C_adj: Tensor,
    undirected: bool = True,
) -> Tensor:
    """
    Full objective: unary + adjacency terms.

    :param C_unary:    (N, W, H, D, L)
    :param X:          (N, W, H, D, L)
    :param C_adj:      (L, L)
    :param undirected: if True, use only 3 directions to avoid double-counting
    :return:           (N, 1)
    """
    directions = [(0, 1), (1, 1), (2, 1)] if undirected else _DIRECTIONS
    return (
        labeling_cost(C_unary, X)
        + adjacency_cost(X, C_adj, directions=directions)
    )