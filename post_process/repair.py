import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from scipy import ndimage

from post_process.objective import (
    total_cost, total_enclosure_cost, compute_escape_count, compute_boundary_mask,
)
from post_process.solve_greedy_search import _dilate

# 3D 6-连通结构 (面邻接), 与 objective.py 中邻接项使用的邻域一致
_STRUCTURE_6CONN = ndimage.generate_binary_structure(3, 1)


def _find_components(mask: Tensor) -> tuple:
    """(W, H, D) bool -> (ndimage.label 输出的分量图, 分量个数)，6-连通。"""
    return ndimage.label(mask.cpu().numpy(), structure=_STRUCTURE_6CONN)


def _component_mask(components, comp_id: int, device) -> Tensor:
    """ndimage.label 输出 + 分量 id -> (1, W, H, D) bool mask。"""
    return torch.from_numpy(components == comp_id).to(device).unsqueeze(0)


def _frontier_all_label(labels_n: Tensor, comp_mask: Tensor, target_label: int) -> bool:
    """comp_mask 的邻居体素（frontier）是否非空，且全部标签都等于 target_label。"""
    frontier = _dilate(comp_mask) & ~comp_mask
    if not frontier.any():
        return False
    return bool((labels_n[frontier] == target_label).all())


def _dilate_component_greedy(
    labels_n: Tensor,
    C_unary_n: Tensor,
    C_adj: Tensor,
    comp_mask: Tensor,
    label_l: int,
    forbidden_enclosure_pairs: Optional[list[tuple[int, int]]],
    C_enclose: float,
    max_iter_per_component: int,
    verbose: bool,
    tag: str,
) -> Tensor:
    """对单个连通分量 comp_mask 逐层贪心 dilation，直到无法进一步降低 total_cost。"""
    cur_cost = _total_cost(labels_n, C_unary_n, C_adj, forbidden_enclosure_pairs, C_enclose)

    n_round = 0
    while n_round < max_iter_per_component:
        frontier = _dilate(comp_mask) & ~comp_mask & (labels_n != label_l)
        if not frontier.any():
            break

        candidate = torch.where(frontier, label_l, labels_n)
        cand_cost = _total_cost(candidate, C_unary_n, C_adj, forbidden_enclosure_pairs, C_enclose)

        if cand_cost >= cur_cost:
            break

        labels_n = candidate
        comp_mask = comp_mask | frontier
        cur_cost = cand_cost
        n_round += 1

        if verbose:
            print(f"{tag}: dilation round {n_round}, cost={cur_cost.item():.4f}")

    if verbose and n_round >= max_iter_per_component:
        print(f"{tag}: stopped at max_iter_per_component={max_iter_per_component}")

    return labels_n


def _total_cost(
    labels: Tensor,
    C_unary: Tensor,
    C_adj: Tensor,
    forbidden_enclosure_pairs: Optional[list[tuple[int, int]]],
    C_enclose: float,
) -> Tensor:
    """(1, W, H, D) labels -> scalar total cost (unary + adjacency [+ enclosure]).

    Computed in float64: enclosure violations contribute C_enclose (default
    1e6) per voxel, which can push the total into the 1e8-1e9 range. Against
    that magnitude, float32's ~7 significant digits can't resolve the much
    smaller unary/adjacency delta from dilating a handful of voxels, causing
    two genuinely different candidates to compare as bitwise-equal costs.
    """
    L = C_unary.shape[-1]
    cost = total_cost(C_unary.double(), F.one_hot(labels, L).double(), C_adj.double())
    # cost = total_cost(C_unary, F.one_hot(labels, L), C_adj)
    if forbidden_enclosure_pairs:
        cost = cost + total_enclosure_cost(labels, forbidden_enclosure_pairs, C_enclose)
    return cost.reshape(())


def greedy_dilation(
    X: Tensor,
    label_l: int,
    C_unary: Tensor,
    C_adj: Tensor,
    *,
    forbidden_enclosure_pairs: Optional[list[tuple[int, int]]] = None,
    C_enclose: float = 1e6,
    max_iter_per_component: int = 1000,
    verbose: bool = True,
) -> Tensor:
    """
    对标签 `label_l` 的每个连通分量做贪心膨胀（dilation）修复。

    Enclosure cost 优化容易把某个连通分量压缩成极窄的"膜"结构（例如只有一层
    体素厚），这里逐个连通分量地向外膨胀一层，只要膨胀后的 total_cost 比膨胀
    前更低就接受并继续膨胀，直到某一层膨胀不再降低 total_cost 为止，再处理
    下一个连通分量。

    :param X:         (N, W, H, D) int64，当前标签图
    :param label_l:   需要 dilate 的标签
    :param C_unary:   (N, W, H, D, L) 每体素每标签的 unary cost
    :param C_adj:     (L, L) 标签邻接 cost 矩阵
    :param forbidden_enclosure_pairs: 传给 `objective.total_enclosure_cost` 的
                      (inner_layer_id, outer_layer_id) 列表；None 时不计入
                      enclosure cost
    :param C_enclose: enclosure 违规的惩罚权重
    :param max_iter_per_component: 每个连通分量膨胀轮数的安全上限
    :param verbose:   是否打印每次膨胀的 cost 变化
    :return:          (N, W, H, D) int64，修复后的标签图（不修改输入）
    """
    device = X.device
    N = X.shape[0]
    X = X.clone().to(torch.int64)

    for n in range(N):
        labels_n = X[n:n + 1]          # (1, W, H, D)
        C_unary_n = C_unary[n:n + 1]   # (1, W, H, D, L)

        components, n_components = _find_components(labels_n[0] == label_l)

        for comp_id in range(1, n_components + 1):
            comp_mask = _component_mask(components, comp_id, device)  # (1, W, H, D)
            tag = f"[n={n}] label {label_l} component {comp_id}/{n_components}"
            labels_n = _dilate_component_greedy(
                labels_n, C_unary_n, C_adj, comp_mask, label_l,
                forbidden_enclosure_pairs, C_enclose, max_iter_per_component, verbose, tag,
            )

        X[n] = labels_n[0]

    return X


def hard_flip(
    X: Tensor,
    forbidden_enclosure_pairs: list[tuple[int, int]],
    verbose: bool = True,
) -> Tensor:
    """
    hard_flip修复算子:

    给定forbidden_enclosure_pairs,对其中每一个(inner, outer) pair
    在X上寻找每个非法inner的所有连通分量.

    对每个非法inner的每个连通分量,获取其邻居节点的label
    当邻居节点的label全部是outer时,将整个inner连通分量全部覆盖为outer

    :param X:  (N, W, H, D) int64，当前标签图
    :param forbidden_enclosure_pairs: (inner_layer_id, outer_layer_id) 列表；
                      "非法inner" 指该体素标签为 inner 且逃逸邻居数为 0
                      （即 `objective.enclosure_cost` 判定为被非法包裹的体素）
    :param verbose:   是否打印每次 flip 的信息
    :return:          (N, W, H, D) int64，修复后的标签图（不修改输入）
    """
    device = X.device
    N = X.shape[0]
    X = X.clone().to(torch.int64)

    for inner_layer_id, outer_layer_id in forbidden_enclosure_pairs:
        escape_count = compute_escape_count(X, inner_layer_id, outer_layer_id)  # (N, W, H, D)
        is_boundary = compute_boundary_mask(X)
        illegal_mask = (X == inner_layer_id) & (escape_count == 0) & ~is_boundary  # (N, W, H, D)

        for n in range(N):
            if not illegal_mask[n].any():
                continue

            components, n_components = _find_components(illegal_mask[n])

            for comp_id in range(1, n_components + 1):
                comp_mask = _component_mask(components, comp_id, device)  # (1, W, H, D)

                if _frontier_all_label(X[n:n + 1], comp_mask, outer_layer_id):
                    X[n:n + 1] = torch.where(comp_mask, outer_layer_id, X[n:n + 1])
                    if verbose:
                        print(f"[n={n}] hard_flip: inner={inner_layer_id} -> outer={outer_layer_id}, "
                              f"component {comp_id}/{n_components} ({comp_mask.sum().item()} voxels) flipped")

    return X


def greedy_dilate_with_hard_flip(
    X: Tensor,
    label_l: int,
    C_unary: Tensor,
    C_adj: Tensor,
    *,
    forbidden_enclosure_pairs: Optional[list[tuple[int, int]]] = None,
    C_enclose: float = 1e6,
    max_iter_per_component: int = 1000,
    verbose: bool = True,
) -> Tensor:
    """结合greedy dilation和hard flip

    在greedy dilate中添加hard flip的逻辑,即,当获得一个inner连通分量后,如果所有邻居全部都是outer,则将inner全部覆盖成outer;否则才进行greedy dilate

    :param X:         (N, W, H, D) int64，当前标签图
    :param label_l:   需要 dilate 的标签（即 hard flip 意义下的 inner）
    :param C_unary:   (N, W, H, D, L) 每体素每标签的 unary cost
    :param C_adj:     (L, L) 标签邻接 cost 矩阵
    :param forbidden_enclosure_pairs: (inner_layer_id, outer_layer_id) 列表；
                      其中 inner_layer_id == label_l 的条目给出 hard flip 检查
                      所用的 outer_layer_id，同时也传给 `total_enclosure_cost`
                      参与 greedy dilate 的 cost 计算
    :param C_enclose: enclosure 违规的惩罚权重
    :param max_iter_per_component: 每个连通分量膨胀轮数的安全上限
    :param verbose:   是否打印每次 flip / dilation 的信息
    :return:          (N, W, H, D) int64，修复后的标签图（不修改输入）
    """
    device = X.device
    N = X.shape[0]
    X = X.clone().to(torch.int64)

    outer_ids = [outer for inner, outer in (forbidden_enclosure_pairs or []) if inner == label_l]

    for n in range(N):
        labels_n = X[n:n + 1]          # (1, W, H, D)
        C_unary_n = C_unary[n:n + 1]   # (1, W, H, D, L)

        components, n_components = _find_components(labels_n[0] == label_l)

        for comp_id in range(1, n_components + 1):
            comp_mask = _component_mask(components, comp_id, device)  # (1, W, H, D)
            tag = f"[n={n}] label {label_l} component {comp_id}/{n_components}"

            outer_hit = next(
                (outer for outer in outer_ids if _frontier_all_label(labels_n, comp_mask, outer)), None
            )
            if outer_hit is not None:
                labels_n = torch.where(comp_mask, outer_hit, labels_n)
                if verbose:
                    print(f"{tag}: hard-flip -> {outer_hit} ({comp_mask.sum().item()} voxels)")
                continue

            labels_n = _dilate_component_greedy(
                labels_n, C_unary_n, C_adj, comp_mask, label_l,
                forbidden_enclosure_pairs, C_enclose, max_iter_per_component, verbose, tag,
            )

        X[n] = labels_n[0]

    return X