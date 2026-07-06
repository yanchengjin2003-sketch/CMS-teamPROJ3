"""Efficient greedy local search (queue-driven ICM) for MRF labeling.

Algorithm sketch:
    0. active := every voxel (the initial "work queue")
    1. while active is not empty:
        a. for each voxel v in active, greedy_flip(v) tries every other
           label and keeps the one with lowest local cost
        b. X_star := voxels where that flip actually lowers the local cost
        c. flip every voxel in X_star to its new label
        d. active := neighbors of X_star (their local cost may have
           changed now that a neighbor's label changed)

The literal reading of `greedy_flip` — recompute `total_cost` on the whole
volume for every candidate label of every voxel — is O(volume) per
candidate, i.e. O(volume^2 * L) per round. Since flipping voxel v only
changes the unary term at v and the adjacency terms on v's 6 incident
edges, that whole-volume recompute is unnecessary: `_local_cost` computes,
for every voxel and every candidate label, exactly that delta in one
batched tensor pass over the whole volume (O(volume * L) per round).

Flips are also restricted to one checkerboard color at a time
(`_checkerboard`): naively flipping every improving voxel in `active`
simultaneously can let two adjacent voxels each flip based on the other's
now-stale label, which is not guaranteed to lower the joint cost and can
oscillate forever. Voxels of the same color are never adjacent (6-
connectivity), so within one color sub-step every flip's cost was computed
against neighbors that are genuinely still fixed — each accepted flip
strictly lowers the total cost, which guarantees termination.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from post_process.objective import total_cost, total_enclosure_cost, local_enclosure_bias

# All 6 axis-aligned neighbor directions in 3D: (axis, shift_direction)
_DIRECTIONS: list[tuple[int, int]] = [
    (0, +1), (0, -1),
    (1, +1), (1, -1),
    (2, +1), (2, -1),
]


def _shift_labels(labels: Tensor, axis: int, shift: int) -> tuple[Tensor, Tensor]:
    """Shift a (N, W, H, D) label map by one voxel along `axis`.

    `shifted[p] == labels[p + shift]` along that axis, i.e. the label of
    the neighbor in direction `(axis, shift)` from `p`.

    :return: (shifted_labels, valid_mask) — `valid_mask` is False at
             boundary voxels that have no such neighbor; `shifted_labels`
             is filled with 0 there (meaningless, must be masked by caller).
    """
    spatial_axis = axis + 1  # offset by 1 for the batch dim
    shifted = torch.roll(labels, shifts=-shift, dims=spatial_axis)

    valid = torch.ones_like(labels, dtype=torch.bool)
    slices = [slice(None)] * labels.dim()
    slices[spatial_axis] = slice(-shift, None) if shift > 0 else slice(None, -shift)
    valid[tuple(slices)] = False

    shifted = shifted.clone()
    shifted[tuple(slices)] = 0
    return shifted, valid


def _local_cost(C_unary: Tensor, C_adj: Tensor, labels: Tensor) -> Tensor:
    """Cost of labeling every voxel with every candidate label, holding all
    *other* voxels fixed at their current label — the vectorized core of
    `greedy_flip`, batched over the whole volume and all L candidates.

    :param C_unary: (N, W, H, D, L)
    :param C_adj:   (L, L)
    :param labels:  (N, W, H, D) int64, current label of every voxel
    :return:        (N, W, H, D, L) cost if that voxel took each label
    """
    cost = C_unary.clone()

    for axis, shift in _DIRECTIONS:
        neighbor_labels, valid = _shift_labels(labels, axis, shift)
        if shift > 0:
            # edge (v, neighbor): C_adj[l_v, l_neighbor] — l_v (candidate) is the row
            pair_cost = C_adj[:, neighbor_labels].permute(1, 2, 3, 4, 0)
        else:
            # edge (neighbor, v): C_adj[l_neighbor, l_v] — l_v (candidate) is the column
            pair_cost = C_adj[neighbor_labels, :]
        cost = cost + pair_cost * valid.unsqueeze(-1)

    return cost


def _checkerboard(shape: tuple[int, int, int], device) -> Tensor:
    """(W, H, D) parity mask splitting voxels into two 6-connectivity-independent sets."""
    w, h, d = shape
    iw = torch.arange(w, device=device).view(w, 1, 1)
    ih = torch.arange(h, device=device).view(1, h, 1)
    id_ = torch.arange(d, device=device).view(1, 1, d)
    return ((iw + ih + id_) % 2).bool()


def _dilate(mask: Tensor) -> Tensor:
    """(N, W, H, D) bool -> union of each True voxel's 6 neighbors (self excluded)."""
    out = torch.zeros_like(mask)
    for axis, shift in _DIRECTIONS:
        shifted, valid = _shift_labels(mask.to(torch.int64), axis, shift)
        out |= shifted.bool() & valid
    return out


def solve_greedy(
    C_unary: Tensor,
    C_adj: Tensor,
    X_init: Optional[Tensor] = None,
    *,
    forbidden_enclosure_pairs: Optional[list[tuple[int, int]]] = None,
    C_enclose: float = 1e6,
    max_iter: int = 1000,
    verbose: bool = True,
) -> Tensor:
    """
    Queue-driven greedy local search (ICM) for MRF labeling.

    :param C_unary: (N, W, H, D, L) per-voxel-per-label unary costs
    :param C_adj:   (L, L) pairwise adjacency cost matrix
    :param X_init:  optional warm-start logits/one-hot (N, W, H, D, L);
                     defaults to the per-voxel unary-greedy labeling
                     (argmin over C_unary)
    :param forbidden_enclosure_pairs: list of (inner_layer_id, outer_layer_id)
                     pairs that must not fully enclose one another; see
                     `objective.enclosure_cost`. None disables the term.
    :param C_enclose: penalty weight applied per forbidden-enclosure violation
    :param max_iter: safety cap on the number of queue-processing rounds
    :param verbose:  print progress per round
    :return:         (N, W, H, D, L) one-hot label assignment
    """
    device = C_unary.device
    orig_dtype = C_unary.dtype
    N, W, H, D, L = C_unary.shape

    labels = (
        C_unary.argmin(dim=-1) if X_init is None else X_init.argmax(dim=-1)
    ).to(torch.int64)  # (N, W, H, D)

    C_adj = C_adj.to(dtype=orig_dtype, device=device)
    color = _checkerboard((W, H, D), device).unsqueeze(0)  # (1, W, H, D)

    active = torch.ones((N, W, H, D), dtype=torch.bool, device=device)

    n_round = 0
    while active.any() and n_round < max_iter:
        flipped = torch.zeros_like(active)

        for c in (0, 1):
            group = active & (color == c)
            if not group.any():
                continue

            local_cost = _local_cost(C_unary, C_adj, labels)         # (N, W, H, D, L)
            if forbidden_enclosure_pairs:
                # safe as a per-half-step bias: a voxel's escape count only
                # depends on its (currently fixed) neighbors, never on its
                # own candidate label.
                local_cost = local_enclosure_bias(local_cost, labels, forbidden_enclosure_pairs, C_enclose)
            best_cost, best_label = local_cost.min(dim=-1)           # (N, W, H, D)
            current_cost = local_cost.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

            improve = group & (best_cost < current_cost)
            labels = torch.where(improve, best_label, labels)
            flipped |= improve

        active = _dilate(flipped)
        n_round += 1

        if verbose:
            cost = total_cost(C_unary, F.one_hot(labels, L).to(orig_dtype), C_adj)
            if forbidden_enclosure_pairs:
                cost = cost + total_enclosure_cost(labels, forbidden_enclosure_pairs, C_enclose)
            print(f"Round {n_round} | flipped {flipped.sum().item()} | Cost: {cost.mean():.4f}")

    if verbose and n_round >= max_iter and active.any():
        print(f"solve_greedy: stopped at max_iter={max_iter} with {active.sum().item()} voxels still queued")

    return F.one_hot(labels, L).to(orig_dtype)
