import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable

from post_process.objective import total_cost, total_enclosure_cost, local_enclosure_bias, _shift_X, _DIRECTIONS


def hard_assign(X: Tensor, out_dtype: torch.dtype) -> Tensor:
    """
    Enforce one-label-per-voxel constraint.
    argmax over label dim is equivalent to softmax → argmax (softmax is monotone).

    :param X:        (N, W, H, D, L) continuous logits
    :param out_dtype: torch dtype for the output one-hot tensor
    :return:         (N, W, H, D, L) one-hot, dtype=out_dtype
    """
    labels = X.argmax(dim=-1)                                       # (N, W, H, D)
    return F.one_hot(labels, num_classes=X.shape[-1]).to(out_dtype) # (N, W, H, D, L)


def _checkerboard_mask(W: int, H: int, D: int, device) -> Tensor:
    """
    2-color the voxel grid so that no two same-color voxels share a
    6-connected edge. Updating one color at a time makes independent,
    per-voxel Metropolis accept/reject exact (a voxel's neighbors are all
    the other color, hence fixed during its own update) — standard
    "red-black" sweep for Gibbs/SA sampling on a grid MRF.

    :return: (W, H, D) bool, True = color A
    """
    ix = torch.arange(W, device=device).view(W, 1, 1)
    iy = torch.arange(H, device=device).view(1, H, 1)
    iz = torch.arange(D, device=device).view(1, 1, D)
    return (ix + iy + iz) % 2 == 0


def _local_field(Xh: Tensor, C_unary: Tensor, C_adj: Tensor) -> Tensor:
    """
    Per-voxel local energy (unary + adjacency) for every candidate label,
    holding every other voxel's current label fixed. This is the standard
    Gibbs "local field": voxel v's own contribution plus its interaction
    with each of its 6 neighbors under their current labels.

    :param Xh:      (N, W, H, D, L) one-hot current labeling
    :param C_unary: (N, W, H, D, L)
    :param C_adj:   (L, L)
    :return:        (N, W, H, D, L) energy if voxel took each label
    """
    field = C_unary.clone()
    for axis, shift in _DIRECTIONS:
        X_neighbor = _shift_X(Xh, axis, shift)  # (N, W, H, D, L), zero at boundary
        field = field + torch.einsum('ij,nwhdj->nwhdi', C_adj, X_neighbor)
    return field


def solve_sa(
    C_unary: Tensor,
    C_adj: Tensor,
    step_init: float,
    X_init: Optional[Tensor] = None,
    *,
    forbidden_enclosure_pairs: Optional[list[tuple[int, int]]] = None,
    C_enclose: float = 1e6,
    T_init: float = 1.0,
    T_final: float = 1e-4,
    n_iter: int = 200,
    generator: Optional[torch.Generator] = None,
    validate_fn: Callable = None,
) -> Tensor:
    """
    Simulated annealing for MRF labeling via checkerboard-parallel single-site updates.

    Optimization variable: continuous logit tensor X of shape (N, W, H, D, L).
    Constraint: exactly one label per voxel, enforced via argmax at each
    evaluation — equivalent to softmax → argmax.

    Each sweep proposes a new logit for every voxel, then accepts/rejects
    each voxel's proposal independently via the Metropolis criterion, using
    that voxel's own local unary+adjacency energy (all other voxels held
    fixed) — this is the textbook single-variable SA move, not a joint
    accept/reject over the whole volume. To keep every decision exact (a
    voxel's acceptance must not depend on a simultaneously-changing
    neighbor), the volume is split into two checkerboard color classes and
    updated in two half-steps per sweep.

    :param C_unary:   (N, W, H, D, L) per-voxel-per-label unary costs
    :param C_adj:     (L, L) pairwise adjacency cost matrix
    :param step_init: initial perturbation magnitude
    :param X_init:    optional warm-start logits; defaults to −C_unary
    :param forbidden_enclosure_pairs: list of (inner_layer_id, outer_layer_id)
                       pairs that must not fully enclose one another; see
                       `objective.enclosure_cost`. None disables the term.
    :param C_enclose: penalty weight applied per forbidden-enclosure violation
    :param T_init:    initial temperature
    :param T_final:   final temperature
    :param n_iter:    number of SA sweeps
    :param generator: torch Generator for reproducibility
    :return:          (N, W, H, D, L) one-hot label assignment
    """
    device = C_unary.device
    orig_dtype = C_unary.dtype

    N, W, H, D, L = C_unary.shape
    C_unary64 = C_unary.to(torch.float64)
    C_adj64 = C_adj.to(dtype=torch.float64, device=device)

    # X = −C_unary: low-cost labels get high logits → argmax recovers the
    # unary-greedy solution as the starting point.
    X = -C_unary64 if X_init is None else X_init.to(torch.float64)
    Xh = hard_assign(X, torch.float64)

    color_a = _checkerboard_mask(W, H, D, device)  # (W, H, D) bool
    masks = [color_a, ~color_a]

    def _total_cost(Xh: Tensor) -> Tensor:
        labels = Xh.argmax(dim=-1)
        cost = total_cost(C_unary, Xh.to(orig_dtype), C_adj)  # (N, 1)
        if forbidden_enclosure_pairs:
            cost = cost + total_enclosure_cost(labels, forbidden_enclosure_pairs, C_enclose)
        return cost

    cost = _total_cost(Xh)  # (N, 1)
    T = float(T_init)
    decay = (T_final / T_init) ** (1.0 / max(n_iter - 1, 1))

    best = {'iter': 0, 'X': X, 'cost': cost}

    for i in range(n_iter):
        step = step_init * (T / T_init)
        noise = torch.empty_like(X).uniform_(-step, step, generator=generator)
        X_prop = X + noise
        Xh_prop = hard_assign(X_prop, torch.float64)  # (N, W, H, D, L)

        for mask in masks:
            field = _local_field(Xh, C_unary64, C_adj64)  # (N, W, H, D, L)
            if forbidden_enclosure_pairs:
                # safe as a per-half-step bias: a voxel's escape count only
                # depends on its (currently fixed) neighbors, never on its
                # own candidate label.
                labels = Xh.argmax(dim=-1)
                field = local_enclosure_bias(field, labels, forbidden_enclosure_pairs, C_enclose)
            cur_e = (Xh * field).sum(dim=-1)               # (N, W, H, D)
            prop_e = (Xh_prop * field).sum(dim=-1)         # (N, W, H, D)
            delta = prop_e - cur_e                          # (N, W, H, D), per-voxel cost change

            log_u = torch.log(
                torch.rand((N, W, H, D), dtype=torch.float64, device=device, generator=generator) + 1e-20
            )
            # Metropolis accept / reject, independently per voxel
            accept = ((delta <= 0) | (log_u < -delta / T)) & mask  # (N, W, H, D)

            accept_full = accept.unsqueeze(-1)
            X = torch.where(accept_full, X_prop, X)
            Xh = torch.where(accept_full, Xh_prop, Xh)

        cost = _total_cost(Xh)  # (N, 1)

        T *= decay

        if cost < best['cost']:
            best['iter'] = i
            best['X'] = X
            best['cost'] = cost

        if validate_fn is not None:
            valid_result = validate_fn(X)
            print(f"Iter {i} | Cost: {cost.mean():.4f} | Valid Result: {valid_result}")
        else:
            print(f"Iter {i} | Cost: {cost.mean():.4f}")

    print(f"Best Iter: {best['iter']} | Cost: {best['cost']}")

    return hard_assign(best['X'], orig_dtype)