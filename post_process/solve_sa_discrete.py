
"""Simulated annealing for MRF labeling with a genuinely discrete decision variable.

Unlike `solve_sa_continuous.solve_sa` (which anneals continuous logits and
recovers a hard labeling via argmax), the state here is the integer label
map itself: `labels` of shape (N, W, H, D). Each sweep proposes, for every
voxel, a new label drawn uniformly from the L-1 labels *other than* its
current one — the natural neighborhood move for a discrete label space —
and accepts/rejects each proposal independently via the Metropolis
criterion, using the standard Gibbs local field (unary + adjacency cost
against the voxel's 6 neighbors, all other voxels held fixed).

As with the continuous and greedy solvers, same-color (checkerboard)
voxels are never 6-adjacent, so updating one color at a time keeps every
accept/reject decision exact within a half-sweep.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable

from post_process.objective import total_cost, local_enclosure_bias, total_enclosure_cost
from post_process.solve_greedy_search import _local_cost, _checkerboard


def _propose_labels(labels: Tensor, L: int, generator: Optional[torch.Generator]) -> Tensor:
    """Propose a new label for every voxel, drawn uniformly from the L-1
    labels other than its current one.

    `offset` in [1, L-1] shifts `labels` cyclically mod L, so `proposal`
    ranges uniformly over the other L-1 labels and is never equal to
    `labels` (offset == 0 excluded).

    :param labels: (N, W, H, D) int64
    :return:       (N, W, H, D) int64, proposal != labels everywhere
    """
    offset = torch.randint(
        1, L, labels.shape, dtype=labels.dtype, device=labels.device, generator=generator
    )
    return (labels + offset) % L


def solve_sa(
    C_unary: Tensor,
    C_adj: Tensor,
    X_init: Optional[Tensor] = None,
    *,
    forbidden_enclosure_pairs: Optional[list[tuple[int, int]]] = None,
    C_enclose: float = 1e6,
    T_init: float = 1.0,
    T_final: float = 1e-4,
    n_iter: int = 200,
    generator: Optional[torch.Generator] = None,
    validate_fn: Callable = None,
    verbose: bool = True,
) -> Tensor:
    """
    Simulated annealing for MRF labeling with a discrete decision variable,
    via checkerboard-parallel single-site relabeling moves.

    Optimization variable: integer label map `labels` of shape (N, W, H, D).
    There is no relaxation/argmax step — every state visited during the
    search is already a valid one-label-per-voxel assignment.

    Each sweep proposes, for every voxel, a new label uniform over the
    L-1 labels other than its current one, then accepts/rejects each
    voxel's proposal independently via the Metropolis criterion using that
    voxel's own local unary+adjacency energy (all other voxels held
    fixed). To keep every decision exact (a voxel's acceptance must not
    depend on a simultaneously-changing neighbor), the volume is split
    into two checkerboard color classes and updated in two half-steps per
    sweep.

    :param C_unary:   (N, W, H, D, L) per-voxel-per-label unary costs
    :param C_adj:     (L, L) pairwise adjacency cost matrix
    :param X_init:    optional warm-start logits/one-hot (N, W, H, D, L);
                       defaults to the per-voxel unary-greedy labeling
                       (argmin over C_unary)
    :param forbidden_enclosure_pairs: list of (inner_layer_id, outer_layer_id)
                       pairs that must not fully enclose one another; see
                       `objective.enclosure_cost`. None disables the term.
    :param C_enclose: penalty weight applied per forbidden-enclosure violation
    :param T_init:    initial temperature
    :param T_final:   final temperature
    :param n_iter:    number of SA sweeps
    :param generator: torch Generator for reproducibility
    :param validate_fn: optional callback(labels) -> printed each iter when verbose
    :param verbose:   print progress per sweep
    :return:          (N, W, H, D, L) one-hot label assignment
    """
    device = C_unary.device
    orig_dtype = C_unary.dtype
    N, W, H, D, L = C_unary.shape

    labels = (
        C_unary.argmin(dim=-1) if X_init is None else X_init.argmax(dim=-1)
    ).to(torch.int64)  # (N, W, H, D)

    C_adj = C_adj.to(dtype=orig_dtype, device=device)
    color_a = _checkerboard((W, H, D), device).unsqueeze(0)  # (1, W, H, D)
    masks = [color_a, ~color_a]

    def _total_cost(labels: Tensor) -> Tensor:
        cost = total_cost(C_unary, F.one_hot(labels, L).to(orig_dtype), C_adj)  # (N, 1)
        if forbidden_enclosure_pairs:
            cost = cost + total_enclosure_cost(labels, forbidden_enclosure_pairs, C_enclose)
        return cost

    cost = _total_cost(labels)
    T = float(T_init)
    decay = (T_final / T_init) ** (1.0 / max(n_iter - 1, 1))

    best = {'iter': 0, 'labels': labels.clone(), 'cost': cost}

    for i in range(n_iter):
        proposal = _propose_labels(labels, L, generator)  # (N, W, H, D)

        for mask in masks:
            local_cost = _local_cost(C_unary, C_adj, labels)  # (N, W, H, D, L)
            if forbidden_enclosure_pairs:
                # safe as a per-half-step bias: a voxel's escape count only
                # depends on its (currently fixed) neighbors, never on its
                # own candidate label.
                local_cost = local_enclosure_bias(local_cost, labels, forbidden_enclosure_pairs, C_enclose)
            cur_e = local_cost.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            prop_e = local_cost.gather(-1, proposal.unsqueeze(-1)).squeeze(-1)
            delta = prop_e - cur_e  # (N, W, H, D), per-voxel cost change

            log_u = torch.log(
                torch.rand(labels.shape, dtype=orig_dtype, device=device, generator=generator) + 1e-20
            )
            # Metropolis accept / reject, independently per voxel
            accept = ((delta <= 0) | (log_u < -delta / T)) & mask

            labels = torch.where(accept, proposal, labels)

        cost = _total_cost(labels)

        T *= decay

        if cost < best['cost']:
            best['iter'] = i
            best['labels'] = labels.clone()
            best['cost'] = cost

        if verbose:
            if validate_fn is not None:
                valid_result = validate_fn(labels)
                print(f"Iter {i} | Cost: {cost.mean():.4f} | Valid Result: {valid_result}")
            else:
                print(f"Iter {i} | Cost: {cost.mean():.4f}")

    if verbose:
        print(f"Best Iter: {best['iter']} | Cost: {best['cost']}")

    return F.one_hot(best['labels'], L).to(orig_dtype)