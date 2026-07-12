from torch import Tensor
from post_process import solve_sa_discrete, repair


def sa_cgd(
    C_unary: Tensor,
    C_adj: Tensor,
    logits,
    forbidden_enclosure_pair: list[tuple[int, int]],
    verbose: bool = True,
):
    """Simulated Annealing with Constrained Greedy Dilation"""
    best_logits = solve_sa_discrete.solve_sa(
        C_unary,
        C_adj,
        X_init=logits,
        T_final=1e-2,
        n_iter=500,
        C_enclose=4e7,
        forbidden_enclosure_pairs=forbidden_enclosure_pair,
        verbose=verbose,
    )

    second_logits = solve_sa_discrete.solve_sa(
        C_unary,
        C_adj,
        X_init=best_logits,
        n_iter=500,
        T_init=4e-2,
        T_final=4e-3,
        C_enclose=2e7,
        forbidden_enclosure_pairs=forbidden_enclosure_pair,
        verbose=verbose,
    )

    C_unary, C_adj = C_unary.double(), C_adj.double()

    dilated_labels = repair.constrained_greedy_dilation(
        X=second_logits.argmax(dim=-1),
        C_unary=C_unary,
        C_adj=C_adj,
        forbidden_enclosure_pairs=forbidden_enclosure_pair,
        C_enclose=1e5,
        verbose=verbose,
    )

    return best_logits, second_logits, dilated_labels

def sa_pd(
        C_unary: Tensor,
        C_adj: Tensor,
        logits,
        forbidden_enclosure_pair: list[tuple[int, int]],
        verbose: bool = True,
):
    """Simulated Annealing with Probabilistic Dilation"""
    optimized_logits = solve_sa_discrete.solve_sa(
        C_unary,
        C_adj,
        X_init=logits,

        # T_init=5e-2,
        T_final=1e-3,
        n_iter=500,
        C_enclose=4e7,
        forbidden_enclosure_pairs=forbidden_enclosure_pair,
    )

    repaired_labels = repair.probability_dilation(
        X=optimized_logits.argmax(dim=-1),
        logits=logits,
        forbidden_enclosure_pairs=forbidden_enclosure_pair,
        min_inner_neighbors=4,
        threshold=0.71
    )

    return optimized_logits, repaired_labels