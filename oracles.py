
import numpy as np
from typing import Tuple, List

def compute_best_responses(
    U: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Compute best-response payoffs and actions for both players."""
    br1_payoffs = U @ q
    a_star = int(np.argmax(br1_payoffs))

    br2_payoffs = p @ U.T
    b_star = int(np.argmax(br2_payoffs))

    return br1_payoffs, br2_payoffs, a_star, b_star


def compute_rectified_best_responses(
    U_full: np.ndarray,
    sigma_res: np.ndarray,
    pop1: List[int],
    pop2: List[int],
) -> Tuple[List[int], List[int]]:
    """
    Compute Rectified Best Responses (JPSRO(CE) expansion).
    Returns unique sets of new best response actions for P1 and P2.
    """
    n_res1, n_res2 = sigma_res.shape
    new_br1 = set()
    new_br2 = set()
    epsilon = 1e-9

    # Optimize by iterating only over non-zero elements
    # But dense iteration is fine for small games (10x10 or 50x50)
    
    # 1. P1 Analysis (Constraint for row i)
    # P1 observes signal i (from its own active set), infers P2 plays from P(j|i)
    for i_idx in range(n_res1):
        row_i = sigma_res[i_idx, :]
        row_sum = row_i.sum()
        if row_sum > epsilon:
            belief_p2 = row_i / row_sum
            
            # Map belief to full game strategy
            q_full = np.zeros(U_full.shape[1])
            q_full[pop2] = belief_p2
            
            # BR
            payoffs = U_full @ q_full
            br_action = int(np.argmax(payoffs))
            new_br1.add(br_action)

    # 2. P2 Analysis (Constraint for col j)
    # P2 observes signal j, infers P1 plays from P(i|j)
    for j_idx in range(n_res2):
        col_j = sigma_res[:, j_idx]
        col_sum = col_j.sum()
        if col_sum > epsilon:
            belief_p1 = col_j / col_sum
            
            p_full = np.zeros(U_full.shape[0])
            p_full[pop1] = belief_p1
            
            payoffs = p_full @ U_full.T
            br_action = int(np.argmax(payoffs))
            new_br2.add(br_action)

    return list(new_br1), list(new_br2)
