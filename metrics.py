import numpy as np
from typing import List

def calculate_ne_regret(
    U: np.ndarray,
    p_full: np.ndarray,
    q_full: np.ndarray,
) -> float:
    """
    Calculate the NashConv (sum of player regrets) for strategy (p_full, q_full) on the full game U.
    U is payoff for player 1. U.T is payoff for player 2 (assuming symmetric structure in generation).
    """
    # Payoff under (p,q)
    v1 = p_full @ U @ q_full
    v2 = p_full @ U.T @ q_full

    # Best response for p1 to q
    br1_payoffs = U @ q_full
    v_br1 = br1_payoffs.max()

    # Best response for p2 to p
    br2_payoffs = p_full @ U.T
    v_br2 = br2_payoffs.max()

    # Improvement gains for each player
    exp1 = v_br1 - v1
    exp2 = v_br2 - v2
    return exp1 + exp2


def calculate_cce_regret_full(
    U_full: np.ndarray,
    mu_res: np.ndarray,
    pop1: List[int],
    pop2: List[int],
) -> float:
    """
    Calculate the Regret of the empirical CCE when played in the full game.
    Regret is defined as the expected gain from the best unilateral deviation 
    given the recommendations from mu.
    """
    if mu_res is None:
        return float('nan')
        
    n_res1, n_res2 = mu_res.shape
    regret1 = 0.0
    
    # Player 1 analysis
    for i_idx, i_act in enumerate(pop1):
        # Marginal probability of recommending i
        prob_i = np.sum(mu_res[i_idx, :])
        if prob_i < 1e-9:
            continue
            
        # Calculate vector V_j = mu[i, j] for all j
        weights_j = mu_res[i_idx, :] # Shape (n_res2,)
        
        # Current payoff contribution: sum_j weights_j * U_full[i_act, pop2[j]]
        curr_payoff = 0
        for j_idx, j_act in enumerate(pop2):
            curr_payoff += weights_j[j_idx] * U_full[i_act, j_act]
            
        # Deviation payoff: max_k sum_j weights_j * U_full[k, pop2[j]]
        # Vectorized: U_full[:, pop2] @ weights_j -> (N_full,)
        dev_payoffs = U_full[:, pop2] @ weights_j
        max_dev = np.max(dev_payoffs)
        
        gain = max_dev - curr_payoff
        if gain > 0:
            regret1 += gain

    # Player 2 analysis
    regret2 = 0.0
    for j_idx, j_act in enumerate(pop2):
        weights_i = mu_res[:, j_idx] # Shape (n_res1,)
        prob_j = np.sum(weights_i)
        if prob_j < 1e-9:
            continue
            
        curr_payoff = 0
        for i_idx, i_act in enumerate(pop1):
            curr_payoff += weights_i[i_idx] * U_full.T[i_act, j_act]

        # Deviation: max_k sum_i weights_i * U2_full[i, k]
        # weights_i @ U2_full[pop1, :] -> shape (N_full,)
        # U2_full = U_full.T
        dev_payoffs = weights_i @ U_full.T[pop1, :]
        max_dev = np.max(dev_payoffs)
        
        gain = max_dev - curr_payoff
        if gain > 0:
            regret2 += gain

    return regret1 + regret2


def extract_restricted_game(
    U: np.ndarray,
    pop1: List[int],
    pop2: List[int],
) -> np.ndarray:
    """Extract a restricted payoff matrix using populations indices."""
    return U[np.ix_(pop1, pop2)]


def uniform_welfare(U: np.ndarray) -> float:
    """Compute uniform-strategy welfare for symmetric games."""
    return float(np.mean(U + U.T))
