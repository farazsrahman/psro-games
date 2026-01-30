import numpy as np
from typing import List, Tuple

def calculate_ne_regret(
    U: np.ndarray,
    p_full: np.ndarray,
    q_full: np.ndarray,
) -> float:
    """
    Calculate the max player regret for strategy (p_full, q_full) on the full game U.
    U is payoff for player 1. U.T is payoff for player 2.
    Regret = max over players of (Best Response Payoff - Current Payoff).
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
    
    # Return max regret over players (not sum/NashConv)
    return max(exp1, exp2)


def calculate_cce_regret(
    U_full: np.ndarray,
    mu_res: np.ndarray,
    pop1: List[int],
    pop2: List[int],
) -> float:
    """
    Calculate the CCE Regret of the empirical joint distribution mu_res when played in the full game.
    CCE Regret is the max regret over players who can change their distribution to a fixed strategy 
    instead of playing the CCE joint action.
    """
    if mu_res is None:
        return float('nan')

    # Reconstruct full joint to get marginals? 
    # Or just work with indices for efficiency.
    
    # Player 1 (Row Player)
    # Expected payoff under mu_res
    # P1 Payoff = sum_{i,j} mu[i,j] * U[pop1[i], pop2[j]]
    
    # To compute efficiently:
    # 1. Compute marginals of mu_res
    # mu_res shape: (len(pop1), len(pop2))
    marg_q = np.sum(mu_res, axis=0) # Player 2's marginal over pop2
    
    # 2. Player 1's expected payoff under mu_res
    # We can use the restricted block
    U_res = U_full[np.ix_(pop1, pop2)]
    curr_payoff_1 = np.sum(mu_res * U_res)
    
    # 3. Player 1's max deviation payoff: Max_k (U[k, :] @ q_full)
    # q_full is 0 everywhere except pop2
    # So we compute U_full[:, pop2] @ marg_q
    dev_payoffs_1 = U_full[:, pop2] @ marg_q
    max_dev_1 = np.max(dev_payoffs_1)
    
    regret1 = max(0.0, max_dev_1 - curr_payoff_1)
    
    # Player 2 (Col Player)
    marg_p = np.sum(mu_res, axis=1) # Player 1's marginal over pop1
    
    # P2 Payoff (using symmetry U_full.T for P2 payoffs)
    U_res_p2 = U_full.T[np.ix_(pop2, pop1)] # Note indices order for T: pop2 are rows of T
    # Wait, U.T has shape (n,n). Row i of U.T corresponds to action i of P2.
    # mu_res[i,j] is prob P1 plays pop1[i] and P2 plays pop2[j].
    # Contribution to P2: mu_res[i,j] * U.T[pop2[j], pop1[i]]
    # This matches U_res_p2[j, i] * mu_res[i, j]
    
    # Easier: P2 payoff = sum(mu_res * U_p2_res) where U_p2_res[i,j] = U.T[pop2[j], pop1[i]] ?
    # NO. U_res is U[pop1, pop2]. P2 payoff matrix given P1 (row) and P2 (col) is (U[pop1, pop2]).T?
    # No, U is P1 payoff. If symmetric, P2 payoff is U^T.
    # So P2 payoff for (a1, a2) is U[a2, a1].
    # Current payoff P2: sum_{i,j} mu[i,j] * U[pop2[j], pop1[i]]
    
    curr_payoff_2 = 0.0
    for i_idx, i_act in enumerate(pop1):
        for j_idx, j_act in enumerate(pop2):
            # P2 payoff matrix element [i_act, j_act] is U_full.T[i_act, j_act] (if U.T is P2's payoff)
            # Standard: U is P1 payoff. U[i, j]. 
            # Symmetric game P2 payoff: u2(i, j) = u1(j, i) = U[j, i] = U.T[i, j].
            # So calculating sum mu[i,j] * U.T[i, j] is correct.
            curr_payoff_2 += mu_res[i_idx, j_idx] * U_full.T[i_act, j_act]
            
    # Deviation for P2: Max_k (Sum_i p_i u2(i, k))
    # u2(i, k) = U.T[i, k].
    # So we need sum_i p_i * U.T[i, k].
    # This corresponds to vector (p @ U.T) where p is row vector.
    # Restricting to pop1: marg_p @ U_full.T[pop1, :]
    # U_full.T[pop1, :] selects rows of U.T corresponding to pop1.
    dev_payoffs_2 = marg_p @ U_full.T[pop1, :]
    max_dev_2 = np.max(dev_payoffs_2)
    
    regret2 = max(0.0, max_dev_2 - curr_payoff_2)
    
    return max(regret1, regret2)





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
