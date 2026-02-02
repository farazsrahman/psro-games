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
    U: np.ndarray,
    mu: np.ndarray,
    rows: List[int] = None,
    cols: List[int] = None,
    U2: np.ndarray = None,
) -> float:
    """
    Calculate the CCE Regret of the joint distribution mu.
    
    If rows/cols are provided, mu is treated as a distribution over the restricted game defined by those indices,
    and regret is calculated against the full game U (Exploitability).
    
    If rows/cols are None, mu is treated as a distribution over the game U itself (Internal Regret).
    """
    if mu is None:
        return float('nan')

    n_rows, n_cols = U.shape
    
    # Resolve indices and restricted game dimensions
    if rows is None:
        rows = list(range(n_rows))
    if cols is None:
        cols = list(range(n_cols))
        
    if U2 is None:
        if U.shape[0] == U.shape[1] and np.allclose(U, U.T): # Symmetric check roughly
             # If strictly symmetric game logic is desired, usually U2 = U.T
             U2 = U.T
        else:
             U2 = U.T # Default to zero-sum/symmetric assumption if not provided? 
                      # In the original code, calculate_regret_cce did U2 = U_res.T
                      # In calculate_cce_regret, P2 payoff was U.T.
                      # So U2 = U.T seems consistent.

    # --- Player 1 (Row Player) ---
    # Expected payoff under mu
    # We need the submatrix of U corresponding to the active strategies in mu
    U_sub = U[np.ix_(rows, cols)]
    curr_payoff_1 = np.sum(mu * U_sub)
    
    # Marginal of P2 over the columns of U
    # q_full is 0 everywhere except 'cols'
    # The marginal of mu corresponds to indices in 'cols'
    marg_q_sub = np.sum(mu, axis=0)
    
    # Deviation payoffs: U @ q_full
    # We can compute this as U[:, cols] @ marg_q_sub
    dev_payoffs_1 = U[:, cols] @ marg_q_sub
    max_dev_1 = np.max(dev_payoffs_1)
    
    regret1 = max(0.0, max_dev_1 - curr_payoff_1)
    
    # --- Player 2 (Col Player) ---
    # Expected payoff under mu
    U2_sub = U2[np.ix_(rows, cols)]
    curr_payoff_2 = np.sum(mu * U2_sub)
    
    # Marginal of P1 over the rows of U
    marg_p_sub = np.sum(mu, axis=1)
    
    # Deviation payoffs for P2: p_full @ U2
    # p_full is 0 everywhere except 'rows'
    # We can compute this as marg_p_sub @ U2[rows, :]
    dev_payoffs_2 = marg_p_sub @ U2[rows, :]
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


def calculate_regret_ce(U_res: np.ndarray, mu: np.ndarray, U2: np.ndarray = None) -> float:
    """
    Calculates the maximum regret for a Correlated Equilibrium (CE).
    
    We compute the "Swap Regret" or "Internal Regret", defined as the maximum expected gain 
    if a player could replace every occurrence of action i with optimal action k.
    
    Regret = max_i max_k sum_j mu[i,j] * (U[k,j] - U[i,j])
             (Gain from swapping i -> k)
             
    Wait, usually Swap Regret is sum_i max_k (Gain(i->k)).
    Or max_phi sum_s P(s) (u(phi(s)) - u(s)).
    
    Here we return the MAXIMUM Regret over players.
    For each player, we calculate:
    Total_Regret = max_{phi: A->A} E[ u(phi(s_i), s_{-i}) - u(s) ]
                 = sum_i max_k E[ u(k, s_{-i}) 1_{s_i=i} - u(i, s_{-i}) 1_{s_i=i} ]
                 = sum_i max_k ( sum_j mu[i,j] (U[k,j] - U[i,j]) )
                 
    This handles numerical stability naturally (no division by P(i)).
    """
    n1, n2 = U_res.shape
    if U2 is None:
        U2 = U_res.T
        
    # --- Player 1 Regret ---
    # We want to find max_phi sum_{i,j} mu[i,j] (U[phi(i), j] - U[i,j])
    # = sum_i max_k sum_j mu[i,j] (U[k,j] - U[i,j])
    
    total_regret_p1 = 0.0
    for i in range(n1):
        # Current expected utility when playing i: sum_j mu[i,j] U[i,j]
        current_u_contribution = np.dot(mu[i, :], U_res[i, :])
        
        # Best deviation from i: max_k sum_j mu[i,j] U[k,j]
        # Calculate vector of utilities for all k against P(j|i)*P(i) = mu[i,:]
        dev_utilities = U_res @ mu[i, :] # shape (n1,)
        best_dev_u_contribution = np.max(dev_utilities)
        
        gain = best_dev_u_contribution - current_u_contribution
        # Gain should be non-negative because one option for k is i itself.
        # But numerically could be -epsilon.
        if gain > 0:
            total_regret_p1 += gain
            
    # --- Player 2 Regret ---
    # mu[:, j] is dist over P1 when P2 plays j.
    total_regret_p2 = 0.0
    for j in range(n2):
        # Current exp util contribution when playing j
        current_u_contribution = np.dot(mu[:, j], U2[:, j])
        
        # Best dev from j: max_l sum_i mu[i, j] U2[i, l]
        # U2 is (n1, n2). mu[:,j] is (n1,)
        # p_vector = mu[:, j]
        # dev_utils[l] = sum_i p_vector[i] * U2[i, l]
        # = p_vector @ U2 -> shape (n2,)
        dev_utilities = mu[:, j] @ U2
        best_dev_u_contribution = np.max(dev_utilities)
        
        gain = best_dev_u_contribution - current_u_contribution
        if gain > 0:
            total_regret_p2 += gain
        
    return max(total_regret_p1, total_regret_p2)
