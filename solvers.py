
import numpy as np
import cvxpy as cp
from typing import Tuple, List, Optional, Any

def fictitious_play_symmetric(
    U: np.ndarray,
    n_steps: int = 300,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate NE of a symmetric 2p game using fictitious play.
    Returns (p, q): mixed strategies for players 1 and 2.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, n2 = U.shape
    if n != n2:
        raise ValueError("Symmetric game must have a square payoff matrix.")
    # Start with uniform actions
    count_p = np.ones(n)
    count_q = np.ones(n)

    # Opponent empirical strategies
    p = count_p / count_p.sum()
    q = count_q / count_q.sum()

    for _ in range(n_steps):
        # Best response for p1 to q
        br1_payoffs = U @ q
        i_star = int(np.argmax(br1_payoffs))

        # Best response for p2 to p
        br2_payoffs = p @ U.T
        j_star = int(np.argmax(br2_payoffs))

        count_p[i_star] += 1.0
        count_q[j_star] += 1.0

        p = count_p / count_p.sum()
        q = count_q / count_q.sum()

    return p, q


def build_cce_constraints(
    U_res: np.ndarray,
    mu: cp.Variable,
) -> List:
    """Construct CCE feasibility constraints for a symmetric restricted game."""
    n1, n2 = U_res.shape
    constraints = [cp.sum(mu) == 1, mu >= 0]
    U1 = U_res
    U2 = U_res # For symmetric games, P2(i, j) = P1(j, i) = U_res[j, i]. So rows of U2 (P2 actions) are rows of U_res.

    # Player 1 constraints
    for i in range(n1):
        for k in range(n1):
            if i == k:
                continue
            gain_vec = U1[i, :] - U1[k, :]
            constraints.append(cp.sum(cp.multiply(mu[i, :], gain_vec)) >= 0)

    # Player 2 constraints
    # u2(i, j) = U_res[j, i] for symmetric games, so use rows of U2 (which is U_res.T)
    for j in range(n2):
        for k in range(n2):
            if j == k:
                continue
            gain_vec = U2[j, :] - U2[k, :]
            constraints.append(cp.sum(cp.multiply(mu[:, j], gain_vec)) >= 0)

    return constraints


def solve_cce_joint(
    U_res: np.ndarray,
    objective_type: str,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Solve for CCE joint distribution under different objectives. Returns mu."""
    n1, n2 = U_res.shape
    mu = cp.Variable((n1, n2), nonneg=True)
    constraints = build_cce_constraints(U_res, mu)

    if objective_type == "random":
        if rng is None:
            rng = np.random.default_rng()
        R = rng.normal(size=(n1, n2))
        obj = cp.Maximize(cp.sum(cp.multiply(mu, R)))
    elif objective_type == "entropy":
        obj = cp.Maximize(cp.sum(cp.entr(mu)))
    elif objective_type == "welfare":
        if n1 != n2:
            raise ValueError("Welfare objective requires a square payoff matrix.")
        W = U_res + U_res.T
        obj = cp.Maximize(cp.sum(cp.multiply(mu, W)))
    elif objective_type == "gini":
        # Minimize sum(mu^2), equivalent to Maximize (1 - sum(mu^2)) aka Gini impurity
        obj = cp.Minimize(cp.sum_squares(mu))
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.ECOS)
    except Exception:
        try:
            prob.solve(solver=cp.SCS)
        except Exception:
            pass

    if prob.status not in ["optimal", "optimal_inaccurate"] or mu.value is None:
        # Fallback to uniform independent (product) distribution
        p = np.ones(n1) / n1
        q = np.ones(n2) / n2
        return np.outer(p, q)

    mu_val = mu.value
    # Ensure it's a valid distribution (sometimes solvers leave slightly neg numbers or unnormalized)
    mu_val = np.maximum(mu_val, 0.0)
    total = mu_val.sum()
    if total > 1e-12:
        mu_val /= total
    else:
        # Fallback if somehow zero
        p = np.ones(n1) / n1
        q = np.ones(n2) / n2
        return np.outer(p, q)
        
    return mu_val


def solve_max_welfare_cce(U_res: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Finds a Coarse Correlated Equilibrium (CCE) of the restricted game that maximizes Total Welfare.
    Returns: (max_welfare, mu_matrix)
    """
    n1, n2 = U_res.shape
    mu = cp.Variable((n1, n2), nonneg=True)
    
    # Payoff matrices
    U1 = U_res
    U2 = U_res # For symmetric games, P2(i, j) = U_res[j, i]. So rows of U2 are rows of U_res.
    
    constraints = [cp.sum(mu) == 1]
    
    # CCE constraints for Player 1:
    # sum_j mu[i,j] * (u1(i,j) - u1(k,j)) >= 0 for all i, k
    for i in range(n1):
        for k in range(n1):
            if i == k: continue
            # Expectation of gain from staying at i vs switching to k, given recommendation i
            # Sum over j: mu[i, j] * (U1[i, j] - U1[k, j])
            gain_vec = U1[i, :] - U1[k, :]
            constraints.append(cp.sum(cp.multiply(mu[i, :], gain_vec)) >= 0)
            
    # CCE constraints for Player 2:
    # sum_i mu[i,j] * (u2(i,j) - u2(i,k)) >= 0 for all j, k
    # u2(i, j) = U2[j, i], so use rows of U2 for player-2 deviations
    for j in range(n2):
        for k in range(n2):
            if j == k: continue
            gain_vec = U2[j, :] - U2[k, :]
            constraints.append(cp.sum(cp.multiply(mu[:, j], gain_vec)) >= 0)
            
    # Objective: Maximize Welfare (sum of utilities)
    # Welfare = sum_ij mu[i,j] * (U1[i,j] + U2[i,j])
    # U2 is stored as U_res.T (shape n2 x n1), so use U2.T to align shapes.
    W = U1 + U2.T
    obj = cp.Maximize(cp.sum(cp.multiply(mu, W)))
    
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.ECOS) 
    except:
        try:
            prob.solve(solver=cp.SCS)
        except:
            pass # Fallback
        
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        # If solver fails, return default
        return 0.0, np.zeros((n1, n2))

    return prob.value, mu.value
