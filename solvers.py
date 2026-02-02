
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
    U2: Optional[np.ndarray] = None,
) -> List:
    """
    Construct CCE feasibility constraints for a game.
    Costraint: E[u_i(mu)] >= E[u_i(dev_k, mu_{-i})] for all deviations k.
    This scales as O(n_actions) constraints, unlike CE which is O(n^2).
    """
    n1, n2 = U_res.shape
    constraints = [cp.sum(mu) == 1, mu >= 0]
    
    # Payoffs
    U1 = U_res
    if U2 is None:
        # Assume symmetric if not provided
        U2 = U_res.T
    
    # --- Player 1 CCE Constraints ---
    # Expected utility from equilibrium distribution
    # E[u1] = sum_{i,j} mu_{ij} U1_{ij}
    exp_u1 = cp.sum(cp.multiply(mu, U1))
    
    # Marginal distribution of Player 2: prob that P2 plays j is sum_i mu_{ij}
    # q[j] = sum_i mu_{ij}
    q = cp.sum(mu, axis=0) # shape (n2,)
    
    # Deviation payoffs for P1: if P1 switches to action k, payoff is sum_j q[j] * U1[k, j]
    # Vectorized: U1 @ q -> shape (n1,)
    dev_payoffs_p1 = U1 @ q
    
    # Constraint: Expected utility >= Deviation utility for all k
    constraints.append(exp_u1 >= dev_payoffs_p1)
    
    # --- Player 2 CCE Constraints ---
    # Expected utility from equilibrium distribution
    # E[u2] = sum_{i,j} mu_{ij} U2_{ij}
    exp_u2 = cp.sum(cp.multiply(mu, U2))
    
    # Marginal distribution of Player 1: prob that P1 plays i is sum_j mu_{ij}
    # p[i] = sum_j mu_{ij}
    p = cp.sum(mu, axis=1) # shape (n1,)
    
    # Deviation payoffs for P2: if P2 switches to action l, payoff is sum_i p[i] * U2[i, l]
    # Note: U2 is (n1, n2). p is (n1,). We want result (n2,)
    # p @ U2 -> shape (n2,)
    dev_payoffs_p2 = p @ U2
    
    # Constraint: Expected utility >= Deviation utility for all l
    constraints.append(exp_u2 >= dev_payoffs_p2)
    
    return constraints


def solve_cce_joint(
    U_res: np.ndarray,
    objective_type: str,
    rng: Optional[np.random.Generator] = None,
    objective_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve for CCE joint distribution under different objectives. Returns mu."""
    n1, n2 = U_res.shape
    mu = cp.Variable((n1, n2), nonneg=True)
    
    # Symmetric game assumption in this file essentially
    U2 = U_res.T
    constraints = build_cce_constraints(U_res, mu, U2=U2)
    
    if objective_type == "random":
        if rng is None:
            rng = np.random.default_rng()
        R = rng.normal(size=(n1, n2))
        obj = cp.Maximize(cp.sum(cp.multiply(mu, R)))
    elif objective_type == "linear":
        if objective_matrix is None:
            raise ValueError("Must provide objective_matrix for 'linear' objective_type")
        if objective_matrix.shape != (n1, n2):
            raise ValueError(f"objective_matrix shape {objective_matrix.shape} must match U_res {(n1, n2)}")
        obj = cp.Maximize(cp.sum(cp.multiply(mu, objective_matrix)))
    elif objective_type == "entropy":
        obj = cp.Maximize(cp.sum(cp.entr(mu)))
    elif objective_type == "welfare":
        W = U_res + U2
        obj = cp.Maximize(cp.sum(cp.multiply(mu, W)))
    elif objective_type == "gini":
        # Minimize sum(mu^2), equivalent to Maximize (1 - sum(mu^2)) aka Gini impurity
        obj = cp.Minimize(cp.sum_squares(mu))
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")

    prob = cp.Problem(obj, constraints)
    
    # Try solvers in order of preference for speed/reliability
    # CLARABEL/SCS/ECOS are common. HiGHS if installed.
    solvers = []
    if hasattr(cp, 'CLARABEL'): solvers.append(cp.CLARABEL)
    solvers.extend([cp.ECOS, cp.SCS])
    
    for solver in solvers:
        try:
            prob.solve(solver=solver)
            if prob.status in ["optimal", "optimal_inaccurate"] and mu.value is not None:
                break
        except Exception:
            continue

    if prob.status not in ["optimal", "optimal_inaccurate"] or mu.value is None:
        # Fallback to uniform independent (product) distribution
        p = np.ones(n1) / n1
        q = np.ones(n2) / n2
        return np.outer(p, q)

    mu_val = mu.value
    mu_val = np.maximum(mu_val, 0.0)
    total = mu_val.sum()
    if total > 1e-12:
        mu_val /= total
    else:
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
    
    # Symmetric game assumption
    U2 = U_res.T
    constraints = build_cce_constraints(U_res, mu, U2=U2)
            
    # Objective: Maximize Welfare (sum of utilities)
    # Welfare = sum_ij mu[i,j] * (U1[i,j] + U2[i,j])
    W = U_res + U2
    obj = cp.Maximize(cp.sum(cp.multiply(mu, W)))
    
    prob = cp.Problem(obj, constraints)
    
    solvers = []
    if hasattr(cp, 'CLARABEL'): solvers.append(cp.CLARABEL)
    solvers.extend([cp.ECOS, cp.SCS])
    
    for solver in solvers:
        try:
            prob.solve(solver=solver)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                break
        except Exception:
            continue
        
    if prob.status not in ["optimal", "optimal_inaccurate"] or mu.value is None:
        return 0.0, np.zeros((n1, n2))

    mu_val = mu.value
    # sanitize
    mu_val = np.maximum(mu_val, 0.0)
    total = mu_val.sum()
    if total > 1e-12:
        mu_val /= total
    
    # Recalculate welfare with sanitized mu
    welfare = np.sum(mu_val * W)
    return welfare, mu_val


def solve_max_welfare_ce(U: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Finds a Correlated Equilibrium (CE) that maximizes Total Welfare.
    Returns: (mu_matrix, max_welfare)
    """
    n = U.shape[0]
    P = cp.Variable((n, n), nonneg=True)
    
    # Symmetric assumption for welfare calc U + U.T
    U2 = U.T
    
    # Welfare Objective 
    # Maximize sum(P * (U + U2))
    W = U + U2
    objective = cp.Maximize(cp.sum(cp.multiply(P, W)))
    
    constraints = [cp.sum(P) == 1]

    # CE constraints: 
    # For every recommended action 'i', player 1 should not prefer 'k'.
    # sum_j P[i,j] * U[i,j] >= sum_j P[i,j] * U[k,j]  for all i, k
    # 
    # And symmetrically for player 2.
    
    # Vectorized constraints
    # For P1:
    # E[u | rec=i] * P(rec=i) = sum_j P[i,j]*U[i,j]
    # E[u(switch k) | rec=i] * P(rec=i) = sum_j P[i,j]*U[k,j]
    # So we need sum_j P[i,j]*(U[i,j] - U[k,j]) >= 0
    
    # We can add these constraints for all pairs (i, k)
    # Ideally O(N^2) constraints
    
    # Let's perform the loop explicitly for clarity as N is small (<= 50)
    
    # P1
    for i in range(n): # recommended
        for k in range(n): # deviation
            if i == k: continue
            constraints.append(cp.sum(cp.multiply(P[i, :], U[i, :] - U[k, :])) >= 0)
            
    # P2
    for j in range(n): # recommended
        for l in range(n): # deviation
            if j == l: continue
            constraints.append(cp.sum(cp.multiply(P[:, j], U2[:, j] - U2[:, l])) >= 0)

    prob = cp.Problem(objective, constraints)
    
    # Try different solvers
    solvers = []
    if hasattr(cp, 'CLARABEL'): solvers.append(cp.CLARABEL)
    solvers.extend([cp.ECOS, cp.SCS])

    for solver in solvers:
        try:
            prob.solve(solver=solver)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                break
        except Exception:
            continue
            
    if prob.status not in ["optimal", "optimal_inaccurate"] or P.value is None:
         # Fallback
        return np.ones((n,n))/n**2, 0.0
  
    P_val = P.value
    P_val = np.maximum(P_val, 0.0)
    total = P_val.sum()
    if total > 1e-12:
        P_val /= total
        
    welfare = np.sum(P_val * W)
    return P_val, welfare




if __name__ == "__main__":
    import time
    from games import sample_competitive_cooperative_interpolation_game

    rng = np.random.default_rng(42)
    
    for n_actions in [10, 25, 50]:
        print(f"\nBenchmarking solvers on sample_competitive_cooperative_interpolation_game with {n_actions} actions... ")
        
        # Generate game: A + 0.5*C
        game = sample_competitive_cooperative_interpolation_game(n_actions=n_actions, alpha=0.5, rng=rng)
        U = game.payoffs_p1

        solvers_to_test = [
            ("Fictitious Play", lambda: fictitious_play_symmetric(U, n_steps=500, rng=rng)),
            ("CCE (Random)", lambda: solve_cce_joint(U, "random", rng=rng)),
            ("CCE (Entropy)", lambda: solve_cce_joint(U, "entropy", rng=rng)),
            ("CCE (Welfare)", lambda: solve_cce_joint(U, "welfare", rng=rng)),
            ("CCE (Gini)", lambda: solve_cce_joint(U, "gini", rng=rng)),
            ("Max Welfare CCE", lambda: solve_max_welfare_cce(U)),
            ("Max Welfare CE", lambda: solve_max_welfare_ce(U)),
        ]

        for name, solver_fn in solvers_to_test:
            start = time.time()
            _ = solver_fn()
            elapsed = time.time() - start
            print(f"  {name}: {elapsed:.4f}s")

