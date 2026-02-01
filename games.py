import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict


@dataclass
class TwoPlayerGame:
    """2-player normal-form game, general-sum."""
    n_actions_p1: int
    n_actions_p2: int
    payoffs_p1: np.ndarray  # shape (m, n)
    payoffs_p2: np.ndarray  # shape (m, n)


@dataclass
class SymmetricGame:
    """2-player symmetric normal-form game: U2 = U1^T."""
    n_actions: int
    payoffs: np.ndarray  # shape (n, n), payoff for player 1

    @property
    def payoffs_p1(self) -> np.ndarray:
        return self.payoffs

    @property
    def payoffs_p2(self) -> np.ndarray:
        return self.payoffs.T


# ---------- Candogan-style: potential (+nonstrategic) constraints ----------

def build_potential_constraint_matrix(m: int, n: int) -> np.ndarray:
    """
    Build linear constraints A x = 0 that characterize 2p games which are
    exact potential + nonstrategic, in the sense of Monderer–Shapley and Candogan.

    x is the stacked vector [vec(U1); vec(U2)] of length 2*m*n.

    Constraint (for all i != i', j != j'):
        (u1[i,j] - u1[i',j]) - (u1[i,j'] - u1[i',j'])
      - (u2[i,j] - u2[i,j']) + (u2[i',j] - u2[i',j']) = 0

    This set of linear equalities describes the subspace of potential(+nonstrategic) games.
    """
    dim = 2 * m * n
    rows = []

    def idx1(i: int, j: int) -> int:
        return i * n + j  # player 1
    def idx2(i: int, j: int) -> int:
        return m * n + i * n + j  # player 2

    for i in range(m):
        for ip in range(i + 1, m):
            for j in range(n):
                for jp in range(j + 1, n):
                    row = np.zeros(dim)

                    # U1 terms
                    row[idx1(i,  j )] += 1.0
                    row[idx1(ip, j )] -= 1.0
                    row[idx1(i,  jp)] -= 1.0
                    row[idx1(ip, jp)] += 1.0

                    # U2 terms (note signs!)
                    row[idx2(i,  j )] -= 1.0
                    row[idx2(i,  jp)] += 1.0
                    row[idx2(ip, j )] += 1.0
                    row[idx2(ip, jp)] -= 1.0

                    rows.append(row)

    if rows:
        A = np.vstack(rows)
    else:
        A = np.zeros((0, dim))
    return A


def compute_nullspace_basis(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Compute an orthonormal basis for the nullspace of A using SVD.

    Returns N of shape (dim, d_null), whose columns form an orthonormal basis
    for { x : A x = 0 }.

    Projector onto this subspace is P = N N^T.
    """
    if A.size == 0:
        # No constraints → whole space is nullspace
        return np.eye(A.shape[1])

    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    rank = int((S > tol).sum())
    V = Vt.T
    N = V[:, rank:]  # dim x (dim - rank)
    return N


# ---------- Projections: potential(+ns) and "harmonic"/anti-potential ----------

class CandoganProjector2P:
    """
    Helper to project 2p general-sum games onto potential(+ns) subspace
    and its orthogonal complement, under the Frobenius inner product.
    """

    def __init__(self, m: int, n: int):
        self.m = m
        self.n = n
        self.dim = 2 * m * n
        A = build_potential_constraint_matrix(m, n)
        self.N = compute_nullspace_basis(A)  # dim x d_null

    def flatten_game(self, U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
        return np.concatenate([U1.ravel(), U2.ravel()])

    def unflatten_game(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m, n = self.m, self.n
        U1 = x[: m * n].reshape(m, n)
        U2 = x[m * n :].reshape(m, n)
        return U1, U2

    def project_potential(self, U1: np.ndarray, U2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Orthogonal projection (Frobenius) of (U1,U2) onto the
        potential(+nonstrategic) subspace: x_p = N N^T x.
        """
        x = self.flatten_game(U1, U2)
        x_p = self.N @ (self.N.T @ x)
        return self.unflatten_game(x_p)

    def project_anti_potential(self, U1: np.ndarray, U2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Orthogonal complement of potential(+ns): "anti-potential" component.
        This is the Candogan-style harmonic-ish part (up to ns details).
        """
        x = self.flatten_game(U1, U2)
        x_p = self.N @ (self.N.T @ x)
        x_h = x - x_p
        return self.unflatten_game(x_h)


# ---------- Isotropic sampler + λ-mixing ----------

def _normalize_fro(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm < eps:
        return x
    return x / norm


# ---------- Zero-sum samplers: transitive + cyclic ----------

def sample_transitive_component(n_actions: int, rng: np.random.Generator) -> np.ndarray:
    """
    Transitive (ranking) component: payoff[i,j] = skill[i] - skill[j].
    """
    skill = rng.normal(size=(n_actions,))
    T = skill[:, None] - skill[None, :]
    np.fill_diagonal(T, 0.0)
    return _normalize_fro(T)


def sample_cyclic_component(n_actions: int, rng: np.random.Generator) -> np.ndarray:
    """
    Cyclic component: random antisymmetric matrix.
    """
    A = rng.normal(size=(n_actions, n_actions))
    C = A - A.T
    np.fill_diagonal(C, 0.0)
    return _normalize_fro(C)


def sample_zero_sum_mixed_game(
    n_actions: int,
    lambda_cyclic: float,
    rng: Optional[np.random.Generator] = None,
) -> SymmetricGame:
    """
    Sample a 2-player zero-sum game as a mixture of transitive and cyclic components.

    lambda_cyclic in [0, 1]:
      0   -> purely transitive (ranking-like)
      1   -> purely cyclic (RPS-like)
    """
    if rng is None:
        rng = np.random.default_rng()

    lambda_cyclic = float(lambda_cyclic)
    assert 0.0 <= lambda_cyclic <= 1.0, "lambda_cyclic must be in [0, 1]"
    alpha = np.sqrt(1.0 - lambda_cyclic ** 2)
    beta = lambda_cyclic

    T = sample_transitive_component(n_actions, rng)
    C = sample_cyclic_component(n_actions, rng)
    M = alpha * T + beta * C  # payoff for player 1

    return SymmetricGame(
        n_actions=n_actions,
        payoffs=M,
    )


def sample_symmetric_mixed_game(
    n_actions: int,
    lambda_anti: float,
    rng: Optional[np.random.Generator] = None,
) -> SymmetricGame:
    """
    Sample a symmetric 2p game by mixing symmetric and antisymmetric components:

        G(λ) = sqrt(1 - λ^2) * S  +  λ * A

    where S is symmetric and A is antisymmetric. U2 = U1^T by construction.
    """
    if rng is None:
        rng = np.random.default_rng()

    lambda_anti = float(lambda_anti)
    assert 0.0 <= lambda_anti <= 1.0, "lambda_anti must be in [0, 1]"

    base = rng.normal(size=(n_actions, n_actions))
    S = 0.5 * (base + base.T)
    A = 0.5 * (base - base.T)

    S = _normalize_fro(S)
    A = _normalize_fro(A)

    alpha = np.sqrt(1.0 - lambda_anti ** 2)
    beta = lambda_anti
    M = alpha * S + beta * A

    return SymmetricGame(
        n_actions=n_actions,
        payoffs=M,
    )


def sample_isotropic_potential_harmonic_components(
    m: int,
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Sample a random 2p general-sum game and decompose it into
    potential(+ns) and anti-potential components in an *isotropic* way:

        1. x ~ N(0, I) in R^(2 m n)
        2. x_pot = P_pot x, x_har = x - x_pot
        3. normalize each component by Frobenius norm

    Returns dict with:
      "pot": (U1_pot, U2_pot)
      "har": (U1_har, U2_har)
    """
    if rng is None:
        rng = np.random.default_rng()

    proj = CandoganProjector2P(m, n)

    # Sample isotropic base game
    x = rng.normal(size=(2 * m * n,))

    # Project
    U1_pot, U2_pot = proj.unflatten_game(proj.N @ (proj.N.T @ x))
    U1_har, U2_har = proj.unflatten_game(x - proj.N @ (proj.N.T @ x))

    # Normalize each component by joint Frobenius norm over both players
    x_pot = proj.flatten_game(U1_pot, U2_pot)
    x_har = proj.flatten_game(U1_har, U2_har)

    x_pot = _normalize_fro(x_pot)
    x_har = _normalize_fro(x_har)

    U1_pot, U2_pot = proj.unflatten_game(x_pot)
    U1_har, U2_har = proj.unflatten_game(x_har)

    return {
        "pot": (U1_pot, U2_pot),
        "har": (U1_har, U2_har),
    }


def sample_mixed_candogan_game(
    m: int,
    n: int,
    lambda_har: float,
    rng: Optional[np.random.Generator] = None,
) -> TwoPlayerGame:
    """
    Sample a 2p general-sum game as an isotropic Candogan-style mixture:

        G(λ) = sqrt(1 - λ^2) * G_pot  +  λ * G_har,

    where G_pot lies in the potential(+ns) subspace
    and G_har lies in its orthogonal complement.

    lambda_har in [0, 1]:
      0   -> purely potential(+ns)-like
      1   -> purely anti-potential / harmonic-like
      mid -> mixed regime
    """
    if rng is None:
        rng = np.random.default_rng()

    lambda_har = float(lambda_har)
    assert 0.0 <= lambda_har <= 1.0, "lambda_har must be in [0, 1]"

    comps = sample_isotropic_potential_harmonic_components(m, n, rng=rng)
    U1_pot, U2_pot = comps["pot"]
    U1_har, U2_har = comps["har"]

    alpha = np.sqrt(1.0 - lambda_har ** 2)
    beta = lambda_har

    U1 = alpha * U1_pot + beta * U1_har
    U2 = alpha * U2_pot + beta * U2_har

    return TwoPlayerGame(
        n_actions_p1=m,
        n_actions_p2=n,
        payoffs_p1=U1,
        payoffs_p2=U2,
    )


def sample_uniform_symmetric_game(
    n_actions: int,
    rng: Optional[np.random.Generator] = None,
) -> SymmetricGame:
    """
    Sample a symmetric 2p game where each unique payoff pair (i, j)
    is drawn independently from Uniform(0, 1).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample standard uniform [0, 1]
    base = rng.uniform(0.0, 1.0, size=(n_actions, n_actions))
    
    # Symmetrize by taking upper triangle and mirroring it
    # triu includes diagonal
    U_upper = np.triu(base)
    # diag(U_upper) is included in U_upper and U_upper.T, so subtract once
    U = U_upper + U_upper.T - np.diag(np.diag(U_upper))
    
    return SymmetricGame(
        n_actions=n_actions,
        payoffs=U,
    )


def sample_t_student_symmetric_game(
    n_actions: int,
    df: float = 3.0,
    rng: Optional[np.random.Generator] = None,
) -> SymmetricGame:
    """
    Sample a symmetric 2p game where each unique payoff pair (i, j)
    is drawn independently from a Student's t-distribution with `df` degrees of freedom.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample standard t-distribution
    base = rng.standard_t(df, size=(n_actions, n_actions))
    
    # Symmetrize by taking upper triangle and mirroring it
    U_upper = np.triu(base)
    # diag(U_upper) is included in U_upper and U_upper.T, so subtract once
    U = U_upper + U_upper.T - np.diag(np.diag(U_upper))
    
    return SymmetricGame(
        n_actions=n_actions,
        payoffs=U,
    )


def sample_normal_symmetric_game(
    n_actions: int,
    rng: Optional[np.random.Generator] = None,
) -> SymmetricGame:
    """
    Sample a symmetric 2p game where each unique payoff pair (i, j)
    is drawn independently from a Standard Normal distribution (Gaussian).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample standard normal
    base = rng.standard_normal(size=(n_actions, n_actions))
    
    # Symmetrize by taking upper triangle and mirroring it
    U_upper = np.triu(base)
    # diag(U_upper) is included in U_upper and U_upper.T, so subtract once
    U = U_upper + U_upper.T - np.diag(np.diag(U_upper))
    
    return SymmetricGame(
        n_actions=n_actions,
        payoffs=U,
    )


def sample_competitive_cooperative_interpolation_game(
    n_actions: int,
    alpha: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> SymmetricGame:

    if rng is None:
        rng = np.random.default_rng()

    A_raw = rng.standard_normal(size=(n_actions, n_actions))
    A = 0.5 * (A_raw - A_raw.T)
    
    C_raw = rng.standard_normal(size=(n_actions, n_actions))
    C = 0.5 * (C_raw + C_raw.T)
    
    U = (1-alpha) * A + alpha * C
    
    return SymmetricGame(
        n_actions=n_actions,
        payoffs=U,
    )



def opt_out_rps(a: float) -> SymmetricGame:
    """
    Opt-out RPS game. A, B, C are standard RPS strategies. D is the opt-out strategy.
    
    Payoff matrix for P1:
         A   B   C   D
      A  0   1  -1   0
      B -1   0   1   0
      C  1  -1   0   0
      D  a   a   a  10
    """
    M = np.array([
        [0.0, 1.0, -1.0, 0.0],
        [-1.0, 0.0, 1.0, 0.0],
        [1.0, -1.0, 0.0, 0.0],
        [float(a), float(a), float(a), 10.0]
    ])
    
    return SymmetricGame(
        n_actions=4,
        payoffs=M,
    )


# ---------- Tiny sanity check ----------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    m, n = 3, 3

    # Sample one mixed game with λ = 0 (pure potential-ish) and λ = 1 (pure harmonic-ish)
    game_pot = sample_mixed_candogan_game(m, n, lambda_har=0.0, rng=rng)
    game_har = sample_mixed_candogan_game(m, n, lambda_har=1.0, rng=rng)

    print("Pure potential-ish U1:\n", game_pot.payoffs_p1)
    print("Pure harmonic-ish U1:\n", game_har.payoffs_p1)

    # Check that potential projection satisfies constraints ~ 0
    A = build_potential_constraint_matrix(m, n)
    proj = CandoganProjector2P(m, n)
    U1_p, U2_p = proj.project_potential(game_har.payoffs_p1, game_har.payoffs_p2)
    x_p = proj.flatten_game(U1_p, U2_p)
    print("||A x_potential|| (should be ~0):", np.linalg.norm(A @ x_p))
