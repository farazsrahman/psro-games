# (J)PSRO in General-Sum Games
![PSRO Diagram](static/psro_diagram.png)

This repository implements Policy Space Response Oracles (PSRO) and related equilibrium solvers for game theoretic analysis, focusing on Coarse Correlated Equilibria (CCE) and Correlated Equilibria (CE) in general-sum games.

## Files Description

### Core Logic
- **`psro.py`**: The main entry point for running PSRO experiments on matrix games. Supports various meta-solvers (Nash, CCE, CE, Random) and tracks welfare/regret over epochs.
- **`solvers.py`**: Implements equilibrium solvers using `cvxpy`, including:
  - `solve_max_welfare_cce`: Max Welfare Coarse Correlated Equilibrium.
  - `solve_max_welfare_ce`: Max Welfare Correlated Equilibrium.
  - `solve_cce_joint`: CCE with various objectives (Entropy, Gini, etc.).
  - `fictitious_play_symmetric`: Standard fictitious play.
- **`games.py`**: Libraries for generating synthetic games, including:
  - Competitive-Cooperative interpolation ("Alpha" games).
  - Candogan-style Potential/Harmonic games.
  - Symmetric and Zero-sum mixtures.
- **`metrics.py`**: Utilities for calculating equilibrium metrics:
  - `calculate_ne_regret`: Nash Exact Regret.
  - `calculate_cce_regret`: CCE Regret.
  - `calculate_regret_cce` / `calculate_regret_ce`: Regret checks for solver verification.
- **`oracles.py`**: Best Response logic for PSRO:
  - `compute_best_responses`: Standard BR against mixed strategies.
  - `compute_rectified_best_responses`: Rectified BR for JPSRO/CCE expansion.

### Experiments & Analysis
- **`debug.py`**: A script performing a parameter sweep on the game correlation "alpha," plotting the relationship between alpha and Max Welfare CCE/CE.
- **`profile_solver_speed.py`**: logical benchmarking script to compare execution times of different solvers.
- **`psro_os.py`**: An OpenSpiel-compatible implementation of the PSRO loop, featuring `TabularPolicy` and `BestResponseOracle` wrappers.

### Testing
- **`test_solvers.py`**: Unit tests validating solver correctness against analytical games (PD, Battle of Sexes, Chicken) and verifying theoretical properties (e.g., Welfare(CCE) $\ge$ Welfare(CE)).
- **`test_psro.py`**: Tests for the PSRO experiment loop.
