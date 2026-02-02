import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time
import json
import re
from typing import Tuple, Dict, List, Any, Optional, Callable

try:
    # Nice progress bars in notebooks/terminals.
    from tqdm.auto import tqdm  # type: ignore
    
    def log_msg(msg: str):
        tqdm.write(msg)
except Exception:  # pragma: no cover
    # Fallback: behave like an identity wrapper if tqdm isn't installed.
    def tqdm(x, **kwargs):  # type: ignore
        return x
    
    def log_msg(msg: str):
        print(msg)

from games import (
    SymmetricGame,  
    sample_zero_sum_mixed_game, 
    sample_symmetric_mixed_game, 
    sample_uniform_symmetric_game, 
    sample_t_student_symmetric_game, 
    sample_normal_symmetric_game, 
    sample_competitive_cooperative_interpolation_game, 
    opt_out_rps
)
from metrics import (
    calculate_ne_regret,
    calculate_cce_regret,

    extract_restricted_game,
    uniform_welfare,
)
from solvers import (
    solve_max_welfare_cce,
    solve_max_welfare_ce,
    solve_cce_joint,
    fictitious_play_symmetric,
)
from oracles import (
    compute_best_responses,
    compute_rectified_best_responses,
)

class StepTimer:
    def __init__(self, name: str):
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.time()
        log_msg(f"[_START_] {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start
        log_msg(f"[_END_]   {self.name} took {duration:.4f}s")

# Set this flag to switch between cases (both are symmetric games):
# True  -> zero-sum (antisymmetric payoff matrix)
# False -> general-sum symmetric mixture (symmetric + antisymmetric)
USE_ZERO_SUM = False

# ---------- (J)PSRO ----------

def psro(
    game: SymmetricGame,
    n_epochs: int,
    fp_steps: int = 300,
    rng: np.random.Generator = None,
    progress: bool = False,
    progress_desc: str = "PSRO epochs",
    mss_mode: str = "nash",
    rectified_br: bool = False,
) -> Dict[str, Any]:
    """
    Vanilla PSRO for symmetric 2p games.
    Populations are pure actions (indices).
    mss_mode: "nash" (default) uses Fictitious Play to approx NE. "uniform" uses Uniform dist.
    Returns dict with history of metrics.
    """
    if rng is None:
        rng = np.random.default_rng()

    U = game.payoffs
    n, n2 = U.shape
    if n != n2:
        raise ValueError("Symmetric game must have a square payoff matrix.")

    # Initial populations (single arbitrary action per player)
    pop1 = [0]
    pop2 = [0]

    history = {
        "ne_regret": [],
        "cce_regret": [],

        "cce_welfare": [],
        "ce_welfare": [],
        "pop_size": [],
        "detailed": []
    }
    
    # --- Index 0: Initial State ---
    # Calc initial regret for singleton population
    # Since we assume symmetric, p_full = q_full = one-hot on action 0
    p_init = np.zeros(n)
    p_init[0] = 1.0
    q_init = np.zeros(n)
    q_init[0] = 1.0
    
    sigma_init = np.outer(p_init, q_init)
    
    ne_reg_init = calculate_ne_regret(U, p_init, q_init)
    
    # For CCE regret on singleton: empirical game is 1x1. CCE is that 1x1. 
    # Max welfare CCE is trivial.
    # We can just use the standard metric function, passing mu that is 1.0 on (0,0) restricted.
    mu_res_init = np.array([[1.0]])
    cce_reg_init = calculate_cce_regret(U, mu_res_init, rows=pop1, cols=pop2)
    
    # Welfare of singleton CCE (which is just the payoff [0,0])
    welfare_init = float(U[pop1[0], pop2[0]])
    
    # For fixed_direction_vertex_CCE:
    # Generate a fixed random matrix R of the full game size to maintain consistent "direction"
    # as the subspace expands.
    R_fixed_full = rng.normal(size=(n, n))

    history["detailed"].append({
        "population_before_BR": list(pop1),
        "population_after_BR": list(pop1),
        "ms_solution_joint": sigma_init,
        "ms_solution_marginal": p_init,
        "BR": None,
        "regret": {
            "NE": float(ne_reg_init),
            "CCE": float(cce_reg_init)
        },
        "max_welfare": welfare_init,
        "max_welfare_ce": welfare_init,  # CE = CCE for 1x1
        "population_size": len(pop1)
    })

    epoch_iter = tqdm(
        range(n_epochs),
        desc=progress_desc,
        disable=not progress,
        leave=False,
    )
    for epoch in epoch_iter:
        # Pre-expansion population
        pop_pre = list(pop1)

        # Ensure symmetric restricted game by using a unified population
        pop_union = sorted(set(pop1) | set(pop2))
        pop1 = pop_union
        pop2 = pop_union

        # Restricted game
        U_res = extract_restricted_game(U, pop1, pop2)

        if mss_mode == "nash":
            # 1. Solve NE (Fictitious Play)
            p_res, q_res = fictitious_play_symmetric(U_res, n_steps=fp_steps, rng=rng)
            sigma_res = np.outer(p_res, q_res)
        elif mss_mode == "uniform":
            # 1. Uniform over empirical game
            n_res = len(pop1)
            p_res = np.ones(n_res) / n_res
            q_res = np.ones(n_res) / n_res
            sigma_res = np.outer(p_res, q_res)
        elif mss_mode == "random_vertex_cce":
            sigma_res = solve_cce_joint(U_res, "random", rng=rng)
        elif mss_mode == "fixed_direction_vertex_cce":
            # Slice the fixed full R matrix to the current restricted game indices
            R_res = R_fixed_full[np.ix_(pop1, pop2)]
            sigma_res = solve_cce_joint(U_res, "linear", objective_matrix=R_res)
        elif mss_mode == "max_entropy_cce":
            sigma_res = solve_cce_joint(U_res, "entropy")
        elif mss_mode == "max_welfare_cce":
            sigma_res = solve_cce_joint(U_res, "welfare")
        elif mss_mode == "max_gini_cce":
            sigma_res = solve_cce_joint(U_res, "gini")
        else:
            raise ValueError(f"Unknown mss_mode: {mss_mode}")

        # Map restricted joint to full joint
        sigma_full = np.zeros((n, n))
        sigma_full[np.ix_(pop1, pop2)] = sigma_res

        # Extract marginals from full joint for metrics/BR
        p_full = np.sum(sigma_full, axis=1)
        q_full = np.sum(sigma_full, axis=0)

        # Metric 1: NE Regret
        ne_reg = calculate_ne_regret(U, p_full, q_full)
        history["ne_regret"].append(ne_reg)

        # 2. Solve Max Welfare CCE
        welfare, mu_res = solve_max_welfare_cce(U_res)
        history["cce_welfare"].append(welfare)

        # 3. Solve Max Welfare CE
        _, welfare_ce = solve_max_welfare_ce(U_res)
        history["ce_welfare"].append(welfare_ce)

        # Metric 2: CCE Regret
        cce_reg = calculate_cce_regret(U, mu_res, rows=pop1, cols=pop2)
        history["cce_regret"].append(cce_reg)

        history["pop_size"].append(len(pop1))

        if rectified_br:
            # JPSRO(CE) Expansion: Rectified Best Responses
            # Pass restricted joint distribution (sigma_res) to compute conditional BRs
            new_br1_list, new_br2_list = compute_rectified_best_responses(
                U, sigma_res, pop1, pop2
            )
            # Just take the first one for logging if multiple (usually specialized code handles lists)
            # For detailed log, we might want to store list or just first. 
            # Given the prompt asks for "BR", implied singular, we'll store best one found.
            br_log = new_br1_list[0] if new_br1_list else None
            
            for a in new_br1_list:
                if a not in pop1:
                    pop1.append(a)
            for b in new_br2_list:
                if b not in pop2:
                    pop2.append(b)
        else:
            # Standard PSRO/JPSRO(CCE) Expansion: Mean-based Best Response
            br1_payoffs, br2_payoffs, a_star, b_star = compute_best_responses(U, p_full, q_full)
            br_log = int(a_star)
            
            if a_star not in pop1:
                pop1.append(a_star)
            if b_star not in pop2:
                pop2.append(b_star)
        
        # Log details for THIS completed epoch
        history["detailed"].append({
            "population_before_BR": pop_pre,
            "population_after_BR": list(pop1), # Population AFTER expansion
            "ms_solution_joint": sigma_full.copy(), 
            "ms_solution_marginal": p_full.copy(),
            "BR": br_log,
            "regret": {
                "NE": float(ne_reg),
                "CCE": float(cce_reg)
            },
            "max_welfare": float(welfare),
            "max_welfare_ce": float(welfare_ce),
            "population_size": len(pop1)
        })

    history["pop1"] = pop1.copy()
    history["pop2"] = pop2.copy()
    return history


# ---------- Experiment + plots ----------

# ---------- Experiment + plots ----------

def plot_experiment_results(
    experiment_results: Dict[int, Dict[str, List[Dict]]],
    games: Dict[int, np.ndarray],
    output_dir: str = "psro_outputs",
    experiment_name: str = "experiment"
):
    """
    Generate plots from experiment results.
    """
    def _bootstrap_ci_mean(
        x: np.ndarray,
        *,
        rng: np.random.Generator,
        n_bootstrap: int,
        ci: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if x.ndim != 2:
            raise ValueError(f"Expected x.ndim==2, got {x.ndim}")
        n = x.shape[0]
        if n < 2:
            m = x.mean(axis=0)
            return m, m, m
        if n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be > 0")

        idx = rng.integers(0, n, size=(n_bootstrap, n))
        samples = x[idx]
        boot_means = samples.mean(axis=1)

        alpha = 1.0 - ci
        lo = np.quantile(boot_means, alpha / 2.0, axis=0)
        hi = np.quantile(boot_means, 1.0 - alpha / 2.0, axis=0)
        mean = x.mean(axis=0)
        return mean, lo, hi

    def _plot_combined_discovery_masked(
        U: np.ndarray,
        populations_by_mss: Dict[str, List[int]],
        seed_idx: int,
        output_dir: str,
        mu_full: Optional[np.ndarray] = None,
        mss_labels: Dict[str, str] = {},
    ) -> None:
        n = U.shape[0]
        wel_support = set()
        if mu_full is not None:
            marginals = np.sum(mu_full, axis=1)
            wel_support = set(np.where(marginals > 1e-4)[0])

        n_mss = len(populations_by_mss)
        
        matrix_size = 6
        index_col_width = 0.5
        fig_width = matrix_size + (n_mss + 1) * index_col_width + 2.0
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        
        im = ax.imshow(U, cmap="viridis", aspect="equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Payoff (Player 1)")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        trans = ax.get_yaxis_transform()
        col_w_axes = index_col_width / matrix_size

        for idx, (mss, pop) in enumerate(populations_by_mss.items()):
            label = mss_labels.get(mss, mss)
            order_by_row = {row: i + 1 for i, row in enumerate(pop)}
            
            x_pos = -0.05 - (n_mss - 1 - idx) * col_w_axes
            
            ax.text(
                x_pos, -1.0, label,
                transform=trans, color="black", fontsize=9,
                rotation=45, ha="left", va="bottom", fontweight="bold"
            )

            for i in range(n):
                if i in order_by_row:
                    ax.text(
                        x_pos, i, str(order_by_row[i]),
                        transform=trans, color="red", fontsize=10,
                        fontweight="bold", ha="center", va="center"
                    )

        star_x = -0.05 - n_mss * col_w_axes
        for i in range(n):
            if i in wel_support:
                ax.text(
                    star_x, i, "*",
                    transform=trans, color="blue", fontsize=20,
                    fontweight="bold", ha="center", va="center"
                )

        ax.set_title(f"PSRO Strategy Discovery (Seed {seed_idx})", pad=60, fontsize=14)
        
        left_space_inches = (n_mss + 1) * index_col_width + 0.5
        left_margin = left_space_inches / fig_width
        
        fig.subplots_adjust(left=left_margin, right=0.95, top=0.85, bottom=0.05)
        
        out_path = os.path.join(output_dir, f"{experiment_name}_seed_{seed_idx}.png")
        plt.savefig(out_path)
        plt.close(fig)
        print(f"Saved refined combined empirical game mask to {out_path}")

    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Aggregate metrics for plotting
    mss_modes = list(list(experiment_results.values())[0].keys())
    
    # Assuming all seeds have same epochs
    first_seed = list(experiment_results.keys())[0]
    n_epochs = len(experiment_results[first_seed][mss_modes[0]]) 
    epochs = np.arange(n_epochs)

    # We need to compute CCE Welfare Fraction w.r.t the specific game for each seed.
    # The 'max_welfare' stored in detailed history is the CURRENT cce welfare.
    # We need to normalize it by the game's max possible welfare.
    
    # Setup storage for aggregation
    agg_metrics = {
        mss: {
            "pop_sizes": [],
            "ne_regrets": [],
            "cce_regrets": [],
            "cce_welfare_fracs": [], # will store absolute CCE welfare now
            "ce_welfare": []
        }
        for mss in mss_modes
    }
    
    # Iterate seeds to populate agg_metrics
    for seed_idx, mss_data in experiment_results.items():
        U = games[seed_idx]["matrix"]
        denom = games[seed_idx]["mw"]
        
        # We need mu_full for the stars in discovery plot. 
        # Since we didn't store mu_full (only mw scalar), we must recompute it solely for the stars 
        # OR we can update experiment to store mu_full too.
        # But for now, let's keep recomputing mu_full just for the "star" locations if needed, 
        # BUT the prompt specifically said "store max_welfare ... since it is heavy".
        # Actually, solve_max_welfare_cce returns (welfare, mu).
        # If we only stored welfare, we can't plot stars without re-solving.
        # Since plotting stars is visual, maybe we re-solve quickly or just plot without?
        # The prompt implies avoiding heavy computation. 
        # Let's check how solve_max_welfare_cce is used.
        # It's used for denom (done) and mu_full (for stars).
        # Let's re-solve for mu_full but assume the user cares most about the metrics loop not re-solving.
        # Wait, if we re-solve for stars, we defeat the purpose for that part.
        # However, denom is used for every MSS normalization.
        # Let's just re-solve for mu_full once per seed here, as original code did.
        # BUT use the stored denom for consistency/speed if we trust it.
        
        # Actually, let's just re-solve for mu_full to keep stars working, 
        # but use stored denom. The solver is likely fast for small games.
        # If the user really wanted to avoid all solving, they'd ask to store mu_full.
        # Storing just "mw" implies they care about the metric normalization cost.
        
        # Re-calc mu_full for plot visual (stars)
        _, mu_full = solve_max_welfare_cce(U)
        
        seed_populations = {}
        
        for mss, steps in mss_data.items():
            # Extract time series
            pops = [step["population_size"] for step in steps]
            nes = [step["regret"]["NE"] for step in steps]
            cces = [step["regret"]["CCE"] for step in steps]
            wels = [step["max_welfare"] for step in steps]
            
            # Normalize welfare - REMOVED, using absolute
            # if abs(denom) < 1e-12:
            #     wel_fracs = [1.0 if abs(w) < 1e-9 else float('nan') for w in wels]
            # else:
            #     wel_fracs = [w / denom for w in wels]
                
            agg_metrics[mss]["pop_sizes"].append(pops)
            agg_metrics[mss]["ne_regrets"].append(nes)
            agg_metrics[mss]["cce_regrets"].append(cces)
            agg_metrics[mss]["cce_welfare_fracs"].append(wels) # Store RAW welfare now, not fraction
            
            # CE Welfare
            wels_ce = [step["max_welfare_ce"] for step in steps]
            agg_metrics[mss]["ce_welfare"].append(wels_ce)
            
            seed_populations[mss] = steps[-1]["population_after_BR"]
            
        # Plot discovery for this seed
        _plot_combined_discovery_masked(
            U,
            seed_populations,
            seed_idx,
            output_dir,
            mu_full=mu_full,
            mss_labels={m:m for m in mss_modes} # Simplified labels
        )

    # 2. Bootstrapping and plotting Aggregate Metrics
    colors = {
        "nash": "tab:blue",
        "uniform": "tab:gray",
        "random_vertex_cce": "tab:orange",
        "fixed_direction_vertex_cce": "tab:red",
        "max_entropy_cce": "tab:purple",
        "max_welfare_cce": "tab:brown",
        "max_gini_cce": "tab:green",
    }
    
    results_boot = {}
    ss_boot = np.random.SeedSequence(0)
    rng_boot = np.random.default_rng(ss_boot)
    
    for mss in mss_modes:
        m = agg_metrics[mss]
        # Stack to (n_seeds, n_epochs)
        pop_arr = np.array(m["pop_sizes"])
        ne_arr = np.array(m["ne_regrets"])
        cce_arr = np.array(m["cce_regrets"])
        wel_arr = np.array(m["cce_welfare_fracs"])
        
        # Bootstrap
        pop_stats = _bootstrap_ci_mean(pop_arr, rng=rng_boot, n_bootstrap=1000, ci=0.95)
        ne_stats = _bootstrap_ci_mean(ne_arr, rng=rng_boot, n_bootstrap=1000, ci=0.95)
        cce_stats = _bootstrap_ci_mean(cce_arr, rng=rng_boot, n_bootstrap=1000, ci=0.95)
        wel_stats = _bootstrap_ci_mean(wel_arr, rng=rng_boot, n_bootstrap=1000, ci=0.95) # Now absolute welfare
        
        wel_ce_arr = np.array(m["ce_welfare"])
        wel_ce_stats = _bootstrap_ci_mean(wel_ce_arr, rng=rng_boot, n_bootstrap=1000, ci=0.95)
        
        results_boot[mss] = {
            "pop": pop_stats,
            "ne": ne_stats,
            "cce": cce_stats,
            "wel": wel_stats,
            "wel_ce": wel_ce_stats
        }

    plt.figure(figsize=(30, 6)) # Width increased for 5 plots
    
    # Subplot 1: Population Size
    plt.subplot(1, 5, 1)
    for mss in mss_modes:
        mean, lo, hi = results_boot[mss]["pop"]
        col = colors.get(mss, "black")
        plt.plot(epochs, mean, label=f"{mss}", color=col)
        plt.fill_between(epochs, lo, hi, alpha=0.1, color=col)
    plt.xlabel("Epoch")
    plt.ylabel("Size")
    plt.title("Population Size")
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize='small')

    # Subplot 2: NE Regret
    plt.subplot(1, 5, 2)
    for mss in mss_modes:
        mean, lo, hi = results_boot[mss]["ne"]
        col = colors.get(mss, "black")
        plt.plot(epochs, mean, label=f"{mss}", color=col)
        plt.fill_between(epochs, lo, hi, alpha=0.1, color=col)
    plt.xlabel("Epoch")
    plt.ylabel("Regret")
    plt.title("NE Regret on Full Game")
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize='small')
    
    # Subplot 3: CCE Regret
    plt.subplot(1, 5, 3)
    for mss in mss_modes:
        mean, lo, hi = results_boot[mss]["cce"]
        col = colors.get(mss, "black")
        plt.plot(epochs, mean, label=f"{mss}", color=col)
        plt.fill_between(epochs, lo, hi, alpha=0.1, color=col)
    plt.xlabel("Epoch")
    plt.ylabel("Regret")
    plt.title("CCE Regret on Full Game")
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize='small')
    
    # Subplot 4: Welfare CCE (Absolute)
    plt.subplot(1, 5, 4)
    for mss in mss_modes:
        mean, lo, hi = results_boot[mss]["wel"]
        col = colors.get(mss, "black")
        plt.plot(epochs, mean, label=f"{mss}", color=col)
        plt.fill_between(epochs, lo, hi, alpha=0.1, color=col)
        
    # Plot Mean Full Game Max Welfare CCE
    full_wel_cce_mean = np.mean([g["mw"] for g in games.values()])
    plt.axhline(full_wel_cce_mean, color="black", linestyle="--", alpha=0.7, label="Full Game Optimal")

    plt.xlabel("Epoch")
    plt.ylabel("Welfare (Absolute)")
    plt.title("Empirical Max-Welfare CCE")
    # plt.ylim(0, 1) # Removed fixed 0-1 range for absolute scale
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize='small')

    # Subplot 5: CE Welfare (Absolute)
    plt.subplot(1, 4 + 1, 5) # Updated to 5 subplots
    for mss in mss_modes:
        mean, lo, hi = results_boot[mss]["wel_ce"]
        col = colors.get(mss, "black")
        plt.plot(epochs, mean, label=f"{mss}", color=col)
        plt.fill_between(epochs, lo, hi, alpha=0.1, color=col)
        
    # Plot Mean Full Game Max Welfare CE
    full_wel_ce_mean = np.mean([g["mw_ce"] for g in games.values()])
    plt.axhline(full_wel_ce_mean, color="black", linestyle="--", alpha=0.7, label="Full Game Optimal")

    plt.xlabel("Epoch")
    plt.ylabel("Welfare (Absolute)")
    plt.title("Empirical Max-Welfare CE")
    
    # Match ylims for comparison
    # Get axes for subplot 4 and 5
    ax4 = plt.gcf().axes[3]
    ax5 = plt.gcf().axes[4]
    
    # Get current limits
    y4_min, y4_max = ax4.get_ylim()
    y5_min, y5_max = ax5.get_ylim()
    
    # Compute combined limits
    y_min = min(y4_min, y5_min)
    y_max = max(y4_max, y5_max)
    
    # Set limits
    ax4.set_ylim(y_min, y_max)
    ax5.set_ylim(y_min, y_max)
    
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize='small')
    
    # Adjust layout for 5 subplots
    plt.gcf().set_size_inches(30, 6) # Increase width
    plt.tight_layout(pad=2.0, h_pad=1.0, w_pad=1.0, rect=[0, 0.08, 1, 1])
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_metrics.png"))
    plt.close()
    print(f"Saved plot to {os.path.join(output_dir, f'{experiment_name}_metrics.png')}")


def run_single_experiment(
    game_sampler: Callable[[np.random.Generator], SymmetricGame],
    experiment_name: str = "experiment",
    n_epochs: int = 50,
    fp_steps: int = 200,
    seed: int = 0,
    n_seeds: int = 10,
    n_bootstrap: int = 5000,
    ci: float = 0.95,
    rectified_br: bool = False,
    output_dir: str = "psro_outputs"
):
    def _bootstrap_ci_mean(
        x: np.ndarray,  # shape: (n_seeds, n_epochs)
        *,
        rng: np.random.Generator,
        n_bootstrap: int,
        ci: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mean, lo, hi) over axis=0 using bootstrap percentile CI."""
        if x.ndim != 2:
            raise ValueError(f"Expected x.ndim==2, got {x.ndim}")
        n = x.shape[0]
        if n < 2:
            m = x.mean(axis=0)
            return m, m, m
        if not (0.0 < ci < 1.0):
            raise ValueError("ci must be in (0, 1)")
        if n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be > 0")

        # Resample seeds with replacement and compute mean curve each time.
        idx = rng.integers(0, n, size=(n_bootstrap, n))
        samples = x[idx]  # (B, n, T)
        boot_means = samples.mean(axis=1)  # (B, T)

        alpha = 1.0 - ci
        lo = np.quantile(boot_means, alpha / 2.0, axis=0)
        hi = np.quantile(boot_means, 1.0 - alpha / 2.0, axis=0)
        mean = x.mean(axis=0)
        return mean, lo, hi

    def _plot_combined_discovery(
        U: np.ndarray,
        populations_by_mss: Dict[str, List[int]],
        seed_idx: int,
        output_dir: str,
        mu_full: Optional[np.ndarray] = None,
        mss_labels: Dict[str, str] = {},
    ) -> None:
        """
        Plots the payoff matrix with multiple columns on the left indicating
        the order in which strategies were added by different meta-solvers.
        """
        n = U.shape[0]
        # Identify welfare-optimal support (rows)
        wel_support = set()
        if mu_full is not None:
            marginals = np.sum(mu_full, axis=1)
            wel_support = set(np.where(marginals > 1e-4)[0])

        n_mss = len(populations_by_mss)
        
        # Dimensions in inches
        matrix_size = 6
        index_col_width = 0.5
        # Total width: left margin space + matrix + colorbar space
        # Calculate dynamic width to keep everything centered and reachable
        fig_width = matrix_size + (n_mss + 1) * index_col_width + 2.0
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        
        im = ax.imshow(U, cmap="viridis", aspect="equal") # aspect=equal for square matrix
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Payoff (Player 1)")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Use yaxis_transform: X is in axis coordinates (0-1), Y is in data coordinates (row indices)
        trans = ax.get_yaxis_transform()
        
        # Column width in axis-relative units (assuming matrix is 1.0 wide)
        col_w_axes = index_col_width / matrix_size

        # Plot discovery indices for each MSS
        # We'll plot them from left to right, ending just before the matrix
        for idx, (mss, pop) in enumerate(populations_by_mss.items()):
            label = mss_labels.get(mss, mss)
            order_by_row = {row: i + 1 for i, row in enumerate(pop)}
            
            # X-offset in axis coordinates (- values are to the left of the matrix)
            x_pos = -0.05 - (n_mss - 1 - idx) * col_w_axes
            
            # Add column header (rotated)
            ax.text(
                x_pos,
                -1.0, # Just above first row (row 0 is at y=0)
                label,
                transform=trans,
                color="black",
                fontsize=9,
                rotation=45,
                ha="left",
                va="bottom",
                fontweight="bold"
            )

            for i in range(n):
                if i in order_by_row:
                    ax.text(
                        x_pos,
                        i,
                        str(order_by_row[i]),
                        transform=trans,
                        color="red",
                        fontsize=10,
                        fontweight="bold",
                        ha="center",
                        va="center"
                    )

        # Plot welfare stars (further left of all columns)
        star_x = -0.05 - n_mss * col_w_axes
        for i in range(n):
            if i in wel_support:
                ax.text(
                    star_x,
                    i,
                    "*",
                    transform=trans,
                    color="blue",
                    fontsize=20,
                    fontweight="bold",
                    ha="center",
                    va="center"
                )

        ax.set_title(f"PSRO Strategy Discovery (Seed {seed_idx})", pad=60, fontsize=14)
        
        # Calculate left margin to accommodate indices
        # Space needed on left = (n_mss + 1) * index_col_width + small padding
        left_space_inches = (n_mss + 1) * index_col_width + 0.5
        left_margin = left_space_inches / fig_width
        
        fig.subplots_adjust(left=left_margin, right=0.95, top=0.85, bottom=0.05)
        
        out_path = os.path.join(output_dir, f"{experiment_name}_seed_{seed_idx}.png")
        plt.savefig(out_path)
        plt.close(fig)
        print(f"Saved refined combined empirical game mask to {out_path}")

    print(
        f"Running PSRO Experiment '{experiment_name}' with {n_epochs} epochs "
        f"over {n_seeds} seeds..."
    )

    os.makedirs(output_dir, exist_ok=True)

    # Use a SeedSequence so each run is reproducible but independent.
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_seeds)

    # Store results for each MSS mode
    results = {}
    mss_modes = [
        "nash",
        "uniform",
        "nash",
        "uniform",
        "random_vertex_cce",
        "fixed_direction_vertex_cce",
        # "max_entropy_cce",
        # "max_welfare_cce",
        "max_gini_cce",
    ]
    colors = {
        "nash": "blue",
        "uniform": "red",
        "nash": "blue",
        "uniform": "red",
        "random_vertex_cce": "orange",
        "fixed_direction_vertex_cce": "darkorange",
        "max_entropy_cce": "purple",
        "max_welfare_cce": "brown",
        "max_gini_cce": "green",
    }
    labels = {
        "nash": "Nash MSS",
        "uniform": "Uniform MSS",
        "nash": "Nash MSS",
        "uniform": "Uniform MSS",
        "random_vertex_cce": "Random Vertex CCE",
        "fixed_direction_vertex_cce": "Fixed Vertex CCE",
        "random_ce": "Random CE",
        "max_entropy_cce": "Max-Ent CCE",
        "max_welfare_cce": "Max-Wel CCE",
        "max_gini_cce": "Max-Gini CCE",
    }


    # Store baselines and raw curves for all MSS modes
    uniform_baselines = {"regret": [], "welfare": []}
    all_raw_metrics = {
        mss: {
            "ne_regrets": [],
            "cce_regrets": [],
            "cce_regrets": [],
            "cce_welfare_fracs": [],
            "pop_sizes": [],
            "welfare_coverages": [],
        }
        for mss in mss_modes
    }
    
    # Store detailed history for return
    # Structure: experiment_results[seed_idx][mss] = List[Dict]
    experiment_results = {}
    games = {}

    # Main seed loop
    seed_iter = tqdm(
        range(n_seeds),
        desc="Overall Seeds",
        disable=False,
        leave=True,
    )

    for seed_idx in seed_iter:
        sseq = child_seeds[seed_idx]
        rng_seed = np.random.default_rng(sseq)
        
        experiment_results[seed_idx] = {}

        # 1. Generate Game and Baselines
        game = game_sampler(rng_seed)
        n_actions = game.n_actions
        
        # Full-game welfare-optimal CCE welfare (normalization target)
        with StepTimer(f"Seed {seed_idx}: Full Game Max Welfare CCE"):
            full_wel_max, mu_full = solve_max_welfare_cce(game.payoffs)
        denom = float(full_wel_max)
        
        games[seed_idx] = {
            "matrix": game.payoffs,
            "mw": denom,
            "mw_ce": 0.0 # Placeholder, computed next
        }
        
        # Full-game welfare-optimal CE
        with StepTimer(f"Seed {seed_idx}: Full Game Max Welfare CE"):
            _, full_wel_ce = solve_max_welfare_ce(game.payoffs)
        games[seed_idx]["mw_ce"] = float(full_wel_ce)

        
        # Baselines
        uni_pol = np.ones(n_actions) / n_actions
        uni_reg = calculate_ne_regret(game.payoffs, uni_pol, uni_pol)
        uni_welfare = uniform_welfare(game.payoffs)
        uni_wel_frac = uni_welfare / denom if abs(denom) > 1e-12 else 1.0
        uniform_baselines["regret"].append(uni_reg)
        uniform_baselines["welfare"].append(uni_wel_frac)

        seed_populations = {}

        # 2. Run all MSS modes for this seed
        for mss in mss_modes:
            # Re-spawn rng for the MSS to ensure it's comparable if needed
            # (though each MSS usually gets its own child seed in original code)
            # To match original behavior closely, we spawn a specific sequence for each MSS
            mss_ss = sseq.spawn(1)[0]
            rng_mss = np.random.default_rng(mss_ss)

            real_mss = mss
            use_rectified = False


            history = psro(
                game,
                n_epochs=n_epochs,
                fp_steps=fp_steps,
                rng=rng_mss,
                progress=False, # Disable nested progress bar
                mss_mode=real_mss,
                rectified_br=use_rectified,
            )

            # Welfare fraction calculation
            welfare_curve = np.asarray(history["cce_welfare"], dtype=float)
            if abs(denom) < 1e-12:
                if np.all(np.isfinite(welfare_curve)) and np.allclose(welfare_curve, 0.0, atol=1e-9):
                    welfare_frac = np.ones_like(welfare_curve, dtype=float)
                else:
                    welfare_frac = np.full_like(welfare_curve, np.nan, dtype=float)
            else:
                welfare_frac = welfare_curve / denom

            # Welfare support coverage
            marginals = np.sum(mu_full, axis=1)
            wel_support = set(np.where(marginals > 1e-4)[0])
            final_pop = set(history["pop1"])
            is_covered = float(wel_support.issubset(final_pop))

            # Store raw metrics
            m = all_raw_metrics[mss]
            m["ne_regrets"].append(np.asarray(history["ne_regret"], dtype=float))
            m["cce_regrets"].append(np.asarray(history["cce_regret"], dtype=float))

            m["cce_welfare_fracs"].append(welfare_frac)
            m["pop_sizes"].append(np.asarray(history["pop_size"], dtype=float))
            m["welfare_coverages"].append(is_covered)
            
            # Store detailed history
            experiment_results[seed_idx][mss] = history["detailed"]

            # Store population for combined plot
            seed_populations[mss] = history["pop1"]


    return experiment_results, games


if __name__ == "__main__":
    
    # # Define your game sampler here
    # # Example 1: Opt-Out RPS with a=2/3
    # def opt_out_sampler(rng: np.random.Generator) -> SymmetricGame:
    #     return opt_out_rps(a=2/3)

    # Example 2: Competitive Cooperative Interpolation Game (uncomment to use)
    def correlated_sampler(rng: np.random.Generator) -> SymmetricGame:
        return sample_competitive_cooperative_interpolation_game(n_actions=25, alpha=0.75, rng=rng)

    results, games = run_single_experiment(
        game_sampler=correlated_sampler,
        experiment_name="cc_interpolation_0_5",
        n_epochs=30, 
        n_seeds=10
    )
    
    # Plot results
    plot_experiment_results(
        results,
        games,
        experiment_name="cc_interpolation_0_5"
    )
    
    # # Verification of returned history structure
    # print("\n--- Detailed History Verification ---")
    
    # def np_encoder(obj):
    #     if isinstance(obj, np.ndarray):
    #         return np.round(obj, 4).tolist()
    #     return str(obj)

    # def pretty_print_json(obj):
    #     s = json.dumps(obj, indent=2, default=np_encoder)
    #     # Collapse primitive numeric arrays (e.g. [1.0, 0.0] or [[1.0, 0.0], ...])
    #     # This regex matches arrays of numbers including scientific notation
    #     return re.sub(
    #         r'\[\s*([0-9.\-eE,\s]+?)\s*\]',
    #         lambda m: '[' + ' '.join(m.group(1).replace(',', '').split()) + ']',
    #         s
    #     )

    # seed_0_mgcce = results[0]["max_gini_cce"]
    # print(f"Seed 0, Max-Gini CCE MSS, Step 0:")
    # print(pretty_print_json(seed_0_mgcce[0]))
    # print(f"Seed 0, Max-Gini CCE MSS, Step 1:")
    # print(pretty_print_json(seed_0_mgcce[1]))
    # print(f"Seed 0, Max-Gini CCE MSS, Step 2:")
    # print(pretty_print_json(seed_0_mgcce[2]))
    # print(f"Seed 0, Max-Gini CCE MSS, Step 3:")
    # print(pretty_print_json(seed_0_mgcce[3]))
