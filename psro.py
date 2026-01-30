import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time
from typing import Tuple, Dict, List, Any, Optional

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

from games import SymmetricGame, sample_zero_sum_mixed_game, sample_symmetric_mixed_game, sample_uniform_symmetric_game, sample_t_student_symmetric_game, sample_normal_symmetric_game, sample_correlated_symmetric_game
from metrics import (
    calculate_ne_regret,
    calculate_cce_regret,

    extract_restricted_game,
    uniform_welfare,
)
from solvers import (
    solve_max_welfare_cce,
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


# ---------- Meta-solver: approximate NE via fictitious play ----------


# ---------- PSRO and PSRO_rN ----------

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
        "pop_size": []
    }

    epoch_iter = tqdm(
        range(n_epochs),
        desc=progress_desc,
        disable=not progress,
        leave=False,
    )
    for epoch in epoch_iter:
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
        elif mss_mode == "random_cce":
            sigma_res = solve_cce_joint(U_res, "random", rng=rng)
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

        # Metric 2: CCE Regret
        cce_reg = calculate_cce_regret(U, mu_res, pop1, pop2)
        history["cce_regret"].append(cce_reg)



        history["pop_size"].append(len(pop1))

        if rectified_br:
            # JPSRO(CE) Expansion: Rectified Best Responses
            # Pass restricted joint distribution (sigma_res) to compute conditional BRs
            new_br1_list, new_br2_list = compute_rectified_best_responses(
                U, sigma_res, pop1, pop2
            )
            for a in new_br1_list:
                if a not in pop1:
                    pop1.append(a)
            for b in new_br2_list:
                if b not in pop2:
                    pop2.append(b)
        else:
            # Standard PSRO/JPSRO(CCE) Expansion: Mean-based Best Response
            br1_payoffs, br2_payoffs, a_star, b_star = compute_best_responses(U, p_full, q_full)
            if a_star not in pop1:
                pop1.append(a_star)
            if b_star not in pop2:
                pop2.append(b_star)

    history["pop1"] = pop1.copy()
    history["pop2"] = pop2.copy()
    return history


# ---------- Experiment + plots ----------

def run_single_experiment(
    n_actions: int = 50,
    lam: float = 0.5,
    n_epochs: int = 50,
    fp_steps: int = 200,
    seed: int = 0,
    n_seeds: int = 10,
    n_bootstrap: int = 5000,
    ci: float = 0.95,
    use_zero_sum: bool = True,
    rectified_br: bool = False,
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
        
        out_path = os.path.join(output_dir, f"psro_empirical_game_seed_{seed_idx}.png")
        plt.savefig(out_path)
        plt.close(fig)
        print(f"Saved refined combined empirical game mask to {out_path}")

    if use_zero_sum:
        desc = f"Zero-sum (λ_cyclic={lam})"
    else:
        desc = f"General-sum (λ_anti={lam})"
    
    # Benchmarking override:
    desc = f"Correlated Symmetric (A + {lam}*C)"

    print(
        f"Running PSRO on {desc} with {n_actions} actions, {n_epochs} epochs "
        f"over {n_seeds} seeds..."
    )

    output_dir = "psro_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Use a SeedSequence so each run is reproducible but independent.
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_seeds)

    # Store results for each MSS mode
    results = {}
    mss_modes = [
        "nash",
        "uniform",
        "random_cce",
        # "max_entropy_cce",
        # "max_welfare_cce",
        "max_gini_cce",
    ]
    colors = {
        "nash": "blue",
        "uniform": "red",
        "random_cce": "orange",
        "random_cce": "orange",
        "max_entropy_cce": "purple",
        "max_welfare_cce": "brown",
        "max_gini_cce": "green",
    }
    labels = {
        "nash": "Nash MSS",
        "uniform": "Uniform MSS",
        "random_cce": "Random CCE",
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

        # 1. Generate Game and Baselines
        game = sample_correlated_symmetric_game(n_actions, alpha=lam, rng=rng_seed)
        
        # Full-game welfare-optimal CCE welfare (normalization target)
        with StepTimer(f"Seed {seed_idx}: Full Game Max Welfare CCE"):
            full_wel_max, mu_full = solve_max_welfare_cce(game.payoffs)
        denom = float(full_wel_max)
        
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

            # Store population for combined plot
            seed_populations[mss] = history["pop1"]

        # 3. Plot combined discovery for this seed
        _plot_combined_discovery(
            game.payoffs,
            seed_populations,
            seed_idx,
            output_dir,
            mu_full=mu_full,
            mss_labels=labels
        )

    # 4. Compute Bootstrap Stats for all MSS modes
    for mss in mss_modes:
        m = all_raw_metrics[mss]
        ne_arr = np.stack(m["ne_regrets"], axis=0)
        cce_arr = np.stack(m["cce_regrets"], axis=0)

        wel_arr = np.stack(m["cce_welfare_fracs"], axis=0)
        pop_arr = np.stack(m["pop_sizes"], axis=0)

        boot_rng = np.random.default_rng(ss.spawn(1)[0])
        ne_mean, ne_lo, ne_hi = _bootstrap_ci_mean(ne_arr, rng=boot_rng, n_bootstrap=n_bootstrap, ci=ci)
        cce_mean, cce_lo, cce_hi = _bootstrap_ci_mean(cce_arr, rng=boot_rng, n_bootstrap=n_bootstrap, ci=ci)

        wel_mean, wel_lo, wel_hi = _bootstrap_ci_mean(wel_arr, rng=boot_rng, n_bootstrap=n_bootstrap, ci=ci)
        pop_mean, pop_lo, pop_hi = _bootstrap_ci_mean(pop_arr, rng=boot_rng, n_bootstrap=n_bootstrap, ci=ci)
        
        results[mss] = {
            "ne": (ne_mean, ne_lo, ne_hi),
            "cce": (cce_mean, cce_lo, cce_hi),

            "wel": (wel_mean, wel_lo, wel_hi),
            "pop": (pop_mean, pop_lo, pop_hi),
            "cov": np.mean(m["welfare_coverages"]),
        }

    mean_uniform_regret = float(np.mean(uniform_baselines["regret"]))
    mean_uniform_welfare_frac = float(np.mean(uniform_baselines["welfare"]))

    def _fmt_ci(m: float, lo: float, hi: float) -> str:
        return f"{m:.4f} [{lo:.4f}, {hi:.4f}]"

    # Print summary for all MSS modes
    for mss in mss_modes:
        print(f"\n--- Final Metrics ({labels[mss]}) ---")
        r = results[mss]
        print(f"Final Population Size: {_fmt_ci(r['pop'][0][-1], r['pop'][1][-1], r['pop'][2][-1])}")
        print(f"Final NE Regret: {_fmt_ci(r['ne'][0][-1], r['ne'][1][-1], r['ne'][2][-1])}")
        print(f"Final CCE Regret: {_fmt_ci(r['cce'][0][-1], r['cce'][1][-1], r['cce'][2][-1])}")

        print(f"Final CCE Welfare Fraction: {_fmt_ci(r['wel'][0][-1], r['wel'][1][-1], r['wel'][2][-1])}")
        print(f"Welfare Support Coverage: {r['cov']*100:.1f}%")

    # Plot metrics
    epochs = np.arange(n_epochs)
    
    plt.figure(figsize=(24, 6))
    
    # Subplot 1: Population Size
    plt.subplot(1, 4, 1)
    for mss in mss_modes:
        mean, lo, hi = results[mss]["pop"]
        plt.plot(epochs, mean, label=f"Pop Size ({mss})", color=colors[mss])
        plt.fill_between(epochs, lo, hi, alpha=0.2, color=colors[mss])
    plt.xlabel("Epoch")
    plt.ylabel("Size")
    plt.title("Population Size")
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize='small')

    # Subplot 2: NE Regret
    plt.subplot(1, 4, 2)
    plt.axhline(y=mean_uniform_regret, color="black", linestyle="--", label="Uniform Regret")
    for mss in mss_modes:
        mean, lo, hi = results[mss]["ne"]
        plt.plot(epochs, mean, label=f"NE Regret ({mss})", color=colors[mss])
        plt.fill_between(epochs, lo, hi, alpha=0.2, color=colors[mss])
    plt.xlabel("Epoch")
    plt.ylabel("Regret")
    plt.title("NE Regret on Full Game")
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize='small')
    
    # Subplot 3: CCE Regret
    plt.subplot(1, 4, 3)
    plt.axhline(y=mean_uniform_regret, color="black", linestyle="--", label="Uniform Regret")
    for mss in mss_modes:
        mean, lo, hi = results[mss]["cce"]
        plt.plot(epochs, mean, label=f"CCE Regret ({mss})", color=colors[mss])
        plt.fill_between(epochs, lo, hi, alpha=0.2, color=colors[mss])
    plt.xlabel("Epoch")
    plt.ylabel("Regret")
    plt.title("CCE Regret on Full Game")
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize='small')
    
    # Subplot 4: Welfare Fraction
    plt.subplot(1, 4, 4)
    plt.axhline(y=mean_uniform_welfare_frac, color="black", linestyle="--", label="Uniform Welfare")
    for mss in mss_modes:
        mean, lo, hi = results[mss]["wel"]
        plt.plot(epochs, mean, label=f"Welfare ({mss})", color=colors[mss])
        plt.fill_between(epochs, lo, hi, alpha=0.2, color=colors[mss])
    plt.xlabel("Epoch")
    plt.ylabel("Fraction of full-game welfare-optimal CCE")
    plt.title("Empirical Max-Welfare CCE vs Full Game")
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize='small')
    
    plt.tight_layout(pad=2.0, h_pad=1.0, w_pad=1.0, rect=[0, 0.08, 1, 1])
    plt.savefig(os.path.join(output_dir, "psro_metrics.png"))
    plt.close()
    print(f"Saved plot to {os.path.join(output_dir, 'psro_metrics.png')}")


if __name__ == "__main__":
    run_single_experiment(use_zero_sum=USE_ZERO_SUM)
