import numpy as np
import pandas as pd
import os
import argparse
from scipy.integrate import solve_ivp

K_A0 = 0.38
K_B0 = 0.14
BETA = 1.3
R_A = 0.56
R_B = 0.01
ALPHA = 0.01

# Ten steady-state ratios. There are more cell groups than targets, so several
# groups settle near the same target.
DEFAULT_RATIO_MIN = 0.5
DEFAULT_RATIO_MAX = 6.0
DEFAULT_N_TARGETS = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Give each cell group a stable ratio near one of ten targets."
    )
    parser.add_argument("--num_ids", type=int, default=40,
                        help="Number of cell groups. Must be greater than the number of target ratios.")
    parser.add_argument("--n_targets", type=int, default=DEFAULT_N_TARGETS,
                        help="Number of distinct stable-ratio targets when --target_ratios is omitted.")
    parser.add_argument("--t_end", type=float, default=24.0,
                        help="Endpoint in hours. Matches steady_state_fraction in the modeling notebook.")
    parser.add_argument("--n_pts", type=int, default=10,
                        help="Time points saved from the start through t_end, including the 24 h ratio.")
    parser.add_argument("--ratio_min", type=float, default=DEFAULT_RATIO_MIN,
                        help="Lowest stable-ratio target on the default log-spaced grid.")
    parser.add_argument("--ratio_max", type=float, default=DEFAULT_RATIO_MAX,
                        help="Highest stable-ratio target on the default log-spaced grid.")
    parser.add_argument("--target_ratios", type=str, default="",
                        help="Comma-separated stable-ratio targets. Overrides --n_targets, --ratio_min, and --ratio_max.")
    parser.add_argument("--tol_rel", type=float, default=0.05,
                        help="Largest relative error allowed between the final ODE ratio and its target before noise is added.")
    parser.add_argument("--noise_sigma", type=float, default=0.02,
                        help="Std dev of Gaussian noise added to ratios.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for parameter draws and observation noise.")
    parser.add_argument("--ratio_file", type=str, default="ratio_data.csv",
                        help="Output filename for ratio data.")
    parser.add_argument("--param_file", type=str, default="param_data.csv",
                        help="Output filename for parameter data.")
    parser.add_argument("--target_file", type=str, default="target_ratios.csv",
                        help="Output filename listing the stable-ratio targets.")
    parser.add_argument("-s", "--save_dir", type=str, default="../Data/Original",
                        help="Save path of data.")
    return parser


def arm_rates(h_A, h_B, k_A0=K_A0, k_B0=K_B0, beta=BETA):
    """Power-law arm-length to cleavage rate mapping."""
    k_A = k_A0 * (h_A ** (-beta))
    k_B = k_B0 * (h_B ** (-beta))
    return k_A, k_B


def bxb1_ode(t, y, k_A, k_B, r_A=R_A, r_B=R_B, alpha=ALPHA):
    """Right-hand side of the five-state BxB1 branching circuit.

    States are D (founder), L_A (Venus+), L_B (mScarlet+), M_A, and M_B.
    """
    D, L_A, L_B, M_A, M_B = y
    v_A = alpha * k_A
    v_B = alpha * k_B

    dD = -(k_A + k_B + r_A + r_B) * D
    dL_A = k_A * D + v_A * M_A + k_A * M_B
    dL_B = k_B * D + v_B * M_B + k_B * M_A
    dM_A = r_A * D - v_A * M_A - k_B * M_A
    dM_B = r_B * D - v_B * M_B - k_A * M_B
    return [dD, dL_A, dL_B, dM_A, dM_B]


def target_ratio_grid(n_targets, ratio_min, ratio_max, target_ratios):
    """Return the stable-ratio targets, sorted and strictly positive."""
    if target_ratios.strip():
        values = np.array([float(part) for part in target_ratios.split(",")], dtype=float)
    else:
        if n_targets < 1:
            raise ValueError("--n_targets must be at least 1.")
        if ratio_min <= 0 or ratio_max <= ratio_min:
            raise ValueError("--ratio_min and --ratio_max must satisfy 0 < ratio_min < ratio_max.")
        values = np.geomspace(ratio_min, ratio_max, n_targets)
    if values.size == 0 or np.any(values <= 0):
        raise ValueError("Target ratios must be positive.")
    return np.sort(values.astype(float))


def ratio_at_time(h_A, h_B, kinetics, t_end):
    """L_A/L_B at ``t_end`` hours, using the same ODE as the modeling notebook."""
    values = simulate_ratio(np.array([t_end]), h_A, h_B, kinetics)
    if values is None:
        return None
    return float(values[-1])


def rate_ratio_for_target(target, kinetics, t_end):
    """Find k_A/k_B whose ratio at ``t_end`` hours equals ``target``."""
    lo, hi = 1e-2, 1e4
    for _ in range(40):
        mid = float(np.sqrt(lo * hi))
        h_A, h_B = arm_lengths_for_rate_ratio(
            kinetics["k_A0"], kinetics["k_B0"], kinetics["beta"], mid
        )
        value = ratio_at_time(h_A, h_B, kinetics, t_end)
        if value is None:
            raise RuntimeError(f"ODE failed while matching target {target:.4f}.")
        if value < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def draw_parameters(rng):
    """Draw one kinetic parameter set inside the historical ranges."""
    return {
        "k_A0": float(rng.uniform(0.2, 0.6)),
        "k_B0": float(rng.uniform(0.05, 0.25)),
        "beta": float(rng.uniform(1.0, 1.8)),
        "r_A": float(rng.uniform(0.5, 0.6)),
        "r_B": float(rng.uniform(0.0, 0.05)),
        "alpha": ALPHA,
    }


def arm_lengths_for_rate_ratio(k_A0, k_B0, beta, rate_ratio, h_A=1000.0):
    """Choose arm lengths with k_A/k_B equal to ``rate_ratio``."""
    h_B = h_A * (rate_ratio * k_B0 / k_A0) ** (1.0 / beta)
    return h_A, float(h_B)


def simulate_ratio(times, h_A, h_B, kinetics):
    """Integrate one cell group and return L_A/L_B at ``times``."""
    k_A, k_B = arm_rates(h_A, h_B, kinetics["k_A0"], kinetics["k_B0"], kinetics["beta"])
    sol = solve_ivp(
        bxb1_ode, (0.0, float(times[-1])), [1.0, 0.0, 0.0, 0.0, 0.0],
        args=(k_A, k_B, kinetics["r_A"], kinetics["r_B"], kinetics["alpha"]),
        t_eval=times, method="RK45", rtol=1e-8, atol=1e-10,
    )
    if not sol.success:
        return None
    l_A, l_B = sol.y[1], sol.y[2]
    if np.any(l_B <= 0):
        return None
    return l_A / l_B


def sample_group(rng, target, n_pts, tol_rel, t_end):
    """One 24 h trajectory whose final ratio lies near ``target``."""
    kinetics = draw_parameters(rng)
    rate_ratio = rate_ratio_for_target(target, kinetics, t_end)
    h_A, h_B = arm_lengths_for_rate_ratio(
        kinetics["k_A0"], kinetics["k_B0"], kinetics["beta"], rate_ratio
    )
    times = np.linspace(t_end / n_pts, t_end, n_pts)
    ratio = simulate_ratio(times, h_A, h_B, kinetics)
    if ratio is None or abs(ratio[-1] - target) / target > tol_rel:
        raise RuntimeError(f"Ratio at {t_end:g} h did not reach target {target:.4f}.")
    kinetics["h_A"] = h_A
    kinetics["h_B"] = h_B
    return kinetics, times, ratio


def generate_synthetic_data(args):
    targets = target_ratio_grid(args.n_targets, args.ratio_min, args.ratio_max, args.target_ratios)
    if args.num_ids <= len(targets):
        raise ValueError(
            f"--num_ids ({args.num_ids}) must be greater than the number of stable-ratio targets ({len(targets)})."
        )
    if args.n_pts < 2:
        raise ValueError("--n_pts must be at least 2.")
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    ratio_records = []
    param_records = []
    for i in range(args.num_ids):
        target = float(targets[i % len(targets)])
        kinetics, times, ratio = sample_group(rng, target, args.n_pts, args.tol_rel, args.t_end)
        param_records.append((
            i, kinetics["h_A"], kinetics["h_B"], kinetics["k_A0"], kinetics["k_B0"],
            kinetics["beta"], kinetics["r_A"], kinetics["r_B"], kinetics["alpha"],
        ))
        for time, true_ratio in zip(times, ratio):
            observed = max(true_ratio + float(rng.normal(0.0, args.noise_sigma)), 1e-6)
            ratio_records.append((i, float(time), observed, target))

    ratio_df = pd.DataFrame(ratio_records, columns=["id", "time", "ratio", "target_ratio"])
    ratio_df.to_csv(os.path.join(args.save_dir, args.ratio_file), index=False)

    param_df = pd.DataFrame(
        param_records,
        columns=["id", "h_A", "h_B", "k_A0", "k_B0", "beta", "r_A", "r_B", "alpha"],
    )
    param_df.to_csv(os.path.join(args.save_dir, args.param_file), index=False)
    pd.DataFrame({"target_ratio": targets}).to_csv(
        os.path.join(args.save_dir, args.target_file), index=False
    )

    final = ratio_df.sort_values("time").groupby("id", as_index=False).tail(1)
    max_rel = np.max(np.abs(final["ratio"] - final["target_ratio"]) / final["target_ratio"])
    print(
        f"Generated {args.num_ids} cell groups around {len(targets)} stable ratios, "
        f"saved to '{args.save_dir}'. Largest final |ratio-target|/target after noise: {max_rel:.3f}."
    )
    return ratio_df, param_df, targets


if __name__ == "__main__":
    generate_synthetic_data(parse_args().parse_args())
