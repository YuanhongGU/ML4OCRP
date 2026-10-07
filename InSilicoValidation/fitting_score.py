import numpy as np
import pandas as pd
import argparse
import os

SCORE_FUNCTIONS = ("gaussian", "log_gaussian", "sigmoid", "bimodal", "linear")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute scores from ratio data with a selectable scoring function."
    )
    parser.add_argument("--ratio_file", type=str, default="ratio_data.csv",
                        help="Input CSV file with columns: id, time, ratio")
    parser.add_argument("--score_file", type=str, default="score_data.csv",
                        help="Output CSV file for id, time, score")
    parser.add_argument("--score_fn", type=str, default="gaussian", choices=SCORE_FUNCTIONS,
                        help="Scoring-function shape.")
    parser.add_argument("--opt_ratio", type=float, default=2.5,
                        help="Location parameter. Peak for gaussian and log_gaussian; midpoint for sigmoid; first mode for bimodal.")
    parser.add_argument("--sigma", type=float, default=0.5,
                        help="Width of the primary peak, or the slope scale of the sigmoid.")
    parser.add_argument("--max_score", type=float, default=1.0,
                        help="Highest score the function is scaled to reach.")
    parser.add_argument("--opt_ratio_2", type=float, default=4.0,
                        help="Second mode of the bimodal function.")
    parser.add_argument("--sigma_2", type=float, default=0.45,
                        help="Width of the second bimodal mode.")
    parser.add_argument("--peak_weight", type=float, default=0.35,
                        help="Relative height of the first bimodal mode before the taller mode is scaled to --max_score.")
    parser.add_argument("--linear_min", type=float, default=0.5,
                        help="Ratio at which the linear score is 0.")
    parser.add_argument("--linear_max", type=float, default=6.0,
                        help="Ratio at which the linear score reaches --max_score.")
    parser.add_argument("-l", "--load_dir", type=str, default="../Data/Original",
                        help="Load path of data.")
    parser.add_argument("-s", "--save_dir", type=str, default="../Data/Original",
                        help="Save path of data.")
    return parser


def gaussian_score(ratio, opt_ratio, sigma, max_score):
    """Symmetric peak: max_score * exp(-((ratio - opt_ratio)^2) / (2 * sigma^2))."""
    return max_score * np.exp(-((ratio - opt_ratio) ** 2) / (2 * sigma ** 2))


def log_gaussian_score(ratio, opt_ratio, sigma, max_score):
    """Asymmetric peak. The Gaussian is evaluated on log(ratio)."""
    safe_ratio = np.maximum(ratio, 1e-8)
    safe_opt = max(float(opt_ratio), 1e-8)
    return max_score * np.exp(-((np.log(safe_ratio) - np.log(safe_opt)) ** 2) / (2 * sigma ** 2))


def sigmoid_score(ratio, opt_ratio, sigma, max_score):
    """Increasing S-shaped score. --opt_ratio is the midpoint, --sigma sets the slope."""
    slope = max(float(sigma), 1e-8)
    return max_score / (1.0 + np.exp(-(ratio - opt_ratio) / slope))


def bimodal_score(ratio, opt_ratio, sigma, max_score, opt_ratio_2, sigma_2, peak_weight):
    """Two Gaussian peaks. The taller peak is scaled to max_score."""
    weight = float(np.clip(peak_weight, 0.0, 1.0))
    first = weight * gaussian_score(ratio, opt_ratio, sigma, 1.0)
    second = (1.0 - weight) * gaussian_score(ratio, opt_ratio_2, sigma_2, 1.0)
    taller = max(weight, 1.0 - weight)
    if taller <= 0:
        return np.zeros_like(np.asarray(ratio, dtype=float))
    return max_score * (first + second) / taller


def linear_score(ratio, linear_min, linear_max, max_score):
    """Straight ramp from 0 at linear_min to max_score at linear_max."""
    span = float(linear_max) - float(linear_min)
    if span == 0:
        raise ValueError("--linear_max must differ from --linear_min.")
    return max_score * np.clip((ratio - linear_min) / span, 0.0, 1.0)


def compute_score(ratio, score_fn, opt_ratio, sigma, max_score,
                  opt_ratio_2=4.0, sigma_2=0.45, peak_weight=0.35,
                  linear_min=0.5, linear_max=6.0):
    """Evaluate one supported scoring function on an array of ratios."""
    ratio = np.asarray(ratio, dtype=float)
    if score_fn == "gaussian":
        return gaussian_score(ratio, opt_ratio, sigma, max_score)
    if score_fn == "log_gaussian":
        return log_gaussian_score(ratio, opt_ratio, sigma, max_score)
    if score_fn == "sigmoid":
        return sigmoid_score(ratio, opt_ratio, sigma, max_score)
    if score_fn == "bimodal":
        return bimodal_score(ratio, opt_ratio, sigma, max_score, opt_ratio_2, sigma_2, peak_weight)
    if score_fn == "linear":
        return linear_score(ratio, linear_min, linear_max, max_score)
    raise ValueError(f"Unknown scoring function '{score_fn}'. Choose from {SCORE_FUNCTIONS}.")


def score_table(df, args):
    """Return an id/time/score table for the ratios in ``df``."""
    required = ["id", "time", "ratio"]
    if not all(col in df.columns for col in required):
        raise ValueError(f"Input file must contain columns: {required}")
    scored = df.copy()
    scored["score"] = compute_score(
        scored["ratio"].to_numpy(),
        args.score_fn,
        args.opt_ratio,
        args.sigma,
        args.max_score,
        opt_ratio_2=args.opt_ratio_2,
        sigma_2=args.sigma_2,
        peak_weight=args.peak_weight,
        linear_min=args.linear_min,
        linear_max=args.linear_max,
    )
    return scored[["id", "time", "score"]]


if __name__ == "__main__":
    args = parse_args().parse_args()
    df = pd.read_csv(os.path.join(args.load_dir, args.ratio_file))
    score_df = score_table(df, args)
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, args.score_file)
    score_df.to_csv(out_path, index=False)
    print(f"Saved {args.score_fn} scores to {out_path}")
