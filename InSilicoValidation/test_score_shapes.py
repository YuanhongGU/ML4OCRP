"""Train ML4OCRP once for each scoring-function shape and record the error.

The ratio table is generated once. Every cell group is sampled near the same
ten target ratios. Each scoring function then labels that table, and the
OCRP model is trained and asked for its optimal ratio. The comparison is
against the true maximum of the scoring function on the same search interval.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fitting_ratio
import fitting_score


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "Model"
RESULTS = Path(__file__).resolve().parent / "results"
SVG_DIR = ROOT.parent / "SoftwareWiki" / "figures"

SEARCH_MIN = 0.5
SEARCH_MAX = 6.0

PRESETS = [
    {
        "name": "gaussian",
        "score_fn": "gaussian",
        "opt_ratio": 2.5,
        "sigma": 0.5,
        "en": "Symmetric peak",
        "zh": "对称单峰",
    },
    {
        "name": "log_gaussian",
        "score_fn": "log_gaussian",
        "opt_ratio": 2.5,
        "sigma": 0.35,
        "en": "Asymmetric peak",
        "zh": "不对称单峰",
    },
    {
        "name": "bimodal",
        "score_fn": "bimodal",
        "opt_ratio": 1.2,
        "sigma": 0.35,
        "opt_ratio_2": 4.0,
        "sigma_2": 0.40,
        "peak_weight": 0.35,
        "en": "Two peaks",
        "zh": "双峰",
    },
    {
        "name": "sigmoid",
        "score_fn": "sigmoid",
        "opt_ratio": 2.5,
        "sigma": 0.6,
        "en": "Increasing sigmoid",
        "zh": "递增 S 形",
    },
    {
        "name": "linear",
        "score_fn": "linear",
        "linear_min": 0.5,
        "linear_max": 6.0,
        "en": "Linear ramp",
        "zh": "线性上升",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare ML4OCRP across scoring-function shapes.")
    parser.add_argument("--num_ids", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def score_kwargs(preset):
    return {
        "score_fn": preset["score_fn"],
        "opt_ratio": preset.get("opt_ratio", 2.5),
        "sigma": preset.get("sigma", 0.5),
        "max_score": 1.0,
        "opt_ratio_2": preset.get("opt_ratio_2", 4.0),
        "sigma_2": preset.get("sigma_2", 0.45),
        "peak_weight": preset.get("peak_weight", 0.35),
        "linear_min": preset.get("linear_min", 0.5),
        "linear_max": preset.get("linear_max", 6.0),
    }


def true_optimum(preset, grid):
    values = fitting_score.compute_score(grid, **score_kwargs(preset))
    index = int(np.argmax(values))
    return float(grid[index]), float(values[index]), values


def run_script(script, arguments):
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    command = [sys.executable, str(script), *arguments]
    subprocess.run(command, check=True, cwd=MODEL_DIR, env=env)


def gp_mean(checkpoint, grid, device_name):
    sys.path.insert(0, str(MODEL_DIR))
    import torch
    from predict_ratio_and_recommend_by_ucb import load_trained_model

    device = torch.device(device_name)
    encoder, model, _likelihood = load_trained_model(str(checkpoint), device)
    ratio = torch.tensor(grid, dtype=torch.float32, device=device).unsqueeze(-1)
    with torch.no_grad():
        features = encoder.get_feature(ratio)
        prediction = model(torch.cat([ratio, features], dim=-1))
        return prediction.mean.detach().cpu().numpy()


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)
    ax.set_axisbelow(True)


def save_svg(fig, stem):
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SVG_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_ratios(ratio_df, targets):
    final = ratio_df.sort_values("time").groupby("id", as_index=False).tail(1)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(final["id"], final["ratio"], s=28, color="#24b8ec", zorder=3)
    for target in targets:
        ax.axhline(target, color="#d1d5db", linewidth=0.6, zorder=0)
    ax.set_xlabel("Cell group")
    ax.set_ylabel("Ratio at 24 h")
    ax.set_title("More than ten cell groups, 24 h ratios near ten targets")
    style_axis(ax)
    save_svg(fig, "fig-03-ratio-targets-en")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(final["id"], final["ratio"], s=28, color="#24b8ec", zorder=3)
    for target in targets:
        ax.axhline(target, color="#d1d5db", linewidth=0.6, zorder=0)
    ax.set_xlabel("细胞群")
    ax.set_ylabel("24 小时的比例")
    ax.set_title("细胞群多于十个，24 小时的比例落在十个目标附近")
    style_axis(ax)
    save_svg(fig, "fig-03-ratio-targets-zh")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def plot_shapes(grid, curves):
    colors = ["#e84d85", "#24b8ec", "#9b6ca8", "#c6da58", "#ee7954"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for color, preset, values in zip(colors, PRESETS, curves):
        ax.plot(grid, values, color=color, linewidth=1.8, label=preset["en"])
    ax.set_xlabel("Ratio")
    ax.set_ylabel("Score")
    ax.set_title("Scoring functions")
    ax.legend(frameon=False)
    style_axis(ax)
    save_svg(fig, "fig-04-score-shapes-en")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for color, preset, values in zip(colors, PRESETS, curves):
        ax.plot(grid, values, color=color, linewidth=1.8, label=preset["zh"])
    ax.set_xlabel("比例")
    ax.set_ylabel("分数")
    ax.set_title("打分函数")
    ax.legend(frameon=False)
    style_axis(ax)
    save_svg(fig, "fig-04-score-shapes-zh")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def plot_fits(grid, bundle):
    fig, axes = plt.subplots(1, len(bundle), figsize=(14.5, 3.3), sharey=True)
    for ax, item in zip(axes, bundle):
        ax.plot(grid, item["true_curve"], color="#111827", linewidth=1.4, label="True")
        ax.plot(grid, item["pred_curve"], color="#24b8ec", linewidth=1.4, label="OCRP")
        ax.scatter(item["ratios"], item["scores"], s=8, color="#e84d85", alpha=0.35, linewidths=0)
        ax.axvline(item["true_opt"], color="#111827", linewidth=0.7, linestyle="--")
        ax.axvline(item["pred_opt"], color="#24b8ec", linewidth=0.7, linestyle="--")
        ax.set_title(item["en"])
        ax.set_xlabel("Ratio")
        style_axis(ax)
    axes[0].set_ylabel("Score")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_svg(fig, "fig-05-score-shape-fit-en")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, len(bundle), figsize=(14.5, 3.3), sharey=True)
    for ax, item in zip(axes, bundle):
        ax.plot(grid, item["true_curve"], color="#111827", linewidth=1.4, label="真实")
        ax.plot(grid, item["pred_curve"], color="#24b8ec", linewidth=1.4, label="OCRP")
        ax.scatter(item["ratios"], item["scores"], s=8, color="#e84d85", alpha=0.35, linewidths=0)
        ax.axvline(item["true_opt"], color="#111827", linewidth=0.7, linestyle="--")
        ax.axvline(item["pred_opt"], color="#24b8ec", linewidth=0.7, linestyle="--")
        ax.set_title(item["zh"])
        ax.set_xlabel("比例")
        style_axis(ax)
    axes[0].set_ylabel("分数")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_svg(fig, "fig-05-score-shape-fit-zh")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def plot_errors(summary):
    labels = [row["en"] for row in summary]
    labels_zh = [row["zh"] for row in summary]
    error = [row["abs_error"] for row in summary]
    regret = [row["regret"] for row in summary]
    x = np.arange(len(summary))

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    axes[0].bar(x, error, color="#24b8ec")
    axes[1].bar(x, regret, color="#e84d85")
    axes[0].set_ylabel("Absolute error of the optimal ratio")
    axes[1].set_ylabel("Score regret")
    for ax, title in zip(axes, ("Where the predicted peak sits", "Score lost at that peak")):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(title)
        style_axis(ax)
    fig.tight_layout()
    save_svg(fig, "fig-06-score-shape-performance-en")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    axes[0].bar(x, error, color="#24b8ec")
    axes[1].bar(x, regret, color="#e84d85")
    axes[0].set_ylabel("最优比例的绝对误差")
    axes[1].set_ylabel("分数损失")
    for ax, title in zip(axes, ("预测峰值的位置", "该位置上损失的分数")):
        ax.set_xticks(x)
        ax.set_xticklabels(labels_zh, rotation=25, ha="right")
        ax.set_title(title)
        style_axis(ax)
    fig.tight_layout()
    save_svg(fig, "fig-06-score-shape-performance-zh")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def main():
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    ratio_dir = RESULTS / "ratio"
    ratio_args = argparse.Namespace(
        num_ids=args.num_ids,
        n_targets=10,
        n_pts=10,
        t_end=24.0,
        ratio_min=SEARCH_MIN,
        ratio_max=SEARCH_MAX,
        target_ratios="",
        tol_rel=0.05,
        max_tries=40,
        noise_sigma=0.02,
        seed=args.seed,
        ratio_file="ratio_data.csv",
        param_file="param_data.csv",
        target_file="target_ratios.csv",
        save_dir=str(ratio_dir),
    )
    ratio_df, _params, targets = fitting_ratio.generate_synthetic_data(ratio_args)
    plot_ratios(ratio_df, targets)

    grid = np.linspace(SEARCH_MIN, SEARCH_MAX, 401)
    curves = []
    summary = []
    fit_bundle = []
    for preset in PRESETS:
        shape_dir = RESULTS / preset["name"]
        original = shape_dir / "original"
        preprocessed = shape_dir / "preprocessed"
        model_dir = shape_dir / "model"
        predict_dir = shape_dir / "predict"
        for path in (original, preprocessed, model_dir, predict_dir):
            path.mkdir(parents=True, exist_ok=True)
        ratio_df.to_csv(original / "ratio_data.csv", index=False)

        score_args = argparse.Namespace(**score_kwargs(preset))
        fitting_score.score_table(ratio_df, score_args).to_csv(original / "score_data.csv", index=False)

        run_script(MODEL_DIR / "preprocess_data.py", [
            "--load_dir", str(original),
            "--save_dir", str(preprocessed),
        ])
        run_script(MODEL_DIR / "ocrp_model_train.py", [
            "--loadfile", str(preprocessed / "id_data_dict.npy"),
            "--save_dir", str(model_dir),
            "--n_epochs", str(args.n_epochs),
            "--pretrain_epochs", str(args.pretrain_epochs),
            "--device", args.device,
            "--seed", str(args.seed),
            "--log_interval", "50",
        ])
        run_script(MODEL_DIR / "predict_ratio_and_recommend_by_ucb.py", [
            "--loadmodel", str(model_dir),
            "--savedir", str(predict_dir),
            "--device", args.device,
            "--opt_ratio_min", str(SEARCH_MIN),
            "--opt_ratio_max", str(SEARCH_MAX),
            "--ucb_ratio_min", str(SEARCH_MIN),
            "--ucb_ratio_max", str(SEARCH_MAX),
        ])

        predicted = pd.read_csv(predict_dir / "predict_result.csv").iloc[0]
        true_opt, true_peak, true_curve = true_optimum(preset, grid)
        pred_opt = float(predicted["GLOBAL_OPTIMAL_RATIO"])
        true_at_pred = float(fitting_score.compute_score(np.array([pred_opt]), **score_kwargs(preset))[0])
        pred_curve = gp_mean(model_dir / "OCRP_model.pth", grid, args.device)
        row = {
            "name": preset["name"],
            "en": preset["en"],
            "zh": preset["zh"],
            "true_optimal_ratio": true_opt,
            "predicted_optimal_ratio": pred_opt,
            "abs_error": abs(pred_opt - true_opt),
            "true_peak_score": true_peak,
            "true_score_at_prediction": true_at_pred,
            "regret": true_peak - true_at_pred,
            "score_rmse": float(np.sqrt(np.mean((pred_curve - true_curve) ** 2))),
            "ucb_recommend": float(predicted["UCB_RECOMMEND"]),
        }
        summary.append(row)
        curves.append(true_curve)
        fit_bundle.append({
            **row,
            "true_curve": true_curve,
            "pred_curve": pred_curve,
            "ratios": ratio_df["ratio"].to_numpy(),
            "scores": pd.read_csv(original / "score_data.csv")["score"].to_numpy(),
            "true_opt": true_opt,
            "pred_opt": pred_opt,
        })
        print(
            f"{preset['name']}: true {true_opt:.3f}, predicted {pred_opt:.3f}, "
            f"abs {row['abs_error']:.3f}, regret {row['regret']:.4f}"
        )

    summary_df = pd.DataFrame(summary).drop(columns=["en", "zh"])
    summary_df.to_csv(RESULTS / "score_shape_summary.csv", index=False)
    plot_shapes(grid, curves)
    plot_fits(grid, fit_bundle)
    plot_errors(summary)
    print(f"Wrote {RESULTS / 'score_shape_summary.csv'}")


if __name__ == "__main__":
    main()
