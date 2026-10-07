"""Load a trained OCRP model and recommend the next experimental ratio.

Two searches are performed on the frozen deep-kernel GP:

* Gradient-based maximization of the GP mean (global "optimal" ratio).
* Upper Confidence Bound (UCB) over a grid (exploration-aware next ratio).

Results are written to ``predict_result.csv``.
"""

from ocrp_model import DeepKernelGP, RatioFeatureEncoder

import os
import torch
import gpytorch
import numpy as np
import pandas as pd
import argparse


def parse_arg():
    """Parse checkpoint paths and the search hyperparameters.

    Gradient-ascent and UCB settings that used to be keyword defaults on
    ``predict_optimal_ratio`` and ``ucb_acquisition`` are optional flags.
    Omitted flags keep those historical defaults.

    :returns: Parsed CLI namespace.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Predict the global optimal cell ratio and a UCB sampling recommendation."
    )

    parser.add_argument("--trained_model", required=False, type=str, default="OCRP_model.pth",
                        help="Checkpoint filename inside --loadmodel.")
    parser.add_argument("--loaddir", required=False, type=str, default="../Data/Preprocessed",
                        help="Fallback output directory used when --savedir is empty.")
    parser.add_argument("--loadmodel", required=False, type=str, default="../Data/Model",
                        help="Directory that contains the trained checkpoint.")
    parser.add_argument("-s", "--savedir", required=False, type=str, default="../Data/Predict",
                        help="Directory in which to write the prediction CSV.")
    parser.add_argument("--save_filename", required=False, type=str, default="predict_result.csv",
                        help="Filename of the prediction CSV.")
    parser.add_argument("--device", required=False, type=str, default=None,
                        help="Torch device (cuda or cpu). CUDA is used when available if omitted.")
    parser.add_argument("--opt_ratio_min", required=False, type=float, default=0.01,
                        help="Lower bound of the gradient search for the optimal ratio.")
    parser.add_argument("--opt_ratio_max", required=False, type=float, default=5.0,
                        help="Upper bound of the gradient search for the optimal ratio.")
    parser.add_argument("--n_starts", required=False, type=int, default=10,
                        help="Number of equally spaced starts for gradient ascent.")
    parser.add_argument("--n_steps", required=False, type=int, default=100,
                        help="Adam steps per gradient-ascent start.")
    parser.add_argument("--opt_lr", required=False, type=float, default=0.01,
                        help="Adam learning rate on the scalar ratio during gradient ascent.")
    parser.add_argument("--ucb_ratio_min", required=False, type=float, default=0.1,
                        help="Lower bound of the UCB candidate grid.")
    parser.add_argument("--ucb_ratio_max", required=False, type=float, default=10.0,
                        help="Upper bound of the UCB candidate grid.")
    parser.add_argument("--n_candidates", required=False, type=int, default=100,
                        help="Number of grid points evaluated by UCB.")
    parser.add_argument("--beta", required=False, type=float, default=2.0,
                        help="UCB exploration coefficient (mean + beta * std).")
    return parser.parse_args()


def get_device(device_arg):
    """Resolve a torch device, preserving CUDA-then-CPU auto-detection.

    :param device_arg: Explicit device name, or ``None`` to auto-detect.
    :type device_arg: str or None
    :returns: Selected device.
    :rtype: torch.device
    """
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_trained_model(checkpoint_path, device):
    """Restore the frozen ratio encoder and ExactGP from a training checkpoint.

    ExactGP needs the original ``train_x`` / ``train_y`` to rebuild the kernel
    cache, so those tensors are loaded from the checkpoint rather than from
    raw data.

    :param checkpoint_path: Path to the ``.pth`` file written by ``ocrp_model_train``.
    :type checkpoint_path: str
    :param device: Device on which to place encoder, GP, and likelihood.
    :type device: torch.device
    :returns: ``(ratio_encoder, model, likelihood)`` all in eval mode, with
        encoder gradients disabled.
    :rtype: tuple
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    hidden_dim = config.get("hidden_dim", 32)
    ratio_encoder = RatioFeatureEncoder(d_model=config["d_model"], hidden_dim=hidden_dim).to(device)
    ratio_encoder.load_state_dict(checkpoint['ratio_encoder_state_dict'])
    ratio_encoder.eval()
    # Encoder was frozen after pretraining; keep it frozen at inference.
    for param in ratio_encoder.parameters():
        param.requires_grad = False

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    train_x = checkpoint['train_x'].to(device)
    train_y = checkpoint['train_y'].to(device)
    model = DeepKernelGP(train_x, train_y, likelihood,
                         input_dim=config['input_dim'],
                         feat_dim=config['feat_dim'],
                         hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint['gp_model_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    model.eval(); likelihood.eval()
    return ratio_encoder, model, likelihood


def predict_optimal_ratio(model, ratio_encoder, device, ratio_range=(0.01, 5.0), n_starts=10, n_steps=100, lr=0.01):
    """Maximize the GP posterior mean over ratio via multi-start gradient ascent.

    Because the landscape can be multi-modal, optimization is restarted from
    ``n_starts`` equally spaced points in ``ratio_range``. Each start clamps
    the ratio to the interval after every Adam step.

    Defaults match ``--opt_ratio_min``, ``--opt_ratio_max``, ``--n_starts``,
    ``--n_steps``, and ``--opt_lr``.

    :param model: Trained ``DeepKernelGP`` in eval mode.
    :type model: ocrp_model.DeepKernelGP
    :param ratio_encoder: Frozen encoder used to build GP inputs.
    :type ratio_encoder: ocrp_model.RatioFeatureEncoder
    :param device: Device for the ratio parameter and model.
    :type device: torch.device
    :param ratio_range: Inclusive ``(low, high)`` search interval.
    :type ratio_range: tuple
    :param n_starts: Number of equally spaced initial ratios.
    :type n_starts: int
    :param n_steps: Adam steps per start.
    :type n_steps: int
    :param lr: Adam learning rate on the scalar ratio.
    :type lr: float
    :returns: ``(best_ratio, best_score)`` where ``best_score`` is the GP mean
        at ``best_ratio``.
    :rtype: tuple
    """
    model.eval()
    best_ratio, best_score = None, -np.inf

    for start in np.linspace(ratio_range[0], ratio_range[1], n_starts):
        ratio = torch.tensor([start], device=device, requires_grad=True, dtype=torch.float32)
        optimizer = torch.optim.Adam([ratio], lr=lr)
        for _ in range(n_steps):
            optimizer.zero_grad()
            feat = ratio_encoder.get_feature(ratio.unsqueeze(-1))  # (1, d_model)
            x = torch.cat([ratio.unsqueeze(-1), feat], dim=-1)     # (1, 1+d_model)
            pred = model(x)  # MultivariateNormal
            mean = pred.mean
            # Gradient ascent on mean == descent on -mean.
            loss = -mean
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                ratio.clamp_(ratio_range[0], ratio_range[1])
        final_ratio = ratio.item()
        final_score = model(torch.cat([ratio.unsqueeze(-1), ratio_encoder.get_feature(ratio.unsqueeze(-1))], dim=-1)).mean.item()
        if final_score > best_score:
            best_score, best_ratio = final_score, final_ratio
    return best_ratio, best_score


def ucb_acquisition(model, likelihood, ratio_encoder, device, ratio_range=(0.1, 10.0), n_candidates=100, beta=2.0):
    """Pick the grid ratio that maximizes GP-UCB: mean + beta * std.

    Higher ``beta`` prefers uncertain regions (exploration); lower ``beta``
    prefers high predicted mean (exploitation). The likelihood is set to eval
    so predictive variance is the posterior, not the training prior.

    Defaults match ``--ucb_ratio_min``, ``--ucb_ratio_max``, ``--n_candidates``,
    and ``--beta``.

    :param model: Trained ``DeepKernelGP``.
    :type model: ocrp_model.DeepKernelGP
    :param likelihood: Matching Gaussian likelihood (eval mode).
    :type likelihood: gpytorch.likelihoods.GaussianLikelihood
    :param ratio_encoder: Frozen ratio encoder.
    :type ratio_encoder: ocrp_model.RatioFeatureEncoder
    :param device: Device for candidate tensors.
    :type device: torch.device
    :param ratio_range: Inclusive ``(low, high)`` grid bounds.
    :type ratio_range: tuple
    :param n_candidates: Number of linspace grid points.
    :type n_candidates: int
    :param beta: UCB exploration coefficient.
    :type beta: float
    :returns: ``(best_ratio, best_ucb)`` for the argmax grid point.
    :rtype: tuple
    """
    model.eval(); likelihood.eval()
    candidates = np.linspace(ratio_range[0], ratio_range[1], n_candidates)
    best_ucb, best_ratio = -np.inf, None
    with torch.no_grad():
        for r in candidates:
            ratio_t = torch.tensor([[r]], dtype=torch.float32, device=device)
            feat = ratio_encoder.get_feature(ratio_t)
            x = torch.cat([ratio_t, feat], dim=-1)
            pred = model(x)
            ucb = pred.mean.item() + beta * pred.stddev.item()
            if ucb > best_ucb:
                best_ucb, best_ratio = ucb, r
    return best_ratio, best_ucb


if __name__ == "__main__":
    args = parse_arg()
    LOAD_MODEL = os.path.join(args.loadmodel, args.trained_model)
    device = get_device(args.device)
    print(f"Using device: {device}")
    ratio_encoder, model, likelihood = load_trained_model(LOAD_MODEL, device)

    opt_ratio, max_score = predict_optimal_ratio(
        model, ratio_encoder, device,
        ratio_range=(args.opt_ratio_min, args.opt_ratio_max),
        n_starts=args.n_starts,
        n_steps=args.n_steps,
        lr=args.opt_lr,
    )
    print(f"Global optimal ratio: {opt_ratio:.4f}, predicted score: {max_score:.4f}")

    next_ratio, ucb_value = ucb_acquisition(
        model, likelihood, ratio_encoder, device,
        ratio_range=(args.ucb_ratio_min, args.ucb_ratio_max),
        n_candidates=args.n_candidates,
        beta=args.beta,
    )
    print(f"UCB recommended ratio: {next_ratio:.4f}, UCB: {ucb_value:.4f}")

    res = {"GLOBAL_OPTIMAL_RATIO": opt_ratio, "MAX_SCORE": max_score,
           "UCB_RECOMMEND": next_ratio, "UCB_SCORE": ucb_value}
    pd.DataFrame([res]).to_csv(
        os.path.join(args.savedir or args.loaddir, args.save_filename), index=False
    )
