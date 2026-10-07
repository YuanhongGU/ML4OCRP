"""Train the OCRP deep-kernel GP and save a checkpoint.

Pipeline:

1. Load per-ID sequences ``{id: (times, ratios, scores)}``.
2. Pretrain ``RatioFeatureEncoder`` on pooled (ratio, score) pairs.
3. Freeze the encoder and build GP inputs ``[ratio | encoder_feat]``.
4. Fit ``DeepKernelGP`` by maximizing the exact marginal log-likelihood.
5. Save encoder, GP, likelihood, training tensors, and config.
"""

from ocrp_model import DeepKernelGP
from ocrp_model import RatioFeatureEncoder
from model_train_functions import pretrain_ratio_encoder, build_global_dataset

import argparse
import torch
import gpytorch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import random_split


def parse_arg() -> argparse.Namespace:
    """Parse CLI hyperparameters for OCRP GP training.

    Every former in-function default (encoder width, pretrain schedule,
    GP learning rate, validation split, log interval, device) is an optional
    flag. Omitted flags keep the historical values.

    :returns: Paths and model hyperparameters.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Train the OCRP deep-kernel Gaussian process.")

    parser.add_argument("-l", "--loadfile", required=False, type=str,   default="../Data/Preprocessed/id_data_dict.npy",
                        help="Path to id_data_dict.npy.")
    parser.add_argument("-s", "--save_dir", required=False, type=str,   default="../Data/Model",
                        help="Directory in which to write OCRP_model.pth.")
    parser.add_argument("--d_model",        required=False, type=int,   default=16,
                        help="Ratio-encoder embedding size.")
    parser.add_argument("--d_feat",         required=False, type=int,   default=8,
                        help="Latent size of the GP feature extractor.")
    parser.add_argument("--hidden_dim",     required=False, type=int,   default=32,
                        help="Hidden width of the ratio encoder and the GP feature MLP.")
    parser.add_argument("--lr",             required=False, type=float, default=0.01,
                        help="Adam learning rate for GP marginal-likelihood training.")
    parser.add_argument("--n_epochs",       required=False, type=int,   default=200,
                        help="Number of GP training epochs.")
    parser.add_argument("--pretrain_epochs", required=False, type=int,  default=100,
                        help="Number of ratio-encoder pretraining epochs.")
    parser.add_argument("--pretrain_lr",    required=False, type=float, default=0.001,
                        help="Adam learning rate for ratio-encoder pretraining.")
    parser.add_argument("--pretrain_batch_size", required=False, type=int, default=256,
                        help="Mini-batch size for ratio-encoder pretraining.")
    parser.add_argument("--val_ratio",      required=False, type=float, default=0.2,
                        help="Fraction of pooled samples held out for validation loss.")
    parser.add_argument("--log_interval",   required=False, type=int,   default=20,
                        help="Print pretrain and GP loss every this many epochs.")
    parser.add_argument("--mll_reduction",  required=False, type=str,   default="sum", choices=["sum", "mean"],
                        help="How to reduce a non-scalar marginal log-likelihood.")
    parser.add_argument("--device",         required=False, type=str,   default=None,
                        help="Torch device (mps, cuda, or cpu). Auto-detect if omitted.")
    parser.add_argument("--seed",           required=False, type=int,   default=None,
                        help="Seed for torch and numpy. Omit to keep the unseeded training split.")

    return parser.parse_args()


def get_device(device_arg):
    """Resolve a torch device from a CLI string or from hardware availability.

    Auto-detection order is MPS, then CUDA, then CPU.

    :param device_arg: Explicit device name, or ``None`` to auto-detect.
    :type device_arg: str or None
    :returns: Selected device.
    :rtype: torch.device
    """
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def reduce_mll(loss, reduction):
    """Collapse a non-scalar marginal log-likelihood to one value.

    :param loss: Loss tensor returned by GPyTorch.
    :type loss: torch.Tensor
    :param reduction: ``"sum"`` or ``"mean"``.
    :type reduction: str
    :returns: A scalar loss tensor.
    :rtype: torch.Tensor
    """
    if loss.numel() > 1:
        return loss.sum() if reduction == "sum" else loss.mean()
    return loss


def train_ocrp_model(id_data_dict, save_file, d_model=16, feat_dim=8, hidden_dim=32,
                     lr=0.01, n_epochs=200, val_ratio=0.2,
                     pretrain_epochs=100, pretrain_lr=0.001, pretrain_batch_size=256,
                     log_interval=20, mll_reduction="sum", device=None, log_fn=None):
    """Pretrain the ratio encoder, fit the deep-kernel GP, and save a checkpoint.

    Defaults match the ``ocrp_model_train.py`` command-line flags. ``log_fn``
    receives the same progress lines the CLI prints.

    :param id_data_dict: ``{id: (times, ratios, scores)}``.
    :type id_data_dict: dict
    :param save_file: Destination ``.pth`` path.
    :type save_file: str or pathlib.Path
    :param d_model: Ratio-encoder embedding size.
    :type d_model: int
    :param feat_dim: Latent size of the GP feature extractor.
    :type feat_dim: int
    :param hidden_dim: Hidden width of the encoder and the GP feature MLP.
    :type hidden_dim: int
    :param lr: Adam learning rate for GP training.
    :type lr: float
    :param n_epochs: Number of GP training epochs.
    :type n_epochs: int
    :param val_ratio: Fraction of pooled samples held out for validation loss.
    :type val_ratio: float
    :param pretrain_epochs: Encoder pretraining epochs.
    :type pretrain_epochs: int
    :param pretrain_lr: Encoder pretraining learning rate.
    :type pretrain_lr: float
    :param pretrain_batch_size: Encoder pretraining batch size.
    :type pretrain_batch_size: int
    :param log_interval: Print loss every this many epochs.
    :type log_interval: int
    :param mll_reduction: ``"sum"`` or ``"mean"`` for a non-scalar MLL.
    :type mll_reduction: str
    :param device: Torch device. Auto-detected when ``None``.
    :type device: torch.device or None
    :param log_fn: Optional ``callable(str)`` used instead of ``print``.
    :type log_fn: callable or None
    :returns: Path of the written checkpoint.
    :rtype: pathlib.Path
    """
    log = log_fn or print
    save_file = Path(save_file)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = get_device(None)
    log(f"Using device: {device}")

    id_sequences = []
    for id_, (times, ratios, scores) in id_data_dict.items():
        id_sequences.append(list(zip(times, ratios, scores)))

    # Initialize and pre-train the ratio encoder, then freeze it so GP training
    # only updates the deep kernel and likelihood.
    ratio_encoder = RatioFeatureEncoder(d_model=d_model, hidden_dim=hidden_dim).to(device)
    ratio_encoder = pretrain_ratio_encoder(
        ratio_encoder, id_sequences, device,
        epochs=pretrain_epochs,
        lr=pretrain_lr,
        batch_size=pretrain_batch_size,
        log_interval=log_interval,
    )
    for param in ratio_encoder.parameters():
        param.requires_grad = False

    # Build the dataset from ratios and scores. Identifiers and times stay out.
    train_dataset = build_global_dataset(id_sequences, ratio_encoder, device)

    # Hold out a random subset for validation loss logging (ExactGP still fits
    # on the train split only).
    epochs = n_epochs
    val_size   = int(len(train_dataset) * val_ratio)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # Batch size = full split: ExactGP uses the entire training set each step.
    train_loader = DataLoader(train_subset, batch_size=len(train_subset), shuffle=False)
    val_loader   = DataLoader(val_subset, batch_size=len(val_subset), shuffle=False)

    for features, scores in train_loader:
        train_x, train_y = features, scores
    for val_features, val_scores in val_loader:
        val_x, val_y = val_features, val_scores

    # The GP input is the raw ratio plus the encoder feature.
    input_dim = 1 + d_model

    # Create the deep-kernel Gaussian process.
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = DeepKernelGP(
        train_x, train_y, likelihood,
        input_dim=input_dim, feat_dim=feat_dim, hidden_dim=hidden_dim,
    )
    model.to(device)

    # Maximize exact MLL: GPyTorch returns -mll as the quantity to minimize.
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = reduce_mll(-mll(output, train_y), mll_reduction)
        loss.backward()
        optimizer.step()
        if log_interval > 0 and epoch % log_interval == 0:
            # Switch to eval for a held-out MLL snapshot, then resume training.
            model.eval(); likelihood.eval()
            with torch.no_grad():
                val_output = model(val_x)
                val_loss = reduce_mll(-mll(val_output, val_y), mll_reduction)
            model.train(); likelihood.train()
            log(f'Epoch {epoch:3d} | Train Loss: {loss.item():.3f} | Val Loss: {val_loss.item():.3f}')

    # Persist encoder + GP + the ExactGP training cache needed at load time.
    checkpoint = {
        'ratio_encoder_state_dict': ratio_encoder.state_dict(),
        'gp_model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'train_x': train_x.cpu(),
        'train_y': train_y.cpu(),
        'config': {
            'd_model': d_model,
            'feat_dim': feat_dim,
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
        }
    }
    torch.save(checkpoint, save_file)
    log(f"Saved checkpoint to {save_file}")
    return save_file


def main():
    """Run encoder pretraining, GP fitting, and checkpoint export."""
    args = parse_arg()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    id_data_dict = np.load(Path(args.loadfile), allow_pickle=True).item()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    train_ocrp_model(
        id_data_dict,
        save_dir.joinpath("OCRP_model.pth"),
        d_model=args.d_model,
        feat_dim=args.d_feat,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        n_epochs=args.n_epochs,
        val_ratio=args.val_ratio,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_lr=args.pretrain_lr,
        pretrain_batch_size=args.pretrain_batch_size,
        log_interval=args.log_interval,
        mll_reduction=args.mll_reduction,
        device=get_device(args.device),
    )

if __name__ == "__main__":
    main()
