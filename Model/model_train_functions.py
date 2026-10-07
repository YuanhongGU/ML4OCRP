"""Training helpers for the OCRP ratio encoder and GP dataset.

``pretrain_ratio_encoder`` teaches the encoder to map ratios to experimental
scores. ``build_global_dataset`` then freezes that encoder and builds GP
training pairs of the form ``[ratio, encoder_features] -> score``.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def pretrain_ratio_encoder(ratio_encoder, id_sequences, device, epochs=100, lr=0.001,
                           batch_size=256, log_interval=20):
    """Supervise the ratio encoder with observed (ratio, score) pairs.

    All IDs are pooled into one regression dataset. The encoder's auxiliary
    score head is trained with MSE so the embedding captures score-relevant
    structure before it is frozen for the GP.

    Defaults match ``ocrp_model_train.py`` flags ``--pretrain_epochs``,
    ``--pretrain_lr``, ``--pretrain_batch_size``, and ``--log_interval``.

    :param ratio_encoder: ``RatioFeatureEncoder`` instance to train in-place.
    :type ratio_encoder: ocrp_model.RatioFeatureEncoder
    :param id_sequences: Iterable of sequences; each sequence is a list of
        ``(time, ratio, score)`` triples.
    :type id_sequences: list
    :param device: Torch device for tensors and the encoder.
    :type device: torch.device
    :param epochs: Number of full passes over the pooled dataset. Defaults to 100.
    :type epochs: int
    :param lr: Adam learning rate. Defaults to 0.001.
    :type lr: float
    :param batch_size: Mini-batch size. Defaults to 256.
    :type batch_size: int
    :param log_interval: Print the mean loss every this many epochs. Defaults to 20.
    :type log_interval: int
    :returns: The encoder in eval mode after pretraining.
    :rtype: ocrp_model.RatioFeatureEncoder
    """
    ratio_encoder.train()
    optimizer = torch.optim.Adam(ratio_encoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Flatten per-ID time series into a single (ratio, score) pool.
    all_ratios, all_scores = [], []
    for seq in id_sequences:
        for _, r, s in seq:
            all_ratios.append(r)
            all_scores.append(s)

    ratios_t = torch.tensor(all_ratios, dtype=torch.float32, device=device).unsqueeze(-1)
    scores_t = torch.tensor(all_scores, dtype=torch.float32, device=device).unsqueeze(-1)

    dataset  = TensorDataset(ratios_t, scores_t)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_r, batch_s in loader:
            optimizer.zero_grad()
            # Use the auxiliary head; the embedding is trained indirectly via MSE.
            _, pred_s = ratio_encoder(batch_r)
            loss = criterion(pred_s, batch_s)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if log_interval > 0 and epoch % log_interval == 0:
            print(f"Pretrain epoch {epoch:3d} | Loss: {total_loss/len(loader):.6f}")

    ratio_encoder.eval()
    return ratio_encoder


def build_global_dataset(id_sequences, ratio_encoder, device):
    """Build GP training data from frozen ratio embeddings.

    Each observed sample becomes ``concat(ratio, encoder_feature)`` as input
    and the experimental score as the target. Time and ID are not included:
    the GP models a global ratio-to-score mapping.

    :param id_sequences: Iterable of sequences of ``(time, ratio, score)``.
    :type id_sequences: list
    :param ratio_encoder: Frozen encoder; only ``get_feature`` is used.
    :type ratio_encoder: ocrp_model.RatioFeatureEncoder
    :param device: Device on which to run the encoder.
    :type device: torch.device
    :returns: Features of shape ``(N, 1 + d_model)`` and scores of shape ``(N,)``.
    :rtype: torch.utils.data.TensorDataset
    """
    all_features, all_scores = [], []
    ratio_encoder.eval()
    with torch.no_grad():
        for seq in id_sequences:
            for _, ratio, score in seq:
                ratio_t = torch.tensor([ratio], dtype=torch.float32, device=device).unsqueeze(0)  # (1,1)
                score_t = torch.tensor(score, dtype=torch.float32, device=device)  # scalar, shape ()
                feat    = ratio_encoder.get_feature(ratio_t)  # (1, d_model)
                # GP input = raw ratio concatenated with the learned embedding.
                feature = torch.cat([ratio_t, feat], dim=-1)  # (1, 1+d_model)
                all_features.append(feature.squeeze(0))
                all_scores.append(score_t)
    return TensorDataset(torch.stack(all_features), torch.stack(all_scores))
