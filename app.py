"""Streamlit GUI for the ML4OCRP pipeline.

Uses the modules in ``Model/`` so the workbench and the CLI share one
implementation. From the ML4OCRP root::

    streamlit run app.py

Tabs mirror the CLI tools: preprocess CSVs, train the deep-kernel GP,
recommend a ratio (GP mean + UCB), then query time-to-target per ID.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# Avoid Intel OpenMP duplicate-runtime abort on Windows (torch + numpy/MKL).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import torch

from preprocess_data import build_id_data_dict
from query_per_id import get_device
from ratio_time_predictor import RatioTimePredictor


def _load_gp_functions():
    """Import train/recommend helpers that require GPyTorch.

    :returns: Tuple of callables, or ``(None, None, None, None, error)``.
    :rtype: tuple
    """
    try:
        from ocrp_model_train import train_ocrp_model
        from predict_ratio_and_recommend_by_ucb import (
            load_trained_model,
            predict_optimal_ratio,
            ucb_acquisition,
        )
        return train_ocrp_model, load_trained_model, predict_optimal_ratio, ucb_acquisition, None
    except Exception as exc:
        return None, None, None, None, exc


STATUS_LABELS = {
    -1: "Reached in history",
    1: "Predicted in the future",
    0: "Not reached",
}

def _npy_bytes(obj) -> bytes:
    """Serialize a numpy object (including dicts) to bytes for download."""
    buf = io.BytesIO()
    np.save(buf, obj, allow_pickle=True)
    return buf.getvalue()


def _id_seq_dict(id_data_dict: dict) -> dict:
    """Convert ``{id: (times, ratios, scores)}`` to sequence triples."""
    return {
        id_: list(zip(times, ratios, scores))
        for id_, (times, ratios, scores) in id_data_dict.items()
    }


def _summary_frame(id_data_dict: dict) -> pd.DataFrame:
    """One-row-per-ID overview of trajectory length and ratio range."""
    rows = []
    for id_, (times, ratios, scores) in id_data_dict.items():
        rows.append({
            "id": id_,
            "n_points": len(times),
            "t_min": float(np.min(times)),
            "t_max": float(np.max(times)),
            "ratio_min": float(np.min(ratios)),
            "ratio_max": float(np.max(ratios)),
            "score_mean": float(np.mean(scores)),
        })
    return pd.DataFrame(rows)


def _ratio_score_frame(id_data_dict: dict) -> pd.DataFrame:
    """Every observation, across all IDs, as one ratio-score point."""
    frames = []
    for _, (_, ratios, scores) in id_data_dict.items():
        frames.append(pd.DataFrame({
            "ratio": np.asarray(ratios, dtype=float),
            "score": np.asarray(scores, dtype=float),
        }))
    if not frames:
        return pd.DataFrame(columns=["ratio", "score"])
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def _load_id_data_dict(path: str) -> dict:
    return np.load(path, allow_pickle=True).item()


def _evaluate_ratio_curve(model, likelihood, ratio_encoder, device,
                          ratio_low, ratio_high, n_candidates, beta) -> pd.DataFrame:
    """Evaluate GP mean / std / UCB on a ratio grid for plotting."""
    model.eval()
    likelihood.eval()
    candidates = np.linspace(ratio_low, ratio_high, n_candidates)
    means, stds, ucbs = [], [], []
    with torch.no_grad():
        for r in candidates:
            ratio_t = torch.tensor([[r]], dtype=torch.float32, device=device)
            feat = ratio_encoder.get_feature(ratio_t)
            x = torch.cat([ratio_t, feat], dim=-1)
            pred = model(x)
            mean = pred.mean.item()
            std = pred.stddev.item()
            means.append(mean)
            stds.append(std)
            ucbs.append(mean + beta * std)
    return pd.DataFrame({
        "ratio": candidates,
        "mean": means,
        "std": stds,
        "ucb": ucbs,
    })


def _init_state():
    defaults = {
        "id_data_dict": None,
        "data_path": "",
        "model_path": "",
        "recommend": None,
        "ratio_curve": None,
        "query_df": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_preprocess():
    """Tab: merge CSVs or load an existing ``id_data_dict.npy``."""
    st.subheader("1. Preprocess")
    st.caption(
        "Inner-join ratio and score tables on ``(id, time)``, then store "
        "aligned per-ID arrays."
    )

    source = st.radio(
        "Data source",
        ("Upload CSVs", "Load existing .npy", "Local CSV paths"),
        horizontal=True,
    )

    col_a, col_b = st.columns(2)
    id_data_dict = None
    save_dir = col_a.text_input(
        "Save directory",
        value=str(PROJECT_ROOT / "Data" / "Preprocessed"),
    )
    save_name = col_b.text_input("Save filename", value="id_data_dict.npy")

    merge_how = st.selectbox(
        "Merge on (id, time)",
        options=["inner", "left", "right", "outer"],
        index=0,
    )

    if source == "Upload CSVs":
        ratio_file = st.file_uploader("ratio CSV (columns: id, time, ratio)", type="csv")
        score_file = st.file_uploader("score CSV (columns: id, time, score)", type="csv")
        if st.button("Merge and save", type="primary") and ratio_file and score_file:
            ratio_df = pd.read_csv(ratio_file)
            score_df = pd.read_csv(score_file)
            id_data_dict = build_id_data_dict(ratio_df, score_df, merge_how=merge_how)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = str(Path(save_dir) / save_name)
            np.save(save_path, id_data_dict)
            st.session_state.id_data_dict = id_data_dict
            st.session_state.data_path = save_path
            st.success(f"Saved {len(id_data_dict)} IDs to {save_path}")
    elif source == "Load existing .npy":
        npy_path = st.text_input(
            "Path to id_data_dict.npy",
            value=str(PROJECT_ROOT / "Data" / "Preprocessed" / "id_data_dict.npy"),
        )
        uploaded_npy = st.file_uploader("Or upload a .npy file", type=["npy"])
        if st.button("Load dictionary", type="primary"):
            if uploaded_npy is not None:
                id_data_dict = np.load(uploaded_npy, allow_pickle=True).item()
                st.session_state.data_path = uploaded_npy.name
            else:
                id_data_dict = _load_id_data_dict(npy_path)
                st.session_state.data_path = npy_path
            st.session_state.id_data_dict = id_data_dict
            st.success(f"Loaded {len(id_data_dict)} IDs")
    else:
        load_dir = st.text_input(
            "Directory containing CSVs",
            value=str(PROJECT_ROOT / "Data" / "Original"),
        )
        ratio_name = st.text_input("Ratio CSV filename", value="ratio_data.csv")
        score_name = st.text_input("Score CSV filename", value="score_data.csv")
        if st.button("Merge local CSVs", type="primary"):
            ratio_df = pd.read_csv(Path(load_dir) / ratio_name)
            score_df = pd.read_csv(Path(load_dir) / score_name)
            id_data_dict = build_id_data_dict(ratio_df, score_df, merge_how=merge_how)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = str(Path(save_dir) / save_name)
            np.save(save_path, id_data_dict)
            st.session_state.id_data_dict = id_data_dict
            st.session_state.data_path = save_path
            st.success(f"Saved {len(id_data_dict)} IDs to {save_path}")

    data = st.session_state.id_data_dict
    if data:
        st.metric("IDs loaded", len(data))
        summary = _summary_frame(data)
        st.dataframe(summary, width="stretch")
        chart_df = _ratio_score_frame(data)
        st.scatter_chart(chart_df, x="ratio", y="score")
        st.download_button(
            "Download id_data_dict.npy",
            data=_npy_bytes(data),
            file_name="id_data_dict.npy",
        )


def render_train(device):
    """Tab: train encoder + DeepKernelGP from the loaded dictionary."""
    st.subheader("2. Train OCRP model")
    st.caption(
        "Pretrains the ratio encoder, then fits a deep-kernel GP on "
        "``[ratio | embedding] → score``."
    )

    if st.session_state.id_data_dict is None:
        st.info("Load or preprocess data in the first tab before training.")
        return

    train_ocrp_model, *_rest, gp_err = _load_gp_functions()
    if gp_err is not None:
        st.error(
            "Training needs GPyTorch (pip install gpytorch). "
            f"Import failed: {gp_err}"
        )
        return

    c1, c2, c3 = st.columns(3)
    d_model = c1.number_input("d_model", min_value=4, max_value=128, value=16, step=4)
    feat_dim = c2.number_input("d_feat", min_value=2, max_value=64, value=8, step=2)
    hidden_dim = c3.number_input("hidden_dim", min_value=4, max_value=256, value=32, step=4)
    c4, c5, c6 = st.columns(3)
    lr = c4.number_input("GP learning rate", min_value=1e-4, max_value=0.1, value=0.01, format="%.4f")
    n_epochs = c5.number_input("GP epochs", min_value=20, max_value=2000, value=200, step=20)
    val_ratio = c6.number_input("val_ratio", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
    c7, c8, c9 = st.columns(3)
    pretrain_epochs = c7.number_input("pretrain epochs", min_value=10, max_value=500, value=100, step=10)
    pretrain_lr = c8.number_input("pretrain learning rate", min_value=1e-5, max_value=0.1, value=0.001, format="%.4f")
    pretrain_batch_size = c9.number_input("pretrain batch size", min_value=8, max_value=4096, value=256, step=8)
    c10, c11 = st.columns(2)
    log_interval = c10.number_input("log interval", min_value=1, max_value=200, value=20, step=1)
    mll_reduction = c11.selectbox("MLL reduction", options=["sum", "mean"])

    save_path = st.text_input(
        "Checkpoint path",
        value=str(PROJECT_ROOT / "Data" / "Model" / "OCRP_model.pth"),
    )

    log_box = st.empty()
    logs = []

    def log_fn(msg):
        logs.append(str(msg))
        log_box.code("\n".join(logs[-12:]))

    if st.button("Start training", type="primary"):
        with st.spinner("Training — encoder pretrain then ExactGP. This can take a while."):
            save_file = train_ocrp_model(
                st.session_state.id_data_dict,
                save_path,
                d_model=int(d_model),
                feat_dim=int(feat_dim),
                hidden_dim=int(hidden_dim),
                lr=float(lr),
                n_epochs=int(n_epochs),
                val_ratio=float(val_ratio),
                pretrain_epochs=int(pretrain_epochs),
                pretrain_lr=float(pretrain_lr),
                pretrain_batch_size=int(pretrain_batch_size),
                log_interval=int(log_interval),
                mll_reduction=mll_reduction,
                device=device,
                log_fn=log_fn,
            )
        st.session_state.model_path = str(save_file)
        st.success(f"Checkpoint saved to {save_file}")

    existing = st.text_input(
        "Or point to an already-trained checkpoint",
        value=st.session_state.model_path or str(PROJECT_ROOT / "Data" / "Model" / "OCRP_model.pth"),
        key="existing_ckpt",
    )
    if st.button("Use this checkpoint"):
        if Path(existing).exists():
            st.session_state.model_path = existing
            st.success(f"Using {existing}")
        else:
            st.error(f"File not found: {existing}")


def render_recommend(device):
    """Tab: global optimal ratio (GP mean) and UCB next experiment."""
    st.subheader("3. Recommend ratio")
    st.caption(
        "Gradient ascent on the GP mean gives a global optimum; UCB "
        "(mean + β·std) recommends the next experimental ratio."
    )

    ckpt = st.text_input(
        "Trained model",
        value=st.session_state.model_path or str(PROJECT_ROOT / "Data" / "Model" / "OCRP_model.pth"),
        key="recommend_ckpt",
    )
    c1, c2, c3 = st.columns(3)
    opt_low = c1.number_input("Mean-search low", value=0.01, format="%.3f")
    opt_high = c2.number_input("Mean-search high", value=5.0, format="%.3f")
    n_starts = c3.number_input("Multi-starts", min_value=3, max_value=50, value=10)
    c4, c5, c6 = st.columns(3)
    n_steps = c4.number_input("Gradient steps", min_value=10, max_value=1000, value=100, step=10)
    opt_lr = c5.number_input("Gradient learning rate", min_value=1e-4, max_value=0.1, value=0.01, format="%.4f")
    beta = c6.number_input("UCB beta", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    c7, c8 = st.columns(2)
    ucb_low = c7.number_input("UCB low", value=0.1, format="%.3f")
    ucb_high = c8.number_input("UCB high", value=10.0, format="%.3f")
    n_candidates = st.slider("UCB / curve grid size", min_value=20, max_value=400, value=100, step=10)

    if st.button("Run recommendation", type="primary"):
        _, load_trained_model, predict_optimal_ratio, ucb_acquisition, gp_err = _load_gp_functions()
        if gp_err is not None:
            st.error(
                "Recommendation needs GPyTorch (pip install gpytorch). "
                f"Import failed: {gp_err}"
            )
            return
        if not Path(ckpt).exists():
            st.error(f"Checkpoint not found: {ckpt}")
            return
        with st.spinner("Loading model and searching the ratio landscape..."):
            ratio_encoder, model, likelihood = load_trained_model(ckpt, device)
            opt_ratio, max_score = predict_optimal_ratio(
                model, ratio_encoder, device,
                ratio_range=(float(opt_low), float(opt_high)),
                n_starts=int(n_starts),
                n_steps=int(n_steps),
                lr=float(opt_lr),
            )
            next_ratio, ucb_value = ucb_acquisition(
                model, likelihood, ratio_encoder, device,
                ratio_range=(float(ucb_low), float(ucb_high)),
                n_candidates=int(n_candidates),
                beta=float(beta),
            )
            curve = _evaluate_ratio_curve(
                model, likelihood, ratio_encoder, device,
                float(ucb_low), float(ucb_high), int(n_candidates), float(beta),
            )
        result = {
            "GLOBAL_OPTIMAL_RATIO": opt_ratio,
            "MAX_SCORE": max_score,
            "UCB_RECOMMEND": next_ratio,
            "UCB_SCORE": ucb_value,
        }
        st.session_state.recommend = result
        st.session_state.ratio_curve = curve
        st.session_state.model_path = ckpt

    rec = st.session_state.recommend
    if rec:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Global optimal ratio", f"{rec['GLOBAL_OPTIMAL_RATIO']:.4f}")
        m2.metric("Predicted score", f"{rec['MAX_SCORE']:.4f}")
        m3.metric("UCB recommended ratio", f"{rec['UCB_RECOMMEND']:.4f}")
        m4.metric("UCB value", f"{rec['UCB_SCORE']:.4f}")
        if st.session_state.ratio_curve is not None:
            st.line_chart(
                st.session_state.ratio_curve,
                x="ratio",
                y=["mean", "ucb"],
            )
        csv = pd.DataFrame([rec]).to_csv(index=False).encode("utf-8")
        st.download_button("Download predict_result.csv", data=csv, file_name="predict_result.csv")


def render_query(device):
    """Tab: per-ID time at which the target ratio is reached."""
    st.subheader("4. Query time per ID")
    st.caption(
        "Fit ODE + residual models per ID, then search for the time at which "
        "the global optimal ratio is reached."
    )

    if st.session_state.id_data_dict is None:
        st.info("Load data in tab 1 first.")
        return

    default_target = 0.5
    if st.session_state.recommend:
        default_target = float(st.session_state.recommend["GLOBAL_OPTIMAL_RATIO"])

    c1, c2, c3 = st.columns(3)
    target = c1.number_input("Target ratio", value=default_target, format="%.4f")
    max_time = c2.number_input("Max historical time", value=12.0, step=0.5)
    tolerance = c3.number_input("Tolerance", value=5e-3, format="%.4f")
    c4, c5, c6 = st.columns(3)
    model_type = c4.selectbox("Residual model", options=["gpr", "poly"])
    horizon = c5.number_input("Horizon factor", value=3.0, step=0.5)
    fit_ode = c6.checkbox("Fit ODE parameters per ID", value=False)
    c7, c8 = st.columns(2)
    search_iter = c7.number_input("Search iterations", min_value=1, max_value=100, value=20)
    n_segments = c8.number_input("Segments per iteration", min_value=2, max_value=100, value=10)
    st.caption(
        "ODE bounds, integrator tolerances, and the residual kernel use the "
        "same defaults as query_per_id.py."
    )

    if st.button("Run per-ID query", type="primary"):
        predictor = RatioTimePredictor(
            tolerance=float(tolerance),
            horizon_factor=float(horizon),
            model_type=model_type,
            fit_ode=bool(fit_ode),
            search_iter=int(search_iter),
            n_segments=int(n_segments),
            device=device,
        )
        id_seq_dict = _id_seq_dict(st.session_state.id_data_dict)
        n = len(id_seq_dict)
        progress = st.progress(0.0, text="Fitting IDs...")
        for i, (id_, seq) in enumerate(id_seq_dict.items(), start=1):
            times = np.array([x[0] for x in seq])
            ratios = np.array([x[1] for x in seq])
            predictor.fit_id(id_, times, ratios)
            progress.progress(i / (2 * n), text=f"Fitting {i}/{n}: {id_}")

        rows = []
        fitted = list(predictor.fitted_ids)
        for j, id_ in enumerate(fitted, start=1):
            found, t = predictor.query(id_, float(target), float(max_time))
            if found:
                status = -1 if t <= max_time else 1
            else:
                status = 0
                t = 0.0
            rows.append({
                "id": id_,
                "time": t,
                "status": status,
                "label": STATUS_LABELS[status],
            })
            progress.progress(0.5 + j / (2 * max(len(fitted), 1)), text=f"Querying {j}/{len(fitted)}: {id_}")
        progress.empty()
        st.session_state.query_df = pd.DataFrame(rows)

    df = st.session_state.query_df
    if df is not None and len(df):
        n1, n2, n3 = st.columns(3)
        n1.metric("Reached in history", int((df["status"] == -1).sum()))
        n2.metric("Predicted future", int((df["status"] == 1).sum()))
        n3.metric("Not reached", int((df["status"] == 0).sum()))
        st.dataframe(df, width="stretch")
        st.bar_chart(df["label"].value_counts())
        st.download_button(
            "Download results CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="predict_id_time.csv",
        )
        result_dict = {
            row["id"]: (row["time"], row["status"]) for _, row in df.iterrows()
        }
        st.download_button(
            "Download predict_id_time_dict.npy",
            data=_npy_bytes(result_dict),
            file_name="predict_id_time_dict.npy",
        )


def main():
    st.set_page_config(page_title="ML4OCRP", layout="wide")
    _init_state()

    st.title("ML4OCRP")
    st.write(
        "ShanghaiTech iGEM 2026: interactive workbench for optimal CRISPR "
        "ratio prediction. Same models as the CLI scripts, without the command line."
    )

    with st.sidebar:
        st.header("Runtime")
        device_arg = st.selectbox("Device", options=["auto", "cpu", "cuda", "mps"])
        device = get_device(None if device_arg == "auto" else device_arg)
        st.write(f"Using **{device}**")
        if st.session_state.data_path:
            st.caption(f"Data: `{st.session_state.data_path}`")
        if st.session_state.model_path:
            st.caption(f"Model: `{st.session_state.model_path}`")
        st.markdown(
            "Pipeline: **Preprocess → Train → Recommend → Query**."
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Preprocess", "Train", "Recommend", "Query IDs"]
    )
    with tab1:
        render_preprocess()
    with tab2:
        render_train(device)
    with tab3:
        render_recommend(device)
    with tab4:
        render_query(device)


if __name__ == "__main__":
    main()
