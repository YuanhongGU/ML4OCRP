"""Query, for each ID, the time at which the global optimal ratio is reached.

Loads ``predict_result.csv`` (from UCB / GP optimization) and the preprocessed
``id_data_dict``, fits a ``RatioTimePredictor`` per ID in parallel, then queries
the target ratio in parallel. Result status codes:

* ``-1``: target already reached in historical data
* ``1``: target predicted at a future time
* ``0``: target not reached within the search horizon
"""

import os
import numpy as np
import pandas as pd
import argparse
from ratio_time_predictor import (
    RatioTimePredictor,
    add_predictor_arguments,
    predictor_kwargs_from_args,
)
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def _resolve_workers(max_workers):
    """Cap the process pool at the number of CPUs.

    :param max_workers: Requested worker count.
    :type max_workers: int
    :returns: A worker count of at least 1 and at most the CPU count.
    :rtype: int
    """
    cpu_count = multiprocessing.cpu_count() or 1
    return max(1, min(int(max_workers), cpu_count))


def parse_arg():
    """Parse paths, search settings, and ODE / residual-model hyperparameters.

    ODE bounds, kinetic defaults, integrator tolerances, and residual-model
    settings are registered by ``add_predictor_arguments``. Their defaults are
    the values previously hard-coded in ``ratio_time_predictor``.

    :returns: Parsed CLI namespace.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Query time for target ratio per ID")

    parser.add_argument("--pred_res",       required=False, type=str,   default="predict_result.csv",
                        help="Filename of the GP/UCB prediction CSV.")
    parser.add_argument("-l","--loaddir",   required=False, type=str,   default="../Data/Preprocessed",
                        help="Directory that contains id_data_dict.npy.")
    parser.add_argument("--loadpred",       required=False, type=str,   default="../Data/Predict",
                        help="Directory that contains the prediction CSV.")
    parser.add_argument("-s", "--savedir",  required=False, type=str,   default="../Data/Predict",
                        help="Directory in which to write predict_id_time_dict.npy.")
    parser.add_argument("--save_filename",  required=False, type=str,   default="predict_id_time_dict.npy",
                        help="Filename of the per-ID query result.")
    parser.add_argument("--tolerance",      required=False, type=float, default=5e-3,
                        help="Absolute tolerance for a historical ratio match.")
    parser.add_argument("--horizon_factor", required=False, type=float, default=3.0,
                        help="Search upper bound = max_time * horizon_factor.")
    parser.add_argument("--search_iter",    required=False, type=int,   default=20,
                        help="Number of nested interval-search iterations.")
    parser.add_argument("--n_segments",     required=False, type=int,   default=10,
                        help="Number of segments evaluated in each search iteration.")
    parser.add_argument("--max_time",       required=False, type=float, default=12.0,
                        help="Maximum historical time used to scale the search horizon.")
    parser.add_argument("--max_workers",    required=False, type=int,   default=8,
                        help="Process-pool size, capped by the CPU count.")
    parser.add_argument("--device",         required=False, type=str,   default=None,
                        help="Device to use (cuda/mps/cpu). Auto-detect if omitted.")
    parser.add_argument("--model_type",     required=False, type=str,   default="gpr", choices=["gpr", "poly"],
                        help="Residual model type.")
    parser.add_argument("--fit_ode",        required=False, action="store_true",
                        help="Fit ODE parameters per ID. If omitted, use the default kinetic parameters.")
    add_predictor_arguments(parser)

    return parser.parse_args()

def get_device(device_arg):
    """Resolve a torch device from a CLI string or from hardware availability.

    :param device_arg: Explicit device name, or ``None`` to auto-detect
        (MPS, then CUDA, then CPU).
    :type device_arg: str or None
    :returns: Selected device.
    :rtype: torch.device
    """
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _fit_id_worker(job):
    """Process-pool worker: fit one ID and return its model artifacts.

    A fresh ``RatioTimePredictor`` is created in the child process because
    sklearn / scipy objects should not be shared across processes.

    :param job: Tuple ``(id_, times, ratios, predictor_kwargs)``.
    :type job: tuple
    :returns: ``(id_, model_entry, hist_data)``. Model/hist may be ``None``
        if fitting was skipped (e.g. too few points).
    :rtype: tuple
    """
    id_, times, ratios, predictor_kwargs = job
    predictor = RatioTimePredictor(**predictor_kwargs)
    predictor.fit_id(id_, times, ratios)
    return id_, predictor.models.get(id_), predictor.hist_data.get(id_)


def fit_ids_parallel(predictor, id_seq_dict, max_workers=8):
    """Fit residual/ODE models for all IDs using a process pool.

    Fitted ``models`` and ``hist_data`` are written back onto ``predictor``.

    :param predictor: Template predictor whose hyperparameters are copied into
        each worker.
    :type predictor: ratio_time_predictor.RatioTimePredictor
    :param id_seq_dict: ``{id: [(t, ratio, score), ...]}``.
    :type id_seq_dict: dict
    :param max_workers: Process count. Defaults to 8 and is capped by the CPU count.
        Override from the CLI with ``--max_workers``.
    :type max_workers: int
    :returns: The same object, with ``models`` / ``hist_data`` /
        ``fitted_ids`` populated.
    :rtype: ratio_time_predictor.RatioTimePredictor
    """
    max_workers = _resolve_workers(max_workers)
    predictor_kwargs = predictor.export_kwargs()
    jobs = []
    for id_, seq in id_seq_dict.items():
        times = np.array([x[0] for x in seq])
        ratios = np.array([x[1] for x in seq])
        jobs.append((id_, times, ratios, predictor_kwargs))

    fitted_models = {}
    fitted_hist = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {executor.submit(_fit_id_worker, job): job[0] for job in jobs}
        for future in tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Fitting IDs"):
            id_ = future_to_id[future]
            try:
                id_ret, model_obj, hist_data = future.result()
                if model_obj is not None and hist_data is not None:
                    fitted_models[id_ret] = model_obj
                    fitted_hist[id_ret] = hist_data
                else:
                    print(f"ID {id_ret} failed to fit or returned no model.")
            except Exception as exc:
                print(f"ID {id_} fit generated an exception: {exc}")

    predictor.models = fitted_models
    predictor.hist_data = fitted_hist
    predictor.fitted_ids = set(fitted_models.keys())
    print(f"Fitted {len(predictor.fitted_ids)} IDs.")
    return predictor


def _query_id_worker(job):
    """Process-pool worker: reconstruct one ID's predictor and run ``query``.

    :param job: Tuple ``(id_, target_ratio, max_time, predictor_kwargs,
        model_obj, hist_data)``.
    :type job: tuple
    :returns: ``(id_, (found, time))`` from ``RatioTimePredictor.query``.
    :rtype: tuple
    """
    id_, target_ratio, max_time, predictor_kwargs, model_obj, hist_data = job
    predictor = RatioTimePredictor(**predictor_kwargs)
    predictor.models[id_] = model_obj
    predictor.hist_data[id_] = hist_data
    predictor.fitted_ids.add(id_)
    return id_, predictor.query(id_, target_ratio, max_time)


def query_ids_parallel(predictor, id_list, target_ratio, max_time, max_workers=8):
    """Query time-to-target-ratio for many IDs in parallel.

    Status encoding written into the returned dict:

    * ``-1`` if ``found`` and ``time <= max_time`` (historical)
    * ``1`` if ``found`` and ``time > max_time`` (future prediction)
    * ``0`` if not found

    :param predictor: Fitted predictor providing per-ID models and history.
    :type predictor: ratio_time_predictor.RatioTimePredictor
    :param id_list: IDs to query; IDs without a fitted model are skipped.
    :type id_list: list
    :param target_ratio: Global optimal ratio from the GP/UCB stage.
    :type target_ratio: float
    :param max_time: Historical time cutoff used for status and search bound.
    :type max_time: float
    :param max_workers: Process count. Defaults to 8 and is capped by the CPU count.
        Override from the CLI with ``--max_workers``.
    :type max_workers: int
    :returns: ``{id: (time, status)}``.
    :rtype: dict
    """
    max_workers = _resolve_workers(max_workers)
    predictor_kwargs = predictor.export_kwargs()
    jobs = []
    for id_ in id_list:
        model_obj = predictor.models.get(id_)
        hist_data = predictor.hist_data.get(id_)
        if model_obj is None or hist_data is None:
            continue
        jobs.append((id_, target_ratio, max_time, predictor_kwargs, model_obj, hist_data))

    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {executor.submit(_query_id_worker, job): job[0] for job in jobs}
        for future in tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Querying IDs"):
            id_ = future_to_id[future]
            try:
                id_ret, query_result = future.result()
                found, time = query_result
                if found:
                    status = -1 if time <= max_time else 1
                else:
                    status = 0
                    time = 0.0
                results[id_ret] = (time, status)
            except Exception as exc:
                print(f"ID {id_} query generated an exception: {exc}")
    return results

if __name__ == "__main__":
    args = parse_arg()
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load GP/UCB prediction (global optimal ratio is the query target).
    pred_res_path = os.path.join(args.loadpred, args.pred_res)
    print(f"Loading prediction results from {pred_res_path}...")
    pred_res = pd.read_csv(pred_res_path)
    GLOBAL_OPTIMAL_RATIO = pred_res["GLOBAL_OPTIMAL_RATIO"].values[0]
    print(f"Global optimal ratio: {GLOBAL_OPTIMAL_RATIO:.4f}")

    # Load per-ID observed trajectories.
    data_path = os.path.join(args.loaddir, "id_data_dict.npy")
    print(f"Loading id_data_dict from {data_path}...")
    id_data_dict = np.load(data_path, allow_pickle=True).item()
    id_list = list(id_data_dict.keys())
    print(f"Loaded {len(id_list)} IDs.")

    # Rebuild {id: [(t, ratio, score), ...]} for the predictor API.
    id_seq_dict = {}
    for id_, (times, ratios, scores) in id_data_dict.items():
        id_seq_dict[id_] = list(zip(times, ratios, scores))

    # Initialize predictor from CLI defaults (ODE, solver, and residual model).
    print("Initializing predictor...")
    predictor = RatioTimePredictor(**predictor_kwargs_from_args(args, device))

    # Fit each cell group in parallel.
    print("Fitting models for each ID in parallel...")
    predictor = fit_ids_parallel(predictor, id_seq_dict, max_workers=args.max_workers)

    # Query each cell group in parallel.
    print("Querying each ID for target ratio in parallel...")
    results = query_ids_parallel(
        predictor, id_list, GLOBAL_OPTIMAL_RATIO, args.max_time, max_workers=args.max_workers,
    )
    # Print a status for every cell group.
    print("\n=== Query Results Summary ===")
    for id_, (t, status) in results.items():
        if status == -1:
            print(f"ID {id_}: reached at historical time {t:.3f}")
        elif status == 1:
            print(f"ID {id_}: predicted at future time {t:.3f}")
        else:
            print(f"ID {id_}: failed to reach")

    # Save the per-id query results.
    save_path = os.path.join(args.savedir, args.save_filename)
    print(f"Saving results to {save_path}...")
    np.save(save_path, results)
    print("Done.")
