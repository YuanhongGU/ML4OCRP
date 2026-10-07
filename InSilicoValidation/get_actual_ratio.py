"""Integrate the BxB1 ODE at each predicted time and record the true ratio.

The ODE matches ``fitting_ratio.py``. Times come from ``predict_id_time_dict.npy``
and kinetic parameters from ``param_data.csv``.
"""

import os
import numpy as np
import pandas as pd
import argparse
from scipy.integrate import solve_ivp


def arm_rates(h_A, h_B, k_A0=0.38, k_B0=0.14, beta=1.3):
    """Power-law arm-length to cleavage rate mapping."""
    k_A = k_A0 * (h_A ** (-beta))
    k_B = k_B0 * (h_B ** (-beta))
    return k_A, k_B


def bxb1_ode(t, y, k_A, k_B, r_A=0.56, r_B=0.01, alpha=0.01):
    """Right-hand side of the five-state BxB1 branching circuit.

    States are D (founder), L_A (Venus+), L_B (mScarlet+), M_A, and M_B.
    """
    D, L_A, L_B, M_A, M_B = y
    v_A = alpha * k_A
    v_B = alpha * k_B

    dD   = -(k_A + k_B + r_A + r_B) * D
    dL_A =  k_A * D + v_A * M_A + k_A * M_B
    dL_B =  k_B * D + v_B * M_B + k_B * M_A
    dM_A =  r_A * D - v_A * M_A - k_B * M_A
    dM_B =  r_B * D - v_B * M_B - k_A * M_B
    return [dD, dL_A, dL_B, dM_A, dM_B]


def simulate_ode(h_A, h_B, t_target, k_A0=0.38, k_B0=0.14, beta=1.3,
                 r_A=0.56, r_B=0.01, alpha=0.01):
    """Return L_A/L_B at ``t_target`` hours.

    Undifferentiated states, where both product pools are empty, return 1.
    """
    k_A, k_B = arm_rates(h_A, h_B, k_A0, k_B0, beta)
    y0 = [1.0, 0.0, 0.0, 0.0, 0.0]
    t_span = (0, t_target)
    # Evaluate only the requested time so the reported ratio is exactly at t_target.
    sol = solve_ivp(bxb1_ode, t_span, y0, args=(k_A, k_B, r_A, r_B, alpha),
                    t_eval=[t_target], method='RK45', rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(f"ODE integration failed at t={t_target}")
    D, L_A, L_B, M_A, M_B = sol.y[:, -1]
    total_diff = L_A + L_B
    if total_diff < 1e-12:
        return 1.0
    frac_A = L_A / total_diff
    frac_B = L_B / total_diff
    return frac_A / frac_B


def parse_args():
    """Parse paths for the predicted times, the kinetic parameters, and the output."""
    parser = argparse.ArgumentParser(description="Compute actual ratio at predicted times using true parameters.")
    parser.add_argument("--pred_time_file", default="predict_id_time_dict.npy",
                        help="Input file name of predict_id_time_dict.npy")
    parser.add_argument("--param_file", default="param_data.csv",
                        help="Input file name of param_data.csv")
    parser.add_argument("--load_pred_dir", default="../Data/Predict",
                        help="Directory containing predict_id_time_dict.npy")
    parser.add_argument("--load_param_dir", default="../Data/Original",
                        help="Directory containing param_data.csv")
    parser.add_argument("-s", "--save_dir", default="../Data/Predict",
                        help="Directory to save output CSV")
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()

    pred_path = os.path.join(args.load_pred_dir, args.pred_time_file)
    id_time_dict = np.load(pred_path, allow_pickle=True).item()
    # Each entry is {id: [time, flag]}. Flag is -1 (already seen), 1 (future), or 0 (not found).

    param_path = os.path.join(args.load_param_dir, args.param_file)
    param_df = pd.read_csv(param_path)
    # Missing kinetic columns fall back to the same defaults as fitting_ratio.py.
    default_params = {
        'k_A0': 0.38, 'k_B0': 0.14, 'beta': 1.3,
        'r_A': 0.56, 'r_B': 0.01, 'alpha': 0.01
    }
    for col, val in default_params.items():
        if col not in param_df.columns:
            param_df[col] = val

    results = []
    for id_, (time, flag) in id_time_dict.items():
        if time == 0:  # Flag 0 stores time 0 when the target was not reached.
            results.append({'id': id_, 'predicted_time': time, 'flag': flag,
                            'actual_ratio': np.nan, 'status': 'unreachable'})
            continue
        row = param_df[param_df['id'] == id_]
        if row.empty:
            print(f"Warning: ID {id_} not found in param_data.csv, skipping.")
            results.append({'id': id_, 'predicted_time': time, 'flag': flag,
                            'actual_ratio': np.nan, 'status': 'no_param'})
            continue
        params = row.iloc[0]
        try:
            actual_ratio = simulate_ode(
                h_A=params['h_A'], h_B=params['h_B'],
                t_target=time,
                k_A0=params['k_A0'], k_B0=params['k_B0'],
                beta=params['beta'],
                r_A=params['r_A'], r_B=params['r_B'],
                alpha=params['alpha']
            )
            status = 'success'
        except Exception as e:
            print(f"Error for ID {id_} at time {time}: {e}")
            actual_ratio = np.nan
            status = 'ode_failed'
        results.append({
            'id': id_,
            'predicted_time': time,
            'flag': flag,
            'actual_ratio': actual_ratio,
            'status': status
        })

    out_df = pd.DataFrame(results)
    out_path = os.path.join(args.save_dir, "actual_ratios_at_predicted_times.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved actual ratios to {out_path}")
