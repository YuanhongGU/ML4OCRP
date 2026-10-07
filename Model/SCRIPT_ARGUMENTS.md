# CLI arguments for `Model/`

Each table is one runnable script. `ocrp_model.py`, `model_train_functions.py`, and `ratio_time_predictor.py` are libraries. Their settings are exposed on the script that calls them. Every script also accepts `-h` / `--help`.

`query_per_id.py` includes the flags registered by `ratio_time_predictor.add_predictor_arguments`. Where a help string contains `%(default)s`, the table shows the value argparse prints.

## `preprocess_data.py`

Merge `ratio_data.csv` and `score_data.csv` into `id_data_dict.npy`.

| Argument | Type | Default | Help |
| --- | --- | --- | --- |
| `-l`, `--load_dir` | str | `../Data/Original` | Directory that contains the raw ratio and score CSVs. |
| `-s`, `--save_dir` | str | `../Data/Preprocessed` | Directory in which to write the preprocessed dictionary. |
| `-n`, `--save_filename` | str | `id_data_dict.npy` | Filename of the saved id dictionary. |
| `--ratio_file` | str | `ratio_data.csv` | Filename of the ratio table inside --load_dir. |
| `--score_file` | str | `score_data.csv` | Filename of the score table inside --load_dir. |
| `--merge_how` | str, choices: `inner`, `left`, `right`, `outer` | `inner` | Join type used when merging on (id, time). |

## `ocrp_model_train.py`

Pretrain the ratio encoder, then train the deep-kernel Gaussian process. Writes `<save_dir>/OCRP_model.pth`.

| Argument | Type | Default | Help |
| --- | --- | --- | --- |
| `-l`, `--loadfile` | str | `../Data/Preprocessed/id_data_dict.npy` | Path to id_data_dict.npy. |
| `-s`, `--save_dir` | str | `../Data/Model` | Directory in which to write OCRP_model.pth. |
| `--d_model` | int | `16` | Ratio-encoder embedding size. |
| `--d_feat` | int | `8` | Latent size of the GP feature extractor. |
| `--hidden_dim` | int | `32` | Hidden width of the ratio encoder and the GP feature MLP. |
| `--lr` | float | `0.01` | Adam learning rate for GP marginal-likelihood training. |
| `--n_epochs` | int | `200` | Number of GP training epochs. |
| `--pretrain_epochs` | int | `100` | Number of ratio-encoder pretraining epochs. |
| `--pretrain_lr` | float | `0.001` | Adam learning rate for ratio-encoder pretraining. |
| `--pretrain_batch_size` | int | `256` | Mini-batch size for ratio-encoder pretraining. |
| `--val_ratio` | float | `0.2` | Fraction of pooled samples held out for validation loss. |
| `--log_interval` | int | `20` | Print pretrain and GP loss every this many epochs. |
| `--mll_reduction` | str, choices: `sum`, `mean` | `sum` | How to reduce a non-scalar marginal log-likelihood. |
| `--device` | str | `None` | Torch device (mps, cuda, or cpu). Auto-detect if omitted. |
| `--seed` | int | `None` | Seed for torch and numpy. Omit to keep the unseeded training split. |

## `predict_ratio_and_recommend_by_ucb.py`

Load a trained checkpoint, maximize the GP mean, and recommend the next ratio by UCB.

| Argument | Type | Default | Help |
| --- | --- | --- | --- |
| `--trained_model` | str | `OCRP_model.pth` | Checkpoint filename inside --loadmodel. |
| `--loaddir` | str | `../Data/Preprocessed` | Fallback output directory used when --savedir is empty. |
| `--loadmodel` | str | `../Data/Model` | Directory that contains the trained checkpoint. |
| `-s`, `--savedir` | str | `../Data/Predict` | Directory in which to write the prediction CSV. |
| `--save_filename` | str | `predict_result.csv` | Filename of the prediction CSV. |
| `--device` | str | `None` | Torch device (cuda or cpu). CUDA is used when available if omitted. |
| `--opt_ratio_min` | float | `0.01` | Lower bound of the gradient search for the optimal ratio. |
| `--opt_ratio_max` | float | `5.0` | Upper bound of the gradient search for the optimal ratio. |
| `--n_starts` | int | `10` | Number of equally spaced starts for gradient ascent. |
| `--n_steps` | int | `100` | Adam steps per gradient-ascent start. |
| `--opt_lr` | float | `0.01` | Adam learning rate on the scalar ratio during gradient ascent. |
| `--ucb_ratio_min` | float | `0.1` | Lower bound of the UCB candidate grid. |
| `--ucb_ratio_max` | float | `10.0` | Upper bound of the UCB candidate grid. |
| `--n_candidates` | int | `100` | Number of grid points evaluated by UCB. |
| `--beta` | float | `2.0` | UCB exploration coefficient (mean + beta * std). |

## `query_per_id.py`

Fit a per-ID ODE plus residual model and query the time at which the global optimal ratio is reached. `--fit_ode` is a flag: omitting it leaves the default kinetic parameters in place.

| Argument | Type | Default | Help |
| --- | --- | --- | --- |
| `--pred_res` | str | `predict_result.csv` | Filename of the GP/UCB prediction CSV. |
| `-l`, `--loaddir` | str | `../Data/Preprocessed` | Directory that contains id_data_dict.npy. |
| `--loadpred` | str | `../Data/Predict` | Directory that contains the prediction CSV. |
| `-s`, `--savedir` | str | `../Data/Predict` | Directory in which to write predict_id_time_dict.npy. |
| `--save_filename` | str | `predict_id_time_dict.npy` | Filename of the per-ID query result. |
| `--tolerance` | float | `5e-3` | Absolute tolerance for a historical ratio match. |
| `--horizon_factor` | float | `3.0` | Search upper bound = max_time * horizon_factor. |
| `--search_iter` | int | `20` | Number of nested interval-search iterations. |
| `--n_segments` | int | `10` | Number of segments evaluated in each search iteration. |
| `--max_time` | float | `12.0` | Maximum historical time used to scale the search horizon. |
| `--max_workers` | int | `8` | Process-pool size, capped by the CPU count. |
| `--device` | str | `None` | Device to use (cuda/mps/cpu). Auto-detect if omitted. |
| `--model_type` | str, choices: `gpr`, `poly` | `gpr` | Residual model type. |
| `--fit_ode` | flag (`store_true`) | `False` | Fit ODE parameters per ID. If omitted, use the default kinetic parameters. |
| `--h_a` | float | `773.0` | Default homology-arm length for type A (default: 773.0). |
| `--h_b` | float | `773.0` | Default homology-arm length for type B (default: 773.0). |
| `--k_a0` | float | `0.38` | Default baseline recombination rate for arm A (default: 0.38). |
| `--k_b0` | float | `0.14` | Default baseline recombination rate for arm B (default: 0.14). |
| `--beta` | float | `1.3` | Default homology-length exponent (default: 1.3). |
| `--r_a` | float | `0.56` | Default rate from D into intermediate M_A (default: 0.56). |
| `--r_b` | float | `0.01` | Default rate from D into intermediate M_B (default: 0.01). |
| `--alpha` | float | `0.01` | Default scale from forward rate k to reverse rate v (default: 0.01). |
| `--h_a_min` | float | `300` | Lower search bound for h_A (default: 300). |
| `--h_a_max` | float | `6000` | Upper search bound for h_A (default: 6000). |
| `--h_b_min` | float | `300` | Lower search bound for h_B (default: 300). |
| `--h_b_max` | float | `6000` | Upper search bound for h_B (default: 6000). |
| `--k_a0_min` | float | `0.2` | Lower search bound for k_A0 (default: 0.2). |
| `--k_a0_max` | float | `0.6` | Upper search bound for k_A0 (default: 0.6). |
| `--k_b0_min` | float | `0.05` | Lower search bound for k_B0 (default: 0.05). |
| `--k_b0_max` | float | `0.25` | Upper search bound for k_B0 (default: 0.25). |
| `--beta_min` | float | `1.0` | Lower search bound for beta (default: 1.0). |
| `--beta_max` | float | `1.8` | Upper search bound for beta (default: 1.8). |
| `--r_a_min` | float | `0.1` | Lower search bound for r_A (default: 0.1). |
| `--r_a_max` | float | `0.6` | Upper search bound for r_A (default: 0.6). |
| `--r_b_min` | float | `0.001` | Lower search bound for r_B (default: 0.001). |
| `--r_b_max` | float | `0.3` | Upper search bound for r_B (default: 0.3). |
| `--alpha_min` | float | `0.001` | Lower search bound for alpha (default: 0.001). |
| `--alpha_max` | float | `0.1` | Upper search bound for alpha (default: 0.1). |
| `--de_maxiter` | int | `50` | Differential-evolution generations (default: 50). |
| `--de_popsize` | int | `15` | Differential-evolution population multiplier (default: 15). |
| `--de_seed` | int | `42` | Random seed for differential evolution (default: 42). |
| `--local_method` | str | `L-BFGS-B` | Local optimizer used when differential evolution fails. |
| `--ode_fail_loss` | float | `1e10` | Penalty loss when an ODE candidate raises (default: 1e10). |
| `--ivp_method` | str | `RK45` | scipy.integrate.solve_ivp method name. |
| `--ivp_rtol` | float | `1e-8` | Relative tolerance of the ODE integrator (default: 1e-8). |
| `--ivp_atol` | float | `1e-10` | Absolute tolerance of the ODE integrator (default: 1e-10). |
| `--product_eps` | float | `1e-12` | Product mass below this is treated as undifferentiated (default: 1e-12). |
| `--init_ratio` | float | `1.0` | Ratio returned when t <= 0 or integration fails (default: 1.0). |
| `--gpr_alpha` | float | `0.0` | Diagonal noise of the residual Gaussian process (default: 0.0). |
| `--gpr_restarts` | int | `10` | Kernel-optimizer restarts for the residual GP (default: 10). |
| `--gpr_seed` | int | `42` | Random seed of the residual GP (default: 42). |
| `--poly_degree` | int | `2` | Polynomial degree when --model_type is poly (default: 2). |
| `--min_points` | int | `3` | Minimum time points required to fit an ID (default: 3). |
| `--kernel_const` | float | `1.0` | Scale of each ConstantKernel factor (default: 1.0). |
| `--matern_length_scale` | float | `1.0` | Initial Matern length scale (default: 1.0). |
| `--matern_nu` | float | `2.5` | Matern smoothness parameter (default: 2.5). |
| `--dot_sigma` | float | `1.0` | DotProduct sigma_0 (default: 1.0). |
| `--white_noise` | float | `0.1` | Initial white-kernel noise level (default: 0.1). |
| `--white_noise_lower` | float | `1e-3` | Lower bound of the white-kernel noise (default: 1e-3). |
| `--white_noise_upper` | float | `10.0` | Upper bound of the white-kernel noise (default: 10.0). |
| `--query_tol_factor` | float | `10.0` | Accept a searched time when the ratio error is within this many tolerances. |
| `--time_atol` | float | `1e-6` | Stop the nested time search when the interval is narrower than this. |
| `--ratio_floor` | float | `0.0` | Lowest ratio returned by prediction (default: 0.0). |
