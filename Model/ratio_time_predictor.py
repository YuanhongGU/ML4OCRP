"""Per-ID ratio-vs-time predictor: mechanistic ODE plus a residual model.

The BxB1 recombination ODE supplies the main trajectory. Residuals against
observed ratios are then fit with a Gaussian process or a quadratic polynomial.
``query`` searches historical data first, then extrapolates with a nested
interval search until the predicted ratio crosses the target.

Kinetic defaults, solver tolerances, residual-model settings, and search
tolerances are keyword arguments. ``add_predictor_arguments`` registers the
same values as optional CLI flags for ``query_per_id.py``.
"""

import numpy as np
import torch
import warnings
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


def arm_rates(h_A, h_B, k_A0=0.38, k_B0=0.14, beta=1.3):
    """Compute length-dependent recombination rates for arms A and B.

    Rates follow a power-law decay in homology-arm length:
    ``k = k0 * h ** (-beta)``.

    :param h_A: Homology-arm length for type A.
    :type h_A: float
    :param h_B: Homology-arm length for type B.
    :type h_B: float
    :param k_A0: Baseline rate for A. Defaults to 0.38.
    :type k_A0: float
    :param k_B0: Baseline rate for B. Defaults to 0.14.
    :type k_B0: float
    :param beta: Length exponent. Defaults to 1.3.
    :type beta: float
    :returns: ``(k_A, k_B)`` instantaneous recombination rates.
    :rtype: tuple
    """
    k_A = k_A0 * (h_A ** (-beta))
    k_B = k_B0 * (h_B ** (-beta))
    return k_A, k_B


def bxb1_ode(t, y, k_A, k_B, r_A=0.56, r_B=0.01, alpha=0.01):
    """Right-hand side of the five-state BxB1 recombination ODE.

    States are ``D`` (unrecombined), ``L_A`` / ``L_B`` (resolved products),
    and ``M_A`` / ``M_B`` (excision intermediates). ``v_* = alpha * k_*``
    are reverse rates from intermediates back to products.

    :param t: Time (unused; the system is autonomous).
    :type t: float
    :param y: State vector ``[D, L_A, L_B, M_A, M_B]``.
    :type y: sequence
    :param k_A: Forward recombination rate toward product A.
    :type k_A: float
    :param k_B: Forward recombination rate toward product B.
    :type k_B: float
    :param r_A: Rate from D into intermediate M_A. Defaults to 0.56.
    :type r_A: float
    :param r_B: Rate from D into intermediate M_B. Defaults to 0.01.
    :type r_B: float
    :param alpha: Scaling from k to reverse rate v. Defaults to 0.01.
    :type alpha: float
    :returns: Time derivatives ``[dD, dL_A, dL_B, dM_A, dM_B]``.
    :rtype: list
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


def simulate_ode_ratio(h_A, h_B, t, k_A0, k_B0, beta, r_A, r_B, alpha,
                       method="RK45", rtol=1e-8, atol=1e-10,
                       product_eps=1e-12, init_ratio=1.0):
    """Integrate the ODE to time ``t`` and return the type-A (Venus+) fraction.

    The fraction is ``L_A / (L_A + L_B)``. If integration fails or almost no
    product has formed, the function returns ``init_ratio`` (the t=0 convention
    that all mass is still in D).

    Solver settings match ``--ivp_method``, ``--ivp_rtol``, ``--ivp_atol``,
    ``--product_eps``, and ``--init_ratio``.

    :param h_A: Homology-arm length A.
    :type h_A: float
    :param h_B: Homology-arm length B.
    :type h_B: float
    :param t: Integration endpoint. Values ``<= 0`` return ``init_ratio``.
    :type t: float
    :param k_A0: Baseline recombination rate for arm A.
    :type k_A0: float
    :param k_B0: Baseline recombination rate for arm B.
    :type k_B0: float
    :param beta: Length exponent in the power-law rate model.
    :type beta: float
    :param r_A: Rate from D into intermediate M_A.
    :type r_A: float
    :param r_B: Rate from D into intermediate M_B.
    :type r_B: float
    :param alpha: Scaling from k to reverse rate v.
    :type alpha: float
    :param method: ``solve_ivp`` method name. Defaults to ``"RK45"``.
    :type method: str
    :param rtol: Relative tolerance passed to ``solve_ivp``. Defaults to 1e-8.
    :type rtol: float
    :param atol: Absolute tolerance passed to ``solve_ivp``. Defaults to 1e-10.
    :type atol: float
    :param product_eps: Product mass below this is treated as "not yet differentiated".
    :type product_eps: float
    :param init_ratio: Ratio returned at t <= 0 or when integration fails.
    :type init_ratio: float
    :returns: Predicted Venus+ (type-A) ratio.
    :rtype: float
    """
    if t <= 0:
        return init_ratio  # Initial state: all mass in D.
    k_A, k_B = arm_rates(h_A, h_B, k_A0, k_B0, beta)
    y0 = [1.0, 0.0, 0.0, 0.0, 0.0]
    sol = solve_ivp(bxb1_ode, (0, t), y0, args=(k_A, k_B, r_A, r_B, alpha),
                    t_eval=[t], method=method, rtol=rtol, atol=atol)
    if not sol.success or sol.y.shape[1] == 0:
        return init_ratio
    D, L_A, L_B, M_A, M_B = sol.y[:, -1]
    total_diff = L_A + L_B
    if total_diff < product_eps:
        return init_ratio
    return L_A / total_diff  # Venus+ (type-A) product fraction


# Search bounds used when fitting ODE parameters from observed trajectories.
PARAM_BOUNDS = {
    'h_A': (300, 6000),
    'h_B': (300, 6000),
    'k_A0': (0.2, 0.6),
    'k_B0': (0.05, 0.25),
    'beta': (1.0, 1.8),
    'r_A': (0.1, 0.6),
    'r_B': (0.001, 0.3),
    'alpha': (0.001, 0.1)
}
# Fallback / initialization values if global search fails.
DEFAULT_PARAMS = {
    'h_A': 773.0,
    'h_B': 773.0,
    'k_A0': 0.38,
    'k_B0': 0.14,
    'beta': 1.3,
    'r_A': 0.56,
    'r_B': 0.01,
    'alpha': 0.01
}


def fit_ode_params(times, ratios, bounds=None, default=None,
                   de_maxiter=50, de_popsize=15, de_seed=42,
                   local_method="L-BFGS-B", ode_fail_loss=1e10,
                   ivp_method="RK45", ivp_rtol=1e-8, ivp_atol=1e-10,
                   product_eps=1e-12, init_ratio=1.0):
    """Fit ODE kinetic parameters by minimizing MSE vs. observed ratios.

    Differential evolution is used first for a global search. If it does not
    succeed, bounded local refinement from the default vector is tried.
    Failure of both returns ``default``.

    Optimizer and solver defaults match the ``--de_*``, ``--local_method``,
    ``--ode_fail_loss``, and ``--ivp_*`` flags.

    :param times: 1-D array of observation times.
    :type times: numpy.ndarray
    :param ratios: 1-D array of observed type-A ratios, same length as ``times``.
    :type ratios: numpy.ndarray
    :param bounds: Per-parameter ``(low, high)`` dict. Defaults to ``PARAM_BOUNDS``.
    :type bounds: dict or None
    :param default: Parameter dict used as init / fallback. Defaults to
        ``DEFAULT_PARAMS``.
    :type default: dict or None
    :param de_maxiter: Differential-evolution generations. Defaults to 50.
    :type de_maxiter: int
    :param de_popsize: Differential-evolution population multiplier. Defaults to 15.
    :type de_popsize: int
    :param de_seed: Random seed for differential evolution. Defaults to 42.
    :type de_seed: int
    :param local_method: ``scipy.optimize.minimize`` method used as fallback.
    :type local_method: str
    :param ode_fail_loss: Loss returned when a candidate ODE evaluation raises.
    :type ode_fail_loss: float
    :param ivp_method: Integrator name forwarded to ``simulate_ode_ratio``.
    :type ivp_method: str
    :param ivp_rtol: Relative integrator tolerance.
    :type ivp_rtol: float
    :param ivp_atol: Absolute integrator tolerance.
    :type ivp_atol: float
    :param product_eps: Minimum product mass treated as differentiated.
    :type product_eps: float
    :param init_ratio: Fallback ratio when the ODE cannot be integrated.
    :type init_ratio: float
    :returns: Fitted values keyed by parameter name.
    :rtype: dict
    """
    if bounds is None:
        bounds = PARAM_BOUNDS
    if default is None:
        default = DEFAULT_PARAMS

    def loss(x):
        """Mean squared error between simulated and observed ratios."""
        h_A, h_B, k_A0, k_B0, beta, r_A, r_B, alpha = x
        try:
            preds = np.array([
                simulate_ode_ratio(
                    h_A, h_B, t, k_A0, k_B0, beta, r_A, r_B, alpha,
                    method=ivp_method, rtol=ivp_rtol, atol=ivp_atol,
                    product_eps=product_eps, init_ratio=init_ratio,
                )
                for t in times
            ])
            return np.mean((preds - ratios) ** 2)
        except Exception:
            return ode_fail_loss

    # Start the local search from the default parameter vector.
    init = [default[p] for p in ['h_A','h_B','k_A0','k_B0','beta','r_A','r_B','alpha']]
    bounds_list = [bounds[p] for p in ['h_A','h_B','k_A0','k_B0','beta','r_A','r_B','alpha']]

    # Search globally with differential evolution.
    res = differential_evolution(
        loss, bounds_list, maxiter=de_maxiter, popsize=de_popsize, seed=de_seed,
    )
    if not res.success:
        # Fall back to bounded local search from the default vector.
        res = minimize(loss, init, bounds=bounds_list, method=local_method)
    if res.success:
        x_opt = res.x
    else:
        x_opt = init
    param_names = ['h_A','h_B','k_A0','k_B0','beta','r_A','r_B','alpha']
    return dict(zip(param_names, x_opt))


class RatioTimePredictor:
    """Fit per-ID ODE + residual models and query time-to-target-ratio.

    For each ID the workflow is: fit ODE parameters (optional), compute
    residuals, then train a GPR or quadratic polynomial on those residuals.
    Prediction at a new time is ODE output plus residual correction.

    :param tolerance: Absolute ratio match tolerance for historical hits.
    :type tolerance: float
    :param horizon_factor: Extrapolation upper bound is
        ``max_time * horizon_factor``.
    :type horizon_factor: float
    :param model_type: Residual model, ``'gpr'`` or ``'poly'``.
    :type model_type: str
    :param kernel: sklearn GP kernel; a Matern + DotProduct + White default is
        used when ``None``.
    :type kernel: sklearn.gaussian_process.kernels.Kernel or None
    :param fit_ode: If True, estimate ODE parameters per ID; otherwise use
        the configured default kinetic parameters.
    :type fit_ode: bool
    :param search_iter: Nested interval-search iterations in ``query``.
    :type search_iter: int
    :param n_segments: Grid points per search iteration (minus one intervals).
    :type n_segments: int
    :param device: Torch device stored for API compatibility (ODE/GPR are numpy).
    :type device: torch.device or None
    :param gpr_alpha: Diagonal noise added by ``GaussianProcessRegressor``.
    :type gpr_alpha: float
    :param gpr_restarts: Optimizer restarts for the residual GP kernel.
    :type gpr_restarts: int
    :param gpr_seed: Random seed for the residual GP.
    :type gpr_seed: int
    :param poly_degree: Degree of the polynomial residual model.
    :type poly_degree: int
    :param min_points: Minimum number of time points required to fit an ID.
    :type min_points: int
    :param kernel_const: Scale of each ``ConstantKernel`` factor.
    :type kernel_const: float
    :param matern_length_scale: Matern length scale.
    :type matern_length_scale: float
    :param matern_nu: Matern smoothness.
    :type matern_nu: float
    :param dot_sigma: ``DotProduct`` inhomogeneity parameter.
    :type dot_sigma: float
    :param white_noise: White-kernel noise level.
    :type white_noise: float
    :param white_noise_lower: Lower bound on the white-kernel noise.
    :type white_noise_lower: float
    :param white_noise_upper: Upper bound on the white-kernel noise.
    :type white_noise_upper: float
    :param h_a: Default homology-arm length for type A.
    :type h_a: float
    :param h_b: Default homology-arm length for type B.
    :type h_b: float
    :param k_a0: Default baseline rate for arm A.
    :type k_a0: float
    :param k_b0: Default baseline rate for arm B.
    :type k_b0: float
    :param beta: Default length exponent.
    :type beta: float
    :param r_a: Default intermediate rate for arm A.
    :type r_a: float
    :param r_b: Default intermediate rate for arm B.
    :type r_b: float
    :param alpha: Default reverse-rate scale.
    :type alpha: float
    :param h_a_min: Lower search bound for ``h_A``.
    :type h_a_min: float
    :param h_a_max: Upper search bound for ``h_A``.
    :type h_a_max: float
    :param h_b_min: Lower search bound for ``h_B``.
    :type h_b_min: float
    :param h_b_max: Upper search bound for ``h_B``.
    :type h_b_max: float
    :param k_a0_min: Lower search bound for ``k_A0``.
    :type k_a0_min: float
    :param k_a0_max: Upper search bound for ``k_A0``.
    :type k_a0_max: float
    :param k_b0_min: Lower search bound for ``k_B0``.
    :type k_b0_min: float
    :param k_b0_max: Upper search bound for ``k_B0``.
    :type k_b0_max: float
    :param beta_min: Lower search bound for ``beta``.
    :type beta_min: float
    :param beta_max: Upper search bound for ``beta``.
    :type beta_max: float
    :param r_a_min: Lower search bound for ``r_A``.
    :type r_a_min: float
    :param r_a_max: Upper search bound for ``r_A``.
    :type r_a_max: float
    :param r_b_min: Lower search bound for ``r_B``.
    :type r_b_min: float
    :param r_b_max: Upper search bound for ``r_B``.
    :type r_b_max: float
    :param alpha_min: Lower search bound for ``alpha``.
    :type alpha_min: float
    :param alpha_max: Upper search bound for ``alpha``.
    :type alpha_max: float
    :param de_maxiter: Differential-evolution generations.
    :type de_maxiter: int
    :param de_popsize: Differential-evolution population multiplier.
    :type de_popsize: int
    :param de_seed: Seed for differential evolution.
    :type de_seed: int
    :param local_method: Local optimizer used when differential evolution fails.
    :type local_method: str
    :param ode_fail_loss: Penalty returned when an ODE evaluation raises.
    :type ode_fail_loss: float
    :param ivp_method: ``solve_ivp`` method name.
    :type ivp_method: str
    :param ivp_rtol: Relative integrator tolerance.
    :type ivp_rtol: float
    :param ivp_atol: Absolute integrator tolerance.
    :type ivp_atol: float
    :param product_eps: Product-mass threshold treated as undifferentiated.
    :type product_eps: float
    :param init_ratio: Ratio returned when the ODE cannot be integrated.
    :type init_ratio: float
    :param query_tol_factor: Acceptance band after the nested search, in units of ``tolerance``.
    :type query_tol_factor: float
    :param time_atol: Stop the nested search when the interval is narrower than this.
    :type time_atol: float
    :param ratio_floor: Lowest ratio returned by ``predict_ratio``.
    :type ratio_floor: float
    """

    def __init__(self, tolerance=1e-2, horizon_factor=3.0,
                 model_type='gpr', kernel=None,
                 fit_ode=True, search_iter=20, n_segments=10, device=None,
                 gpr_alpha=0.0, gpr_restarts=10, gpr_seed=42,
                 poly_degree=2, min_points=3,
                 kernel_const=1.0, matern_length_scale=1.0, matern_nu=2.5,
                 dot_sigma=1.0, white_noise=0.1,
                 white_noise_lower=1e-3, white_noise_upper=10.0,
                 h_a=DEFAULT_PARAMS['h_A'], h_b=DEFAULT_PARAMS['h_B'],
                 k_a0=DEFAULT_PARAMS['k_A0'], k_b0=DEFAULT_PARAMS['k_B0'],
                 beta=DEFAULT_PARAMS['beta'], r_a=DEFAULT_PARAMS['r_A'],
                 r_b=DEFAULT_PARAMS['r_B'], alpha=DEFAULT_PARAMS['alpha'],
                 h_a_min=PARAM_BOUNDS['h_A'][0], h_a_max=PARAM_BOUNDS['h_A'][1],
                 h_b_min=PARAM_BOUNDS['h_B'][0], h_b_max=PARAM_BOUNDS['h_B'][1],
                 k_a0_min=PARAM_BOUNDS['k_A0'][0], k_a0_max=PARAM_BOUNDS['k_A0'][1],
                 k_b0_min=PARAM_BOUNDS['k_B0'][0], k_b0_max=PARAM_BOUNDS['k_B0'][1],
                 beta_min=PARAM_BOUNDS['beta'][0], beta_max=PARAM_BOUNDS['beta'][1],
                 r_a_min=PARAM_BOUNDS['r_A'][0], r_a_max=PARAM_BOUNDS['r_A'][1],
                 r_b_min=PARAM_BOUNDS['r_B'][0], r_b_max=PARAM_BOUNDS['r_B'][1],
                 alpha_min=PARAM_BOUNDS['alpha'][0], alpha_max=PARAM_BOUNDS['alpha'][1],
                 de_maxiter=50, de_popsize=15, de_seed=42,
                 local_method='L-BFGS-B', ode_fail_loss=1e10,
                 ivp_method='RK45', ivp_rtol=1e-8, ivp_atol=1e-10,
                 product_eps=1e-12, init_ratio=1.0,
                 query_tol_factor=10.0, time_atol=1e-6, ratio_floor=0.0):
        self.tolerance = tolerance
        self.horizon_factor = horizon_factor
        self.model_type = model_type          # 'gpr' or 'poly'
        self.fit_ode = fit_ode
        self.search_iter = search_iter
        self.n_segments = n_segments
        self.device = device if device is not None else torch.device('cpu')
        self.gpr_alpha = gpr_alpha
        self.gpr_restarts = gpr_restarts
        self.gpr_seed = gpr_seed
        self.poly_degree = poly_degree
        self.min_points = min_points
        self.kernel_const = kernel_const
        self.matern_length_scale = matern_length_scale
        self.matern_nu = matern_nu
        self.dot_sigma = dot_sigma
        self.white_noise = white_noise
        self.white_noise_lower = white_noise_lower
        self.white_noise_upper = white_noise_upper
        self.de_maxiter = de_maxiter
        self.de_popsize = de_popsize
        self.de_seed = de_seed
        self.local_method = local_method
        self.ode_fail_loss = ode_fail_loss
        self.ivp_method = ivp_method
        self.ivp_rtol = ivp_rtol
        self.ivp_atol = ivp_atol
        self.product_eps = product_eps
        self.init_ratio = init_ratio
        self.query_tol_factor = query_tol_factor
        self.time_atol = time_atol
        self.ratio_floor = ratio_floor
        self.default_params = {
            'h_A': h_a, 'h_B': h_b, 'k_A0': k_a0, 'k_B0': k_b0,
            'beta': beta, 'r_A': r_a, 'r_B': r_b, 'alpha': alpha,
        }
        self.param_bounds = {
            'h_A': (h_a_min, h_a_max), 'h_B': (h_b_min, h_b_max),
            'k_A0': (k_a0_min, k_a0_max), 'k_B0': (k_b0_min, k_b0_max),
            'beta': (beta_min, beta_max), 'r_A': (r_a_min, r_a_max),
            'r_B': (r_b_min, r_b_max), 'alpha': (alpha_min, alpha_max),
        }
        if kernel is None:
            # Smooth Matern trend + linear DotProduct term + observation noise.
            self.kernel = (
                ConstantKernel(kernel_const) * Matern(length_scale=matern_length_scale, nu=matern_nu) +
                ConstantKernel(kernel_const) * DotProduct(sigma_0=dot_sigma) +
                WhiteKernel(noise_level=white_noise,
                            noise_level_bounds=(white_noise_lower, white_noise_upper))
            )
        else:
            self.kernel = kernel
        self.models = {}          # id -> ((model_type, model_obj), ode_params)
        self.hist_data = {}       # id -> (times, ratios)
        self.fitted_ids = set()

    def export_kwargs(self):
        """Return constructor kwargs that rebuild this predictor in a worker.

        Fitted models and historical data are not included; workers receive
        those separately.

        :returns: Keyword arguments for ``RatioTimePredictor``.
        :rtype: dict
        """
        return {
            'tolerance': self.tolerance,
            'horizon_factor': self.horizon_factor,
            'model_type': self.model_type,
            'kernel': self.kernel,
            'fit_ode': self.fit_ode,
            'search_iter': self.search_iter,
            'n_segments': self.n_segments,
            'device': self.device,
            'gpr_alpha': self.gpr_alpha,
            'gpr_restarts': self.gpr_restarts,
            'gpr_seed': self.gpr_seed,
            'poly_degree': self.poly_degree,
            'min_points': self.min_points,
            'kernel_const': self.kernel_const,
            'matern_length_scale': self.matern_length_scale,
            'matern_nu': self.matern_nu,
            'dot_sigma': self.dot_sigma,
            'white_noise': self.white_noise,
            'white_noise_lower': self.white_noise_lower,
            'white_noise_upper': self.white_noise_upper,
            'h_a': self.default_params['h_A'],
            'h_b': self.default_params['h_B'],
            'k_a0': self.default_params['k_A0'],
            'k_b0': self.default_params['k_B0'],
            'beta': self.default_params['beta'],
            'r_a': self.default_params['r_A'],
            'r_b': self.default_params['r_B'],
            'alpha': self.default_params['alpha'],
            'h_a_min': self.param_bounds['h_A'][0],
            'h_a_max': self.param_bounds['h_A'][1],
            'h_b_min': self.param_bounds['h_B'][0],
            'h_b_max': self.param_bounds['h_B'][1],
            'k_a0_min': self.param_bounds['k_A0'][0],
            'k_a0_max': self.param_bounds['k_A0'][1],
            'k_b0_min': self.param_bounds['k_B0'][0],
            'k_b0_max': self.param_bounds['k_B0'][1],
            'beta_min': self.param_bounds['beta'][0],
            'beta_max': self.param_bounds['beta'][1],
            'r_a_min': self.param_bounds['r_A'][0],
            'r_a_max': self.param_bounds['r_A'][1],
            'r_b_min': self.param_bounds['r_B'][0],
            'r_b_max': self.param_bounds['r_B'][1],
            'alpha_min': self.param_bounds['alpha'][0],
            'alpha_max': self.param_bounds['alpha'][1],
            'de_maxiter': self.de_maxiter,
            'de_popsize': self.de_popsize,
            'de_seed': self.de_seed,
            'local_method': self.local_method,
            'ode_fail_loss': self.ode_fail_loss,
            'ivp_method': self.ivp_method,
            'ivp_rtol': self.ivp_rtol,
            'ivp_atol': self.ivp_atol,
            'product_eps': self.product_eps,
            'init_ratio': self.init_ratio,
            'query_tol_factor': self.query_tol_factor,
            'time_atol': self.time_atol,
            'ratio_floor': self.ratio_floor,
        }

    def _ode_ratio(self, params, t):
        """Evaluate the configured ODE at time ``t`` for one parameter set.

        :param params: Kinetic parameter dictionary.
        :type params: dict
        :param t: Query time.
        :type t: float
        :returns: Predicted type-A ratio.
        :rtype: float
        """
        return simulate_ode_ratio(
            params['h_A'], params['h_B'], t,
            params['k_A0'], params['k_B0'], params['beta'],
            params['r_A'], params['r_B'], params['alpha'],
            method=self.ivp_method, rtol=self.ivp_rtol, atol=self.ivp_atol,
            product_eps=self.product_eps, init_ratio=self.init_ratio,
        )

    def fit_id(self, id_, times, ratios):
        """Fit ODE parameters and a residual model for a single ID.

        IDs with fewer than ``min_points`` time points are skipped. Residuals are
        ``observed_ratio - ode_prediction``.

        :param id_: Identifier of the construct / experiment.
        :param times: 1-D array of observation times.
        :type times: numpy.ndarray
        :param ratios: 1-D array of observed ratios aligned with ``times``.
        :type ratios: numpy.ndarray
        """
        if len(times) < self.min_points:
            print(f"ID {id_} has too few data points, skip.")
            return
        # Fit the ODE parameters, or keep the defaults.
        if self.fit_ode:
            params = fit_ode_params(
                times, ratios,
                bounds=self.param_bounds,
                default=self.default_params,
                de_maxiter=self.de_maxiter,
                de_popsize=self.de_popsize,
                de_seed=self.de_seed,
                local_method=self.local_method,
                ode_fail_loss=self.ode_fail_loss,
                ivp_method=self.ivp_method,
                ivp_rtol=self.ivp_rtol,
                ivp_atol=self.ivp_atol,
                product_eps=self.product_eps,
                init_ratio=self.init_ratio,
            )
        else:
            params = self.default_params.copy()
        # Evaluate the mechanistic trajectory at the observed times.
        ode_pred = np.array([self._ode_ratio(params, t) for t in times])
        # The residual is the observation minus the ODE output. That is what the statistical model learns.
        residuals = ratios - ode_pred
        # Train the residual model, either a Gaussian process or a quadratic polynomial.
        X = times.reshape(-1, 1)
        if self.model_type == 'gpr':
            gpr = GaussianProcessRegressor(
                kernel=self.kernel,
                alpha=self.gpr_alpha,
                n_restarts_optimizer=self.gpr_restarts,
                random_state=self.gpr_seed
            )
            gpr.fit(X, residuals)
            model_obj = ('gpr', gpr)
        elif self.model_type == 'poly':
            coeffs = np.polyfit(times, residuals, deg=self.poly_degree)
            model_obj = ('poly', coeffs)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        self.models[id_] = (model_obj, params)
        self.hist_data[id_] = (times, ratios)
        self.fitted_ids.add(id_)
        print(f"\nFitted ID {id_}")

    def fit_from_dict(self, id_seq_dict):
        """Fit every ID in a ``{id: [(t, ratio, score), ...]}`` dictionary.

        The score field is ignored during fitting.

        :param id_seq_dict: Mapping from ID to a sequence of triples.
        :type id_seq_dict: dict
        """
        for id_, seq in id_seq_dict.items():
            times = np.array([x[0] for x in seq])
            ratios = np.array([x[1] for x in seq])
            self.fit_id(id_, times, ratios)
        print(f"Fitted {len(self.fitted_ids)} IDs.")

    def predict_ratio(self, id_, t):
        """Predict the type-A ratio at time ``t`` (ODE + residual).

        If ODE integration raises, the ODE term is replaced by linear
        interpolation (or last-two-point slope extrapolation) on historical
        ratios. The residual is still added. Negative totals are clipped to 0.

        :param id_: A previously fitted ID.
        :param t: Query time.
        :type t: float
        :returns: Predicted ratio, at least 0.
        :rtype: float
        :raises ValueError: If ``id_`` has not been fitted.
        """
        if id_ not in self.models:
            raise ValueError(f"ID {id_} not fitted.")
        (model_type, model_obj), params = self.models[id_]
        # Add the mechanistic ODE prediction.
        try:
            ode_val = self._ode_ratio(params, t)
        except Exception:
            # If simulation fails, interpolate/extrapolate from history.
            times_hist, ratios_hist = self.hist_data[id_]
            if t <= times_hist[-1]:
                # Linear interpolation between surrounding historical points.
                idx = np.searchsorted(times_hist, t)
                if idx == 0:
                    ode_val = ratios_hist[0]
                elif idx >= len(times_hist):
                    ode_val = ratios_hist[-1]
                else:
                    t1, t2 = times_hist[idx-1], times_hist[idx]
                    r1, r2 = ratios_hist[idx-1], ratios_hist[idx]
                    ode_val = r1 + (r2-r1)*(t-t1)/(t2-t1)
            else:
                # Extrapolation: slope of the last two observed points.
                slope = (ratios_hist[-1] - ratios_hist[-2]) / (times_hist[-1] - times_hist[-2])
                ode_val = ratios_hist[-1] + slope * (t - times_hist[-1])
        # Add the residual correction.
        if model_type == 'gpr':
            res_mean, _ = model_obj.predict(np.array([[t]]), return_std=True)
            res_val = res_mean[0]
        else:  # Quadratic polynomial residual.
            res_val = np.polyval(model_obj, t)
        total = ode_val + res_val
        return total if total >= self.ratio_floor else self.ratio_floor

    def query(self, id_, target_ratio, max_time):
        """Find the earliest time at which the predicted ratio hits ``target_ratio``.

        Search order:

        1. Historical observations within ``tolerance``.
        2. Trend check: if the last slope cannot move toward the target, fail.
        3. Nested interval search from the last observed time to
           ``max_time * horizon_factor``, shrinking around the first sign change.

        :param id_: A previously fitted ID.
        :param target_ratio: Desired type-A ratio.
        :type target_ratio: float
        :param max_time: Historical horizon used to scale the search upper bound.
        :type max_time: float
        :returns: ``(found, time)``. ``found`` is True if a crossing was
            located (within ``query_tol_factor * tolerance`` after search).
            ``time`` is 0.0 when not found.
        :rtype: tuple
        :raises ValueError: If ``id_`` has not been fitted.
        """
        if id_ not in self.models:
            raise ValueError(f"ID {id_} not fitted.")
        times_hist, ratios_hist = self.hist_data[id_]
        # Return the earliest historical time already within tolerance of the target.
        hist_matches = times_hist[np.abs(ratios_hist - target_ratio) <= self.tolerance]
        if len(hist_matches) > 0:
            return (True, hist_matches.min())
        # Reject a target that sits outside the observed range while the recent slope moves away from it.
        if len(times_hist) >= 2:
            slope = (ratios_hist[-1] - ratios_hist[-2]) / (times_hist[-1] - times_hist[-2])
            # Target above historical max with non-positive slope (or below min
            # with non-negative slope) cannot be reached by continuing the trend.
            if target_ratio > ratios_hist.max() and slope <= 0:
                return (False, 0.0)
            if target_ratio < ratios_hist.min() and slope >= 0:
                return (False, 0.0)
        # Search past the last observation.
        t_min = times_hist[-1]
        t_max = max_time * self.horizon_factor
        # Nested interval search (bisection-style): evaluate a grid, keep the
        # first interval where predicted_ratio - target changes sign.
        for _ in range(self.search_iter):
            points = np.linspace(t_min, t_max, self.n_segments + 1)
            values = np.array([self.predict_ratio(id_, t) for t in points])
            diffs = values - target_ratio
            sign_changes = np.where(np.diff(np.sign(diffs)))[0]
            if len(sign_changes) == 0:
                return (False, 0.0)  # trajectory never crosses the target
            idx = sign_changes[0]
            t_min = points[idx]
            t_max = points[idx + 1]
            if t_max - t_min < self.time_atol:
                break
        pred_time = (t_min + t_max) / 2.0
        pred_ratio = self.predict_ratio(id_, pred_time)
        if abs(pred_ratio - target_ratio) <= self.query_tol_factor * self.tolerance:
            return (True, pred_time)
        else:
            return (False, 0.0)


def add_predictor_arguments(parser):
    """Register ODE, solver, and residual-model options on ``parser``.

    Flags already defined by ``query_per_id.py`` (tolerance, horizon, search
    depth, device, model type, and whether to fit the ODE) are not repeated.
    Each default is the value previously hard-coded in this module.

    :param parser: Parser that will receive the optional flags.
    :type parser: argparse.ArgumentParser
    :returns: The same parser, for chaining.
    :rtype: argparse.ArgumentParser
    """
    parser.add_argument("--h_a", required=False, type=float, default=DEFAULT_PARAMS["h_A"],
                        help="Default homology-arm length for type A (default: %(default)s).")
    parser.add_argument("--h_b", required=False, type=float, default=DEFAULT_PARAMS["h_B"],
                        help="Default homology-arm length for type B (default: %(default)s).")
    parser.add_argument("--k_a0", required=False, type=float, default=DEFAULT_PARAMS["k_A0"],
                        help="Default baseline recombination rate for arm A (default: %(default)s).")
    parser.add_argument("--k_b0", required=False, type=float, default=DEFAULT_PARAMS["k_B0"],
                        help="Default baseline recombination rate for arm B (default: %(default)s).")
    parser.add_argument("--beta", required=False, type=float, default=DEFAULT_PARAMS["beta"],
                        help="Default homology-length exponent (default: %(default)s).")
    parser.add_argument("--r_a", required=False, type=float, default=DEFAULT_PARAMS["r_A"],
                        help="Default rate from D into intermediate M_A (default: %(default)s).")
    parser.add_argument("--r_b", required=False, type=float, default=DEFAULT_PARAMS["r_B"],
                        help="Default rate from D into intermediate M_B (default: %(default)s).")
    parser.add_argument("--alpha", required=False, type=float, default=DEFAULT_PARAMS["alpha"],
                        help="Default scale from forward rate k to reverse rate v (default: %(default)s).")

    parser.add_argument("--h_a_min", required=False, type=float, default=PARAM_BOUNDS["h_A"][0],
                        help="Lower search bound for h_A (default: %(default)s).")
    parser.add_argument("--h_a_max", required=False, type=float, default=PARAM_BOUNDS["h_A"][1],
                        help="Upper search bound for h_A (default: %(default)s).")
    parser.add_argument("--h_b_min", required=False, type=float, default=PARAM_BOUNDS["h_B"][0],
                        help="Lower search bound for h_B (default: %(default)s).")
    parser.add_argument("--h_b_max", required=False, type=float, default=PARAM_BOUNDS["h_B"][1],
                        help="Upper search bound for h_B (default: %(default)s).")
    parser.add_argument("--k_a0_min", required=False, type=float, default=PARAM_BOUNDS["k_A0"][0],
                        help="Lower search bound for k_A0 (default: %(default)s).")
    parser.add_argument("--k_a0_max", required=False, type=float, default=PARAM_BOUNDS["k_A0"][1],
                        help="Upper search bound for k_A0 (default: %(default)s).")
    parser.add_argument("--k_b0_min", required=False, type=float, default=PARAM_BOUNDS["k_B0"][0],
                        help="Lower search bound for k_B0 (default: %(default)s).")
    parser.add_argument("--k_b0_max", required=False, type=float, default=PARAM_BOUNDS["k_B0"][1],
                        help="Upper search bound for k_B0 (default: %(default)s).")
    parser.add_argument("--beta_min", required=False, type=float, default=PARAM_BOUNDS["beta"][0],
                        help="Lower search bound for beta (default: %(default)s).")
    parser.add_argument("--beta_max", required=False, type=float, default=PARAM_BOUNDS["beta"][1],
                        help="Upper search bound for beta (default: %(default)s).")
    parser.add_argument("--r_a_min", required=False, type=float, default=PARAM_BOUNDS["r_A"][0],
                        help="Lower search bound for r_A (default: %(default)s).")
    parser.add_argument("--r_a_max", required=False, type=float, default=PARAM_BOUNDS["r_A"][1],
                        help="Upper search bound for r_A (default: %(default)s).")
    parser.add_argument("--r_b_min", required=False, type=float, default=PARAM_BOUNDS["r_B"][0],
                        help="Lower search bound for r_B (default: %(default)s).")
    parser.add_argument("--r_b_max", required=False, type=float, default=PARAM_BOUNDS["r_B"][1],
                        help="Upper search bound for r_B (default: %(default)s).")
    parser.add_argument("--alpha_min", required=False, type=float, default=PARAM_BOUNDS["alpha"][0],
                        help="Lower search bound for alpha (default: %(default)s).")
    parser.add_argument("--alpha_max", required=False, type=float, default=PARAM_BOUNDS["alpha"][1],
                        help="Upper search bound for alpha (default: %(default)s).")

    parser.add_argument("--de_maxiter", required=False, type=int, default=50,
                        help="Differential-evolution generations (default: %(default)s).")
    parser.add_argument("--de_popsize", required=False, type=int, default=15,
                        help="Differential-evolution population multiplier (default: %(default)s).")
    parser.add_argument("--de_seed", required=False, type=int, default=42,
                        help="Random seed for differential evolution (default: %(default)s).")
    parser.add_argument("--local_method", required=False, type=str, default="L-BFGS-B",
                        help="Local optimizer used when differential evolution fails.")
    parser.add_argument("--ode_fail_loss", required=False, type=float, default=1e10,
                        help="Penalty loss when an ODE candidate raises (default: %(default)s).")
    parser.add_argument("--ivp_method", required=False, type=str, default="RK45",
                        help="scipy.integrate.solve_ivp method name.")
    parser.add_argument("--ivp_rtol", required=False, type=float, default=1e-8,
                        help="Relative tolerance of the ODE integrator (default: %(default)s).")
    parser.add_argument("--ivp_atol", required=False, type=float, default=1e-10,
                        help="Absolute tolerance of the ODE integrator (default: %(default)s).")
    parser.add_argument("--product_eps", required=False, type=float, default=1e-12,
                        help="Product mass below this is treated as undifferentiated (default: %(default)s).")
    parser.add_argument("--init_ratio", required=False, type=float, default=1.0,
                        help="Ratio returned when t <= 0 or integration fails (default: %(default)s).")

    parser.add_argument("--gpr_alpha", required=False, type=float, default=0.0,
                        help="Diagonal noise of the residual Gaussian process (default: %(default)s).")
    parser.add_argument("--gpr_restarts", required=False, type=int, default=10,
                        help="Kernel-optimizer restarts for the residual GP (default: %(default)s).")
    parser.add_argument("--gpr_seed", required=False, type=int, default=42,
                        help="Random seed of the residual GP (default: %(default)s).")
    parser.add_argument("--poly_degree", required=False, type=int, default=2,
                        help="Polynomial degree when --model_type is poly (default: %(default)s).")
    parser.add_argument("--min_points", required=False, type=int, default=3,
                        help="Minimum time points required to fit an ID (default: %(default)s).")
    parser.add_argument("--kernel_const", required=False, type=float, default=1.0,
                        help="Scale of each ConstantKernel factor (default: %(default)s).")
    parser.add_argument("--matern_length_scale", required=False, type=float, default=1.0,
                        help="Initial Matern length scale (default: %(default)s).")
    parser.add_argument("--matern_nu", required=False, type=float, default=2.5,
                        help="Matern smoothness parameter (default: %(default)s).")
    parser.add_argument("--dot_sigma", required=False, type=float, default=1.0,
                        help="DotProduct sigma_0 (default: %(default)s).")
    parser.add_argument("--white_noise", required=False, type=float, default=0.1,
                        help="Initial white-kernel noise level (default: %(default)s).")
    parser.add_argument("--white_noise_lower", required=False, type=float, default=1e-3,
                        help="Lower bound of the white-kernel noise (default: %(default)s).")
    parser.add_argument("--white_noise_upper", required=False, type=float, default=10.0,
                        help="Upper bound of the white-kernel noise (default: %(default)s).")
    parser.add_argument("--query_tol_factor", required=False, type=float, default=10.0,
                        help="Accept a searched time when the ratio error is within this many tolerances.")
    parser.add_argument("--time_atol", required=False, type=float, default=1e-6,
                        help="Stop the nested time search when the interval is narrower than this.")
    parser.add_argument("--ratio_floor", required=False, type=float, default=0.0,
                        help="Lowest ratio returned by prediction (default: %(default)s).")
    return parser


def predictor_kwargs_from_args(args, device):
    """Build ``RatioTimePredictor`` keyword arguments from a CLI namespace.

    :param args: Namespace produced by ``query_per_id.parse_arg``.
    :type args: argparse.Namespace
    :param device: Torch device already resolved from ``--device``.
    :type device: torch.device
    :returns: Constructor keyword arguments.
    :rtype: dict
    """
    return {
        "tolerance": args.tolerance,
        "horizon_factor": args.horizon_factor,
        "model_type": args.model_type,
        "fit_ode": args.fit_ode,
        "search_iter": args.search_iter,
        "n_segments": args.n_segments,
        "device": device,
        "gpr_alpha": args.gpr_alpha,
        "gpr_restarts": args.gpr_restarts,
        "gpr_seed": args.gpr_seed,
        "poly_degree": args.poly_degree,
        "min_points": args.min_points,
        "kernel_const": args.kernel_const,
        "matern_length_scale": args.matern_length_scale,
        "matern_nu": args.matern_nu,
        "dot_sigma": args.dot_sigma,
        "white_noise": args.white_noise,
        "white_noise_lower": args.white_noise_lower,
        "white_noise_upper": args.white_noise_upper,
        "h_a": args.h_a,
        "h_b": args.h_b,
        "k_a0": args.k_a0,
        "k_b0": args.k_b0,
        "beta": args.beta,
        "r_a": args.r_a,
        "r_b": args.r_b,
        "alpha": args.alpha,
        "h_a_min": args.h_a_min,
        "h_a_max": args.h_a_max,
        "h_b_min": args.h_b_min,
        "h_b_max": args.h_b_max,
        "k_a0_min": args.k_a0_min,
        "k_a0_max": args.k_a0_max,
        "k_b0_min": args.k_b0_min,
        "k_b0_max": args.k_b0_max,
        "beta_min": args.beta_min,
        "beta_max": args.beta_max,
        "r_a_min": args.r_a_min,
        "r_a_max": args.r_a_max,
        "r_b_min": args.r_b_min,
        "r_b_max": args.r_b_max,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "de_maxiter": args.de_maxiter,
        "de_popsize": args.de_popsize,
        "de_seed": args.de_seed,
        "local_method": args.local_method,
        "ode_fail_loss": args.ode_fail_loss,
        "ivp_method": args.ivp_method,
        "ivp_rtol": args.ivp_rtol,
        "ivp_atol": args.ivp_atol,
        "product_eps": args.product_eps,
        "init_ratio": args.init_ratio,
        "query_tol_factor": args.query_tol_factor,
        "time_atol": args.time_atol,
        "ratio_floor": args.ratio_floor,
    }
