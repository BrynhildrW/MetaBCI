# -*- coding: utf-8 -*-

"""Transfer learning algorithm 'TNSRE-20233250953'. [1]

.. [1] Y. Zhang, S. Q. Xie, C. Shi, J. Li and Z. -Q. Zhang, "Cross-Subject Transfer Learning
for Boosting Recognition Performance in SSVEP-Based BCIs," in IEEE Transactions on
Neural Systems and Rehabilitation Engineering, vol. 31, pp. 1574-1583, 2023,
doi: 10.1109/TNSRE.2023.3250953.

Source code: https://github.com/BrynhildrW/SSVEP_algorithms/blob/main/programs/transfer.py

**Notations**:

* Number of events: *Ne*
* Number of (training) samples (for each event): *Nt*
* Number of (testing) samples: *Nte*
* Total number of training trials: *Ne*Nt*
* Total number of testing trials: *Ne*Nte*
* Number of channels: *Nc*
* Number of sampling points: *Np*
* Number of dimensions of sub-space: *Nk*
* Number of harmonics for sinusoidal templates: *Nh*
* Number of filter banks: *Nb*
"""

# %% Basic modules
import numpy as np

from numba import njit

import scipy.linalg as sLA

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


# %% utils functions
def pick_subspace(e_vals, ratio, min_n=None, max_n=None):
    """
    Optimize the number of subspaces.

    Parameters
    ----------
    e_vals : *array-like of shape (Nk,).*
        Sequence of eigenvalues sorted in descending order.
    ratio : *float.*
        0-1. The ratio of the sum of picked eigenvalues to the total.
    min_n: *int.*
        Minimum number of the dimension of subspace.
    max_n: *int.*
        Maximum number of the dimension of subspace.

    Returns
    ----------
    n_components : *int.*
        The optimized number of subspaces.
    """
    # basic information
    e_vals_sum = np.sum(e_vals)
    threshould = ratio * e_vals_sum

    # check non-compliant input parameters
    if (min_n is None) or (min_n < 1):
        min_n = 1
    if (max_n is None) or (max_n > len(e_vals)):
        max_n = len(e_vals)

    # main process
    temp_sum = 0
    for nev, e_val in enumerate(e_vals):
        temp_sum += e_val
        n_components = nev + 1
        if temp_sum >= threshould:
            if n_components < min_n:
                return min_n
            elif n_components >= max_n:
                return max_n
            else:
                return n_components


def solve_gep(
        A,
        B=None,
        n_components=1,
        mode='Max',
        ratio=None,
        min_n=None,
        max_n=None):
    r"""
    Solve generalized eigenvalue problems (GEPs) based on Rayleigh quotient:

    :math:`f(\pmb{w}) = \dfrac{\pmb{w} \pmb{A} {\pmb{w}}^T} {\pmb{w} \pmb{B} {\pmb{w}}^T}
    \Longrightarrow  \pmb{A} \pmb{w} = \pmb{\lambda} \pmb{Bw}.`

    If ``B`` is ``None``, solve eigenvalue problems (EPs):

    :math:`f(\pmb{w}) = \dfrac{\pmb{w} \pmb{A} {\pmb{w}}^T}{\pmb{w} {\pmb{w}}^T}
    \Longrightarrow \pmb{A} \pmb{w} = \pmb{\lambda} \pmb{w}.`

    Parameters
    ----------
    A : *ndarray of shape (m,m).*
        Input matrix.
    B : *ndarray of shape (m,m), optional.*
        Input matrix.
    n_components : *int.*
        Number of eigenvectors picked as filters.
    mode : *str.*
        ``'Max'`` or ``'Min'``. Depends on target function.
    ratio : *float.*
        0-1. The ratio of the sum of useful (``mode='Max'``)
        or deprecated (``mode='Min'``) eigenvalues to the total.
        Only useful when ``n_components=None``.
    min_n: *int.*
        Minimum number of the dimension of subspace.
    max_n: *int.*
        Maximum number of the dimension of subspace.

    Returns
    ----------
    w : *ndarray of shape (Nk,m).*
        Picked eigenvectors.
    """
    # solve EPs
    if B is not None:  # f(w) = (w @ A @ w^T) / (w @ B @ w^T)
        # faster than sLA.eig(a=A, b=B)
        e_val, e_vec = sLA.eig(sLA.solve(a=B, b=A, assume_a='sym'))  # ax=b -> x=a^{-1}b
    else:  # f(w) = (w @ A @ w^T) / (w @ w^T)
        e_val, e_vec = sLA.eig(A)

    # pick the optimal subspaces
    w_index = np.flip(np.argsort(e_val))
    if n_components is None:
        n_components = pick_subspace(
            e_vals=e_val[w_index],
            ratio=ratio,
            min_n=min_n,
            max_n=max_n
        )
    if mode == 'Min':
        return np.real(e_vec[:, w_index][:, n_components:].T)
    elif mode == 'Max':
        return np.real(e_vec[:, w_index][:, :n_components].T)


def spatial_filtering(w, X, y=None):
    """
    Generate spatial filtered data.

    Parameters
    ----------
    w : *ndarray of shape (Ne,Nk,Nc) or (Nk,Nc).*
        Spatial filters
    X : *ndarray of shape (Ne,Nc,Np) or (Ne*Nt,Nc,Np).*
        Input dataset.
    y : *ndarray of shape (Ne*Nt,).*
        If ``None``, ``X`` is trial-averaged; Else ``X`` is multi-trial.

    Returns
    ----------
    wX : *ndarray of shape (Ne,Nk,Np) or (Ne*Nt,Nk,Np).*
        Spatial-filtered ``X``.
    """
    # basic information
    n_components = w.shape[-2]  # Nk
    n_points = X.shape[-1]  # Np
    if y is not None:  # multi-trial data
        event_type = list(np.unique(y))
        n_events = len(event_type)  # Ne
    else:
        n_events = X.shape[0]  # Ne

    # check the dimension of filter w
    if w.ndim == 2:  # (Nk,Nc)
        w = np.tile(A=w, reps=(n_events, 1, 1))

    # spatial filtering process
    wX = np.zeros((X.shape[0], n_components, n_points))  # (Ne,Nk,Np) or (Ne*Nt,Nk,Np)
    if y is not None:
        for ntr in range(X.shape[0]):
            idx = event_type.index(y[ntr])
            wX[ntr] = w[idx] @ X[ntr]
    else:
        for ne in range(n_events):
            wX[ne] = w[ne] @ X[ne]
    return wX


@njit(fastmath=True)
def fast_corr_2d(X, Y):
    """
    Use the JIT compiler to calculate Pearson correlation coefficients for 2-D input.

    NOTE: X.shape[-1] may by the reshaped length, not real length:
    e.g. (Ne,Ne * Nk,Np) -reshape-> (Ne,Ne * Nk * Np) -> (Ne,) (return)
    -> 1 / Np * (Ne,) (real corr)

    Parameters
    ----------
    X : *ndarray of shape (d1,d2).*
        Input data.
    Y : *ndarray of shape (d1,d2).*
        Input data.

    Returns
    ----------
    corr : *ndarray of shape (d1,).*
        Equivalent correlation coefficients.
    """
    dim_1 = X.shape[0]
    corr = np.zeros((dim_1))
    for d1 in range(dim_1):
        corr[d1] = X[d1, :] @ Y[d1, :].T
    return corr


@njit(fastmath=True)
def fast_corr_3d(X, Y):
    """
    Use the JIT compiler to calculate Pearson correlation coefficients for 3-D input.

    Parameters
    ----------
    X : *ndarray of shape (d1,d2,d3).*
        Input data.
    Y : *ndarray of shape (d1,d2,d3).*
        Input data.

    Returns
    ----------
    corr : *ndarray of shape (d1,d2).*
        Equivalent correlation coefficients.
    """
    dim_1, dim_2 = X.shape[0], X.shape[1]
    corr = np.zeros((dim_1, dim_2))
    for d1 in range(dim_1):
        for d2 in range(dim_2):
            corr[d1, d2] = X[d1, d2, :] @ Y[d1, d2, :].T
    return corr


@njit(fastmath=True)
def fast_stan_2d(X):
    """
    Use the JIT compiler to apply standardization for 2-D input.

    Parameters
    ----------
    X : *ndarray of shape (d1,d2).*
        Input dataset.

    Returns
    ----------
    X_new : *ndarray of shape (d1,d2).*
        Data after standardization.
    """
    dim_1 = X.shape[0]
    X_new = np.zeros_like(X)
    for d1 in range(dim_1):
        X_new[d1, :] = X[d1, :] - np.mean(X[d1, :])  # centralization
        X_new[d1, :] = X_new[d1, :] / np.std(X_new[d1, :])
    return X_new


@njit(fastmath=True)
def fast_stan_4d(X):
    """
    Use the JIT compiler to apply standardization for 4-D input.

    Parameters
    ----------
    X : *ndarray of shape (d1,d2,d3,d4).*
        Input dataset.

    Returns
    ----------
    X_new : *ndarray of shape (d1,d2,d3,d4).*
        Data after standardization.
    """
    X_new = np.zeros_like(X)
    for d1 in range(X.shape[0]):
        for d2 in range(X.shape[1]):
            for d3 in range(X.shape[2]):
                X_new[d1, d2, d3, :] = X[d1, d2, d3, :] - np.mean(X[d1, d2, d3, :])
                X_new[d1, d2, d3, :] = X_new[d1, d2, d3, :] / np.std(X_new[d1, d2, d3, :])
    return X_new


def sign_sta(x):
    r"""
    Standardization of decision coefficient based on :math:`{\rm sign} \left(x \right)`.

    Parameters
    ----------
    x : *int or float or ndarray.*
        Input data (or value).

    Returns
    ----------
    out : *int or float or ndarray.*
        :math:`{\rm sign} \left(x \right) \times x^2`
    """
    return np.sign(x) * (x**2)


def combine_feature(features, func=sign_sta):
    """
    Coefficient-level integration.

    Parameters
    ----------
    features : *List[Union[int, float, ndarray]].*
        Different features.
    func : *function.*
        Quantization function.

    Returns
    ----------
    coef : *List[Union[int, float, ndarray]].*
        Integrated coefficients.
    """
    coef = np.zeros_like(features[0])
    for feature in features:
        coef += func(feature)
    return coef


def generate_data_info(X, y):
    """
    Generate basic data information.

    Parameters
    ----------
    X : *ndarray of shape (Ne*Nt,Nc,Np).*
        Input data.
    y : *ndarray of shape (Ne*Nt,).*
        Labels for ``X``.

    Returns (Dict)
    --------------
    'event_type' : *ndarray of shape (Ne,).*
        Unique labels.
    'n_events' : *int.*
        Number of events.
    'n_train' : *ndarray of shape (Ne,).*
        Trials of each event.
    'n_chans' : *int.*
        Number of channels.
    'n_points' : *int.*
        Number of sampling points.
    """
    event_type = np.unique(y)
    return {
        'event_type': event_type,
        'n_events': event_type.shape[0],
        'n_train': np.array([np.sum(y == et) for et in event_type]),
        'n_chans': X.shape[1],
        'n_points': X.shape[-1]
    }


# %% algorithm-based functions
def generate_tnsre_20233250953_mat(X, y, sine_template):
    r"""
    Generate covariance matrices :math:`\pmb{Q}` & :math:`\pmb{S}`.

    Parameters
    ----------
    X : *ndarray of shape (Ne*Nt,Nc,Np).*
        Sklearn-style dataset. *Nt>=2*.
    y : *ndarray of shape (Ne*Nt,).*
        Labels for ``X``.
    sine_template : *ndarray of shape (Ne,2*Nh,Np).*
        Sinusoidal templates.

    Returns
    ----------
    Q : *ndarray of shape (Ne,2*Nc+2*Nh,2*Nc+2*Nh).*
        Covariance matrices.
    S : *ndarray of shape (Ne,2*Nc+2*Nh,2*Nc+2*Nh).*
        Variance matrices.
    X_mea : *ndarray of shape (Ne,Nc,Np).*
        Trial-averaged ``X``.
    """
    # basic information
    event_type = np.unique(y)
    n_events = event_type.shape[0]  # Ne
    n_chans = X.shape[1]  # Nc
    n_points = X.shape[-1]  # Np
    n_dims = sine_template.shape[1]  # 2Nh

    # block covariance matrices: S & Q
    S = np.tile(A=np.eye(2 * n_chans + n_dims), reps=(n_events, 1, 1))
    Q = np.zeros_like(S)  # (Ne,2*Nc+2*Nh,2*Nc+2*Nh)
    X_mean = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne, et in enumerate(event_type):
        X_temp = X[y == et]  # (Nt,Nc,Np)
        n_train = X_temp.shape[0]  # Nt
        assert n_train > 1, 'The number of training samples is too small!'

        X_sum = np.sum(X_temp, axis=0)  # (Nc,Np)
        X_mean[ne] = X_sum / n_train  # (Nc,Np)

        # blocks preparation
        Css = X_sum @ X_sum.T  # (Nc,Nc)
        Csm = X_sum @ X_mean[ne].T  # (Nc,Nc)
        Cmm = X_mean[ne] @ X_mean[ne].T  # (Nc,Nc)
        Csy = X_sum @ sine_template[ne].T  # (Nc,2Nh)
        Cmy = X_mean[ne] @ sine_template[ne].T  # (Nc,2Nh)
        Cyy = sine_template[ne] @ sine_template[ne].T  # (2Nh,2Nh)
        Cxx = np.zeros_like(Css)  # (Nc,Nc)
        for ntr in range(n_train):
            Cxx += X_temp[ntr] @ X_temp[ntr].T

        # block covariance matrices S: [[S11,S12,S13],[S21,S22,S23],[S31,S32,S33]]
        # S11: inter-trial covariance
        S[ne, :n_chans, :n_chans] = Css

        # S12 & S21.T covariance between the SSVEP trials & the individual template
        S[ne, :n_chans, n_chans:2 * n_chans] = Csm
        S[ne, n_chans:2 * n_chans, :n_chans] = Csm.T

        # S13 & S31.T: similarity between the SSVEP trials & sinusoidal template
        S[ne, :n_chans, 2 * n_chans:] = Csy
        S[ne, 2 * n_chans:, :n_chans] = Csy.T

        # S23 & S32.T: covariance between the individual template & sinusoidal template
        S[ne, n_chans:2 * n_chans, 2 * n_chans:] = Cmy
        S[ne, 2 * n_chans:, n_chans:2 * n_chans] = Cmy.T

        # S22 & S33: variance of individual template & sinusoidal template
        S[ne, n_chans:2 * n_chans, n_chans:2 * n_chans] = 2 * Cmm
        S[ne, 2 * n_chans:, 2 * n_chans:] = 2 * Cyy

        # block covariance matrices Q: blkdiag(Q1,Q2,Q3)
        # Q1: variance of the single-trial SSVEP
        Q[ne, :n_chans, :n_chans] = Cxx

        # Q2 & Q3: variance of individual template & sinusoidal template
        Q[ne, n_chans:2 * n_chans, n_chans:2 * n_chans] = Cmm
        Q[ne, 2 * n_chans:, 2 * n_chans:] = Cyy
    return Q, S, X_mean


def solve_tnsre_20233250953_func(Q, S, n_chans, n_components=1):
    """
    Solve TNSRE_20233250953 target function.

    Parameters
    ----------
    Q : *ndarray of shape (Ne,2*Nc+2*Nh,2*Nc+2*Nh).*
        Covariance matrices.
    S : *ndarray of shape (Ne,2*Nc+2*Nh,2*Nc+2*Nh).*
        Variance matrices.
    n_components : *int.*
        Number of eigenvectors picked as filters. Defaults to ``1``.

    Returns
    ----------
    w : *ndarray of shape (Ne,Nk,Nc).*
        Spatial filters for original signal.
    u : *ndarray of shape (Ne,Nk,Nc).*
        Spatial filters for averaged template.
    v : *ndarray of shape (Ne,Nk,2*Nh).*
        Spatial filters for sinusoidal template.
    ew : *ndarray of shape (Ne*Nk,Nc).*
        Concatenated ``w``.
    eu : *ndarray of shape (Ne*Nk,Nc).*
        Concatenated ``u``.
    ev : *ndarray of shape (Ne*Nk,2*Nh).*
        Concatenated ``v``.
    """
    # basic information
    n_events = Q.shape[0]  # Ne
    n_dims = int(Q.shape[1] - 2 * n_chans)  # 2Nh

    # solve GEPs
    w = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    u = np.zeros_like(w)  # (Ne,Nk,Nc)
    v = np.zeros((n_events, n_components, n_dims))  # (Ne,Nk,2Nh)
    for ne in range(n_events):
        spatial_filter = solve_gep(A=S[ne], B=Q[ne], n_components=n_components)
        w[ne] = spatial_filter[:, :n_chans]  # for raw signal
        u[ne] = spatial_filter[:, n_chans:2 * n_chans]  # for averaged template
        v[ne] = spatial_filter[:, 2 * n_chans:]  # for sinusoidal template
    ew = np.reshape(w, (n_events * n_components, n_chans), 'C')  # (Ne*Nk,Nc)
    eu = np.reshape(u, (n_events * n_components, n_chans), 'C')  # (Ne*Nk,Nc)
    ev = np.reshape(v, (n_events * n_components, n_dims), 'C')  # (Ne*Nk,Nc)
    return w, u, v, ew, eu, ev


def generate_tnsre_20233250953_template(eu, ev, X_mean, sine_template):
    """
    Generate TNSRE_20233250953 templates.

    Parameters
    ----------
    eu : *ndarray of shape (Ne*Nk,Nc).*
        Concatenated ``u``. See details in ``solve_tnsre_20233250953_func()``.
    ev : *ndarray of shape (Ne*Nk,2*Nh).*
        Concatenated ``v``. See details in ``solve_tnsre_20233250953_func()``.
    X_mean : *ndarray of shape (Ne,Nc,Np).*
        Trial-averaged data.
    sine_template : *ndarray of shape (Ne,2*Nh,Np).*
        Sinusoidal templates.

    Returns
    ----------
    euX : *ndarray of shape (Ne,Ne*Nk,Np).*
        Filtered averaged templates.
    evY : *ndarray of shape (Ne,Ne*Nk,Np).*
        Filtered sinusoidal templates.
    """
    # spatial filtering process
    euX = spatial_filtering(w=eu, X=X_mean)  # (Ne,Ne*Nk,Np)
    evY = spatial_filtering(w=ev, X=sine_template)  # (Ne,Ne*Nk,Np)
    return euX, evY


def tnsre_20233250953_kernel(X_train, y_train, sine_template, n_components=1):
    """
    Intra-domain modeling process of TNSRE_20233250953.

    Parameters
    ----------
    X_train : *ndarray of shape (Ne*Nt,Nc,Np).*
        Sklearn-style training dataset. *Nt>=2*.
    y_train : *ndarray of shape (Ne*Nt,).*
        Labels for ``X_train``.
    sine_template : *ndarray of shape (Ne,2*Nh,Np).*
        Sinusoidal templates. Defaults to ``None``.
    n_components : *int.*
        Number of eigenvectors picked as filters. Defaults to ``1``.

    Returns (Dict)
    --------------
    Q : *ndarray of shape (Ne,2*Nc+2*Nh,2*Nc+2*Nh).*
        Covariance matrices.
    S : *ndarray of shape (Ne,2*Nc+2*Nh,2*Nc+2*Nh).*
        Variance matrices.
    w : *ndarray of shape (Ne,Nk,Nc).*
        Spatial filters for original signal.
    u : *ndarray of shape (Ne,Nk,Nc).*
        Spatial filters for averaged template.
    v : *ndarray of shape (Ne,Nk,2*Nh).*
        Spatial filters for sinusoidal template.
    ew : *ndarray of shape (Ne*Nk,Nc).*
        Concatenated ``w``.
    eu : *ndarray of shape (Ne*Nk,Nc).*
        Concatenated ``u``.
    ev : *ndarray of shape (Ne*Nk,2*Nh).*
        Concatenated ``v``.
    euX : *ndarray of shape (Ne,Ne*Nk*Np).*
        Filtered averaged templates.
    evY : *ndarray of shape (Ne,Ne*Nk*Np).*
        Filtered sinusoidal templates.
    """
    # solve target functions
    Q, S, X_mean = generate_tnsre_20233250953_mat(
        X=X_train,
        y=y_train,
        sine_template=sine_template
    )
    w, u, v, ew, eu, ev = solve_tnsre_20233250953_func(
        Q=Q,
        S=S,
        n_chans=X_mean.shape[1],
        n_components=n_components
    )

    # generate spatial-filtered templates
    euX, evY = generate_tnsre_20233250953_template(
        eu=eu,
        ev=ev,
        X_mean=X_mean,
        sine_template=sine_template
    )
    return {
        'Q': Q, 'S': S,
        'w': w, 'u': u, 'v': v, 'ew': ew, 'eu': eu, 'ev': ev,
        'euX': euX, 'evY': evY
    }


def tnsre_20233250953_feature(X_test, source_model, trans_model, target_model):
    """
    The pattern matching process of TNSRE_20233250953.

    Parameters
    ----------
    X_test : *ndarray of shape (Ne*Nte,Nc,Np).*
        Test dataset.
    source_model : *Dict[str, ndarray].*
        See details in ``TNSRE_20233250953.intra_source_training()``
    trans_model : *Dict[str, ndarray].*
        See details in: ``TNSRE_20233250953.transfer_learning()``,
        ``TNSRE_20233250953.distance_calculation()``,
        ``TNSRE_20233250953.weight_optimization()``.
    target_model : *Dict[str, ndarray].*
        See details in ``TNSRE_20233250953.intra_target_training()``.

    Returns
    ----------
    rho_temp : *ndarray of shape (Ne*Nte,Ne,4).*
        4-D features.
    rho : *ndarray of shape (Ne*Nte,Ne).*
        Intergrated features.
    """
    # load in models
    euX_source = source_model['euX_source']  # (Ns,Ne,Ne*Nk,Np)
    evY_source = source_model['evY_source']  # (Ns,Ne,Ne*Nk,Np)
    euX_trans = trans_model['euX_trans']  # (Ns,Ne,Ne*Nk,Nc)
    evY_trans = trans_model['evY_trans']  # (Ns,Ne,Ne*Nk,Nc)
    weight_euX = trans_model['weight_euX']  # (Ns,Ne)
    weight_evY = trans_model['weight_evY']  # (Ns,Ne)
    ew_target = target_model['ew']  # (Ne*Nk,Nc)
    euX_target = target_model['euX']  # (Ne,Ne*Nk,Np)
    evY_target = target_model['evY']  # (Ne,Ne*Nk,Np)

    # basic information
    n_subjects = euX_source.shape[0]  # Ns
    n_events = euX_source.shape[1]  # Ne
    n_test = X_test.shape[0]  # Ne*Nte
    n_points = X_test.shape[-1]  # Np

    # reshape matrix for faster computing
    euX_source = np.reshape(euX_source, (n_subjects, n_events, -1), 'C')  # (Ns,Ne,Ne*Nk*Np)
    evY_source = np.reshape(evY_source, (n_subjects, n_events, -1), 'C')  # (Ns,Ne,Ne*Nk*Np)
    euX_target = np.reshape(euX_target, (n_events, -1), 'C')  # (Ne,Ne*Nk*Np)
    evY_target = np.reshape(evY_target, (n_events, -1), 'C')  # (Ne,Ne*Nk*Np)

    # 4-D features
    rho_temp = np.zeros((n_test, n_events, 4))  # (Ne*Nte,Ne,4)
    for nte in range(n_test):
        X_trans_x = np.reshape(
            a=fast_stan_4d(euX_trans @ X_test[nte]),
            newshape=(n_subjects, n_events, -1),
            order='C'
        )  # (Ns,Ne,Ne*Nk,Nc) @ (Nc,Np) -flatten-> (Ns,Ne,Ne*Nk*Np)
        X_trans_y = np.reshape(
            a=fast_stan_4d(evY_trans @ X_test[nte]),
            newshape=(n_subjects, n_events, -1),
            order='C'
        )  # (Ns,Ne,Ne*Nk,Nc) @ (Nc,Np) -flatten-> (Ns,Ne,Ne*Nk*Np)
        X_temp = np.tile(
            A=np.reshape(
                a=fast_stan_2d(ew_target @ X_test[nte]),
                newshape=-1,
                order='C'
            ),
            reps=(n_events, 1)
        )  # (Ne*Nk,Nc) @ (Nc,Np) -flatten-> -repeat-> (Ne,Ne*Nk*Np)

        # rho 1 & 2: transferred pattern matching
        rho_temp[nte, :, 0] = np.sum(
            a=weight_euX * fast_corr_3d(X=X_trans_x, Y=euX_source),
            axis=0
        )  # (Ns,Ne,Ne*Nk*Np) -corr-> (Ns,Ne) -sum-> (Ne,)
        rho_temp[nte, :, 1] = np.sum(
            a=weight_evY * fast_corr_3d(X=X_trans_y, Y=evY_source),
            axis=0
        )  # (Ns,Ne,Ne*Nk*Np) -corr-> (Ns,Ne) -sum-> (Ne,)

        # rho 3 & 4: target-domain pattern matching
        # (Ne,Ne*Nk*Np) -corr-> (Ne,)
        rho_temp[nte, :, 2] = fast_corr_2d(X=X_temp, Y=euX_target)
        rho_temp[nte, :, 2] = fast_corr_2d(X=X_temp, Y=evY_target)
    rho_temp /= n_points  # real Pearson correlation coefficients in scale
    features = {
        'rho_temp': rho_temp,
        'rho': combine_feature([
            rho_temp[..., 0],
            rho_temp[..., 1],
            rho_temp[..., 2],
            rho_temp[..., 3]
        ])
    }
    return features


class TNSRE_20233250953(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    Transfer learning algorithm TNSRE-20233250953. [1]

    author: BrynhildrWu <brynhildrwu@gmail.com>

    Parameters
    ----------
    standard : *bool.*
        Use standard model. Defaults to ``True``.
    ensemble : *bool.*
        Use ensemble model. Defaults to ``True``.
    n_components : *int.*
        Number of eigenvectors picked as filters. Defaults to ``1``.

    Attributes
    ----------
    X_train : *ndarray of shape (Ne*Nt,Nc,Np).*
        Sklearn-style target-domain training dataset. *Nt>=2*.
    y_train : *ndarray of shape (Ne*Nt,).*
        Labels for ``X_train``.
    X_source : *List[ndarray] of shape Ns*(Ne*Nt,Nc,Np).*
        Source dataset.
    y_source : *List[ndarray]): Ns*(Ne*Nt,).*
        Labels for ``X_source``.
    sine_template : *ndarray of shape (Ne,2*Nh,Np).*
        Sinusoidal templates. Defaults to ``None``.
    n_subjects : int.
        Number of source-domain subjects.
    source_info : *List[Dict[str, Any]].*
        Source-domain information. See details in ``generate_data_info()``.
    target_info : *Dict[str, ndarray].*
        Target-domain information. See details in ``generate_data_info()``.
    event_type : *ndarray of shape (Ne,).*
        Unique labels.
    source_model : *Dict[str, ndarray].*
        The keys and items are:
            * ``'euX_source'`` : *ndarray of shape (Ns,Ne,Ne*Nk,Np).*
            * ``'evY_source'`` : *ndarray of shape (Ns,Ne,Ne*Nk,Np).*
    trans_model : *Dict[str, ndarray].*
        The keys and items are:
            * ``'eu_trans'`` : *ndarray of shape (Ns,Ne,Ne*Nk,Nc).*
            * ``'ev_trans'`` : *ndarray of shape (Ns,Ne,Ne*Nk,Nc).*
            * ``'dist_euX'`` : *ndarray of shape (Ns,Ne).*
            * ``'dist_evY'`` : *ndarray of shape (Ns,Ne).*
            * ``'weight_euX'`` : *ndarray of shape (Ns,Ne).*
            * ``'weight_evY'`` : *ndarray of shape (Ns,Ne).*
    target_model : *Dict[str, ndarray].*
        See details in ``tnsre_20233250953_kernel()``.
    features : *Dict[str, ndarray].*
        See details in ``tnsre_20233250953_feature()``.
    y_pred : *ndarray of shape (Ne*Nte,) or int.*
        Predict label(s)

    References
    ----------
    .. [1] Y. Zhang, S. Q. Xie, C. Shi, J. Li and Z. -Q. Zhang, "Cross-Subject Transfer
        Learning for Boosting Recognition Performance in SSVEP-Based BCIs," in IEEE
        Transactions on Neural Systems and Rehabilitation Engineering, vol. 31,
        pp. 1574-1583, 2023, doi: 10.1109/TNSRE.2023.3250953.

    Tip
    ----------
    .. code-block:: python
        :linenos:
        :emphasize-lines: 2
        :caption: A example using TNSRE_20233250953

        from metabci.brainda.algorithms.transfer_learning import TNSRE_20233250953
        model = TNSRE_20233250953(n_components=1)
        model.fit(
            X_source=X_source,
            y_source=y_source,
            X_train=X_train,
            y_train=y_train,
            sine_template=sine_template
        )
        y_pred = model.predict(X_test=X_test)
    """
    def __init__(self, standard=True, ensemble=True, n_components=1):
        """Basic configuration."""
        # config model
        self.n_components = n_components
        self.standard = standard
        self.ensemble = ensemble

    def intra_source_training(self):
        """Intra-domain model training for source dataset."""
        # basic information & initialization
        # self.intra_model_source = []
        euX_source, evY_source = [], []  # List[ndarray]: Ns*(Ne,Ne*Nk,Np)

        # obtain source model
        for nsub in range(self.n_subjects):
            intra_model = tnsre_20233250953_kernel(
                X_train=self.X_source[nsub],
                y_train=self.y_source[nsub],
                sine_template=self.sine_template,
                n_components=self.n_components
            )
            # self.intra_model_source.append(intra_model)
            euX_source.append(intra_model['euX'])  # (Ne,Ne*Nk,Np)
            evY_source.append(intra_model['evY'])  # (Ne,Ne*Nk,Np)
        self.source_model = {
            'euX_source': np.stack(euX_source),
            'evY_source': np.stack(evY_source)
        }  # (Ns,Ne,Ne*Nk,Np)

    def transfer_learning(self):
        """Transfer learning process."""
        # basic information
        n_events = self.target_info['n_events']  # Ne
        n_chans = self.target_info['n_chans']  # Nc
        n_train = self.target_info['n_train']  # [Nt1,Nt2,...]

        # obtain transfer model (partial)
        eu_trans, ev_trans = [], []  # List[ndarray]: Ns*(Ne,Ne*Nk,Nc)
        for nsub in range(self.n_subjects):
            euX = self.source_model['euX_source'][nsub]  # (Ne,Ne*Nk,Np)
            evY = self.source_model['evY_source'][nsub]  # (Ne,Ne*Nk,Np)

            # LST alignment
            eu_trans.append(np.zeros((n_events, euX.shape[1], n_chans)))  # (Ne,Ne*Nk,Nc)
            ev_trans.append(np.zeros((n_events, evY.shape[1], n_chans)))  # (Ne,Ne*Nk,Nc)
            for ne, et in enumerate(self.event_type):
                X_temp = self.X_train[self.y_train == et]  # (Nt,Nc,Np)
                train_trials = n_train[ne]
                for tt in range(train_trials):  # w = min ||b - A w||
                    uX_trans_temp, _, _, _ = sLA.lstsq(a=X_temp[tt].T, b=euX[ne].T)
                    vY_trans_temp, _, _, _ = sLA.lstsq(a=X_temp[tt].T, b=evY[ne].T)
                    eu_trans[nsub][ne] += uX_trans_temp.T
                    ev_trans[nsub][ne] += vY_trans_temp.T
                eu_trans[nsub][ne] /= train_trials
                ev_trans[nsub][ne] /= train_trials
        self.trans_model = {
            'eu_trans': np.stack(eu_trans),
            'ev_trans': np.stack(ev_trans)
        }  # (Ns,Ne,Ne*Nk,Nc)

    def dist_calc(self):
        """Calculate the spatial distances between source and target domain."""
        # load in models & basic information
        n_events = self.target_info['n_events']  # Ne
        n_train = self.target_info['n_train']  # [Nt1,Nt2,...]
        eu_trans = self.trans_model['eu_trans']  # (Ns,Ne,Ne*Nk,Nc)
        ev_trans = self.trans_model['ev_trans']  # (Ns,Ne,Ne*Nk,Nc)
        euX_source = self.source_model['euX_source']  # (Ns,Ne,Ne*Nk,Np)
        evY_source = self.source_model['evY_source']  # (Ns,Ne,Ne*Nk,Np)

        # reshape for fast computing: (Ns,Ne,Ne*Nk,Np) -reshape-> (Ns,Ne,Ne*Nk*Np)
        euX_source = np.reshape(euX_source, (self.n_subjects, n_events, -1), 'C')
        evY_source = np.reshape(evY_source, (self.n_subjects, n_events, -1), 'C')

        # calculate distances
        dist_euX = np.zeros((self.n_subjects, n_events))  # (Ns,Ne)
        dist_evY = np.zeros_like(self.dist_euX)
        for ne, et in enumerate(self.event_type):
            X_temp = self.X_train[self.y_train == et]  # (Nt,Nc,Np)
            train_trials = n_train[ne]  # Nt
            for tt in range(train_trials):
                X_trans_x = np.reshape(
                    a=eu_trans[:, ne, ...] @ X_temp[tt],
                    newshape=(self.n_subjects, -1),
                    order='C'
                )  # (Ns,Ne*Nk,Nc) @ (Nc,Np) -reshape-> (Ns,Ne*Nk*Np)
                X_trans_y = np.reshape(
                    a=ev_trans[:, ne, ...] @ X_temp[tt],
                    newshape=(self.n_subjects, -1),
                    order='C'
                )  # (Ns,Ne*Nk,Np) @ (Nc,Np) -reshape-> (Ns,Ne*Nk*Np)
                dist_euX[:, ne] += fast_corr_2d(X=X_trans_x, Y=euX_source[:, ne, :])
                dist_evY[:, ne] += fast_corr_2d(X=X_trans_y, Y=evY_source[:, ne, :])
        self.trans_model['dist_euX'] = dist_euX
        self.trans_model['dist_evY'] = dist_evY

    def weight_calc(self):
        """Optimize the transfer weights."""
        dist_euX = self.trans_model['dist_euX']  # (Ns,Ne)
        dist_evY = self.trans_model['dist_evY']  # (Ns,Ne)
        self.trans_model['weight_euX'] = dist_euX / np.sum(dist_euX, axis=0, keepdims=True)
        self.trans_model['weight_evY'] = dist_evY / np.sum(dist_evY, axis=0, keepdims=True)

    def intra_target_training(self):
        """Intra-domain model training for target dataset."""
        self.target_model = tnsre_20233250953_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=self.sine_template,
            n_components=self.n_components
        )

    def fit(self, X_source, y_source, X_train, y_train, sine_template):
        """
        Train model.

        Parameters
        ----------
        X_source : *List[ndarray] of shape Ns*(Ne*Nt,Nc,Np).*
            Source dataset.
        y_source : *List[ndarray] of shape Ns*(Ne*Nt,).*
            Labels for ``X_source``.
        X_train : *ndarray of shape (Ne*Nt,Nc,Np).*
            Sklearn-style target-domain training dataset. *Nt>=2*.
        y_train : *ndarray of shape (Ne*Nt,).*
            Labels for ``X_train``.
        sine_template : *ndarray of shape (Ne,2*Nh,Np).*
            Sinusoidal templates. Defaults to ``None``.
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.X_source = X_source
        self.y_source = y_source
        self.sine_template = sine_template

        # basic information of source & target domain
        self.n_subjects = len(self.X_source)
        self.source_info = []
        for nsub in range(self.n_subjects):
            self.source_info.append(
                generate_data_info(X=self.X_source[nsub], y=self.y_source[nsub])
            )
        self.target_info = generate_data_info(X=self.X_train, y=self.y_train)
        self.event_type = self.source_info[0]['event_type']

        # main process
        self.intra_source_training()
        self.transfer_learning()
        self.dist_calc()
        self.weight_calc()
        self.intra_target_training()

    def transform(self, X_test):
        """
        Transform test dataset to features.

        Parameters
        --------------
        X_test : *ndarray of shape (Ne*Nte,Nc,Np).*
            Test dataset.

        Returns (Dict)
        --------------
        rho_temp : *ndarray of shape (Ne*Nte,Ne,4).*
            4-D features.
        rho : *ndarray of shape (Ne*Nte,Ne).*
            Integrated features.
        """
        return tnsre_20233250953_feature(
            X_test=X_test,
            source_model=self.source_model,
            trans_model=self.trans_model,
            target_model=self.target_model
        )

    def predict(self, X_test):
        """
        Predict test data.

        Parameters
        ----------
        X_test : *ndarray of shape (Ne*Nte,Nc,Np).*
            Test dataset.

        Returns
        ----------
        y_pred : *ndarray of shape (Ne*Nte,) or int.*
            Predict label(s)
        """
        self.features = self.transform(X_test)
        self.y_pred = self.event_type[np.argmax(self.features['rho'], axis=-1)]
        return self.y_pred
