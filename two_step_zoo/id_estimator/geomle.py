"""
Code from: https://github.com/ppope/dimensions

This file modified from:
* https://github.com/stat-ml/GeoMLE/blob/master/geomle/geomle.py
"""

import numpy as np
import pandas as pd
from functools import partial

import torch
from .utils import KNNComputerNoCheck, update_nn
from sklearn.linear_model import Ridge


def mle_center(k=5, dist=None):
    """
    Returns Levina-Bickel dimensionality estimation
    """
    dist = dist[:, 0:k]
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k - 1])
    d = d.sum(axis=1) / (k - 2)
    d = 1. / d
    intdim_sample = d
    Rs = dist[:, -1]

    return intdim_sample, Rs


def intrinsic_dim_sample_wise_double_mle(k=5, dist=None):
    """
    Returns Levina-Bickel dimensionality estimation and the correction by MacKay-Ghahramani
    """
    dist = dist[:, 0:k]
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k - 1])
    d = d.sum(axis=1) / (k - 2)
    inv_mle = d.copy()
    d = 1. / d
    mle = d
    Rs = dist[:, -1]
    inv_Rs = 1. / dist[:, -1]
    return mle, inv_mle, Rs, inv_Rs


def tolist(x):
    if type(x) in {int, float}:
        return [x]
    if type(x) in {list, tuple}:
        return list(x)
    if type(x) == np.ndarray:
        return x.tolist()


def fit_poly_reg(x, y, w=None, degree=(1, 2), alpha=5e-3):
    """
    Fit regression and return f(0)
    """
    X = np.array([x ** i for i in tolist(degree)]).T
    lm = Ridge(alpha=alpha)
    lm.fit(X, y, w)
    return lm.intercept_


def _func(df, degree, alpha, inv_mle=False):
    gr_df = df.groupby('k')
    if inv_mle:
        d = gr_df['inv_mle_dim'].mean().values
        std = gr_df['inv_mle_dim'].std().values
        R = gr_df['inv_mle_R'].mean().values
    else:
        d = gr_df['dim'].mean().values
        std = gr_df['dim'].std().values
        R = gr_df['R'].mean().values
    if np.isnan(std).any():
        std = np.ones_like(std)
    return fit_poly_reg(R, d, std**-1, degree=degree, alpha=alpha)


def drop_zero_values(dist):
    mask = dist[:, 0] == 0
    dist[mask] = np.hstack([dist[mask][:, 1:], dist[mask][:, 0:1]])
    dist = dist[:, :-1]
    assert np.all(dist > 0)
    return dist


def geomle(full_dataset, k1=10, k2=40, nb_iter1=10, nb_iter2=20, degree=(1, 2),
           alpha=5e-3, ver='GeoMLE', random_state=None, debug=False, args=None):
    """
    Returns regression dimensionality estimates for k = k1..k2 averaged over bootstrap samples.
    """
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)
    nb_examples = len(full_dataset)

    result = []
    data_reg = []
    for _ in range(nb_iter1):
        dim_all, R_all, k_all, idx_all = [], [], [], []
        inv_dim_all = []
        inv_R_all = []
        for _ in range(nb_iter2):
            idx = np.unique(rng.randint(0, nb_examples - 1, size=nb_examples))
            X_bootstrap = torch.utils.data.Subset(full_dataset, idx)
            nn_computer = KNNComputerNoCheck(len(full_dataset), K=k2 + 1).cuda()

            anchor_loader = torch.utils.data.DataLoader(
                full_dataset,
                batch_size=args.bsize, shuffle=False,
                num_workers=args.n_workers
            )
            bootstrap_loader = torch.utils.data.DataLoader(
                X_bootstrap,
                batch_size=args.bsize, shuffle=False,
                num_workers=args.n_workers
            )

            update_nn(anchor_loader, 0, bootstrap_loader, 0, nn_computer)

            dist = nn_computer.min_dists.cpu().numpy()
            dist = drop_zero_values(dist)
            dist = dist[:, :k2]
            assert np.all(dist > 0)

            for k in range(k1, k2 + 1):
                dim, R = mle_center(k, dist)
                dim_all += dim.tolist()
                R_all += R.tolist()
                idx_all += list(range(nb_examples))
                k_all += [k] * dim.shape[0]

        data = {'dim': dim_all, 'R': R_all, 'idx': idx_all, 'k': k_all}
        df = pd.DataFrame(data)
        if ver.lower() == 'geomle':
            func = partial(_func, degree=degree, alpha=alpha, inv_mle=False)
            reg = df.groupby('idx').apply(func).values
            if args.inv_mle:
                reg = 1. / (1. / reg).mean()
            else:
                reg = reg.mean()
            data_reg.append(df)
        elif ver.lower() == 'fastgeomle':
            df_gr = df.groupby(['idx', 'k']).mean()[['R', 'dim']]
            R = df_gr.groupby('k').R.mean()
            d = df_gr.groupby('k').dim.mean()
            std = df_gr.groupby('k').dim.std()
            reg = fit_poly_reg(R, d, std ** -1, degree=degree, alpha=alpha)
            data_reg.append((R, d, std))
        else:
            raise ValueError(f"Unknown mode {ver}")

        reg = 0 if reg < 0 else reg
        result.append(reg)

    if debug:
        return np.array(result), data_reg
    return np.array(result)
