import numpy as np

from config import Config
from sim import simulate_truth, make_sensors, sync_gyro, DEFAULT_SCRIPT, STRAIGHT_SCRIPT
from ekf import EKF, wrap
from baselines import encoder_only, encoder_gyro


def run_once(cfg, seed, truth=None):
    if truth is None:
        truth = simulate_truth(cfg)
    rng = np.random.default_rng(seed)
    s = make_sensors(truth, cfg, rng)

    keep, gyro = sync_gyro(s['t_enc'], s['t_gyro'], s['gyro'])
    t, wL, wR, true = s['t_enc'][keep], s['wL'][keep], s['wR'][keep], s['true'][keep]
    return run_all_methods(t, wL, wR, gyro, true, cfg)


def run_ekf(t, z, mu0, cfg, gps=None, sigma_gps=None, gate=None, bias=False):
    # gps: N x 2, NaN where there is no fix
    dt = np.diff(t, prepend=t[0])
    N = len(t)

    n = 6 if bias else 5
    mu0 = np.r_[mu0[:5], 0.0][:n]
    Sigma0 = cfg.sigma0_var * np.eye(n)
    if bias:
        Sigma0[5, 5] = cfg.sigma_b0**2
    ekf = EKF(mu0, Sigma0, cfg)
    mu = np.zeros((N, n))
    Sigma = np.zeros((N, n, n))
    innov = np.zeros((N, 3))
    nis = np.full(N, np.nan)
    used = np.zeros(N, dtype=bool)
    mu[0], Sigma[0] = ekf.mu, ekf.Sigma
    for k in range(1, N):
        ekf.predict(dt[k])
        y, S = ekf.update(z[k])
        nis[k] = y @ np.linalg.solve(S, y)
        if gps is not None and not np.isnan(gps[k, 0]):
            used[k] = ekf.update_gps(gps[k], sigma_gps, gate)
        mu[k], Sigma[k], innov[k] = ekf.mu, ekf.Sigma, y
    return mu, Sigma, innov, nis, used


def run_all_methods(t, wL, wR, gyro, true, cfg):
    z = np.column_stack([wL, wR, gyro])
    dt = np.diff(t, prepend=t[0])
    mu, Sigma, innov, nis, _ = run_ekf(t, z, true[0], cfg)
    mu_b, Sigma_b, _, nis_b, _ = run_ekf(t, z, true[0], cfg, bias=True)

    enc = encoder_only(true[0, :3], wL, wR, dt, cfg)
    encg = encoder_gyro(true[0, :3], wL, wR, gyro, dt, cfg)

    return dict(t=t, true=true, z=z, mu=mu, Sigma=Sigma, innov=innov, nis=nis,
                mu_b=mu_b, Sigma_b=Sigma_b, nis_b=nis_b, enc=enc, enc_gyro=encg)


def pose_error(true, est):
    e = true[:, :3] - est[:, :3]
    e[:, 2] = wrap(e[:, 2])
    return e


def nees_pose(res, key='mu'):
    e = pose_error(res['true'], res[key])
    P = res['Sigma' + key[2:]][:, :3, :3]
    return np.einsum('ki,kij,kj->k', e, np.linalg.inv(P), e)


def final_errors(res):
    out = {}
    for name in ['mu', 'mu_b', 'enc', 'enc_gyro', 'mu_cal', 'mu_gps_nb', 'mu_gps']:
        if name not in res:
            continue
        e = pose_error(res['true'], res[name])[-1]
        out[name] = (np.hypot(e[0], e[1]), abs(e[2]))
    return out


def monte_carlo(cfg, seeds, truth=None):
    if truth is None:
        truth = simulate_truth(cfg)
    return [run_once(cfg, s, truth) for s in seeds]


def chi2_bounds(dof, n_runs=1, p=0.95):
    # Wilson-Hilferty approximation (no scipy)
    k = dof * n_runs
    z = 1.959964 if p == 0.95 else 2.575829
    def q(zz):
        return k * (1 - 2 / (9 * k) + zz * np.sqrt(2 / (9 * k)))**3
    return q(-z) / n_runs, q(z) / n_runs


SCENARIOS = {
    'default': (DEFAULT_SCRIPT, 'outputs', {}),
    'straight': (STRAIGHT_SCRIPT, 'trajectory_1', {}),
    'bias': (DEFAULT_SCRIPT, 'outputs_bias', {'gyro_bias': 0.01}),
}

if __name__ == '__main__':
    # python run.py [straight | bias]
    import sys
    from dataclasses import replace
    import plots
    name = sys.argv[1] if len(sys.argv) > 1 else 'default'
    script, out, changes = SCENARIOS[name]
    plots.make_all(replace(Config(), **changes), script, out)
    if name == 'bias':
        plots.bias_study(Config(), script, range(20))
