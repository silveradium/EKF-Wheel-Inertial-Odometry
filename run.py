"""Single run and Monte Carlo runs. `python run.py` makes all the figures."""
import numpy as np

from config import Config
from sim import simulate_truth, make_sensors, sync_gyro, DEFAULT_SCRIPT, STRAIGHT_SCRIPT
from ekf import EKF, wrap
from baselines import encoder_only, encoder_gyro


def run_once(cfg, seed, truth=None):
    """Simulate sensors, run the EKF and the two baselines.

    `truth` can be passed in to save time (the true motion has no randomness,
    only the sensors do).
    """
    if truth is None:
        truth = simulate_truth(cfg)
    rng = np.random.default_rng(seed)
    s = make_sensors(truth, cfg, rng)

    # put the gyro on the encoder timeline
    keep, gyro = sync_gyro(s['t_enc'], s['t_gyro'], s['gyro'])
    t, wL, wR, true = s['t_enc'][keep], s['wL'][keep], s['wR'][keep], s['true'][keep]
    return run_all_methods(t, wL, wR, gyro, true, cfg)


def run_all_methods(t, wL, wR, gyro, true, cfg):
    """Run the EKF and the two baselines on synced sensor data.

    true[0] (the start state) is used to initialise everything.
    Also used for the real NCLT data.
    """
    z = np.column_stack([wL, wR, gyro])
    dt = np.diff(t, prepend=t[0])     # dt[k] = t[k] - t[k-1]
    N = len(t)

    # EKF: start at the true state with a small covariance
    ekf = EKF(true[0], cfg.sigma0_var * np.eye(5), cfg)
    mu = np.zeros((N, 5))
    Sigma = np.zeros((N, 5, 5))
    innov = np.zeros((N, 3))
    nis = np.full(N, np.nan)
    mu[0], Sigma[0] = ekf.mu, ekf.Sigma
    for k in range(1, N):
        ekf.predict(dt[k])
        y, S = ekf.update(z[k])
        mu[k], Sigma[k], innov[k] = ekf.mu, ekf.Sigma, y
        nis[k] = y @ np.linalg.solve(S, y)

    enc = encoder_only(true[0, :3], wL, wR, dt, cfg)
    encg = encoder_gyro(true[0, :3], wL, wR, gyro, dt, cfg)

    return dict(t=t, true=true, z=z, mu=mu, Sigma=Sigma, innov=innov, nis=nis,
                enc=enc, enc_gyro=encg)


def pose_error(true, est):
    """Error [ex, ey, etheta] with the heading error wrapped."""
    e = true[:, :3] - est[:, :3]
    e[:, 2] = wrap(e[:, 2])
    return e


def nees_pose(res):
    """NEES on the pose (x, y, theta): e^T Sigma^-1 e. 3 degrees of freedom."""
    e = pose_error(res['true'], res['mu'])
    P = res['Sigma'][:, :3, :3]
    return np.einsum('ki,kij,kj->k', e, np.linalg.inv(P), e)


def final_errors(res):
    """Final position error [m] and |heading error| [rad] for each method."""
    out = {}
    for name in ['mu', 'enc', 'enc_gyro']:
        e = pose_error(res['true'], res[name])[-1]
        out[name] = (np.hypot(e[0], e[1]), abs(e[2]))
    return out


def monte_carlo(cfg, seeds, truth=None):
    if truth is None:
        truth = simulate_truth(cfg)
    return [run_once(cfg, s, truth) for s in seeds]


def chi2_bounds(dof, n_runs=1, p=0.95):
    """Two-sided chi-square bounds for the average of n_runs NEES/NIS values.

    Uses the Wilson-Hilferty approximation so we only need numpy.
    """
    k = dof * n_runs
    z = 1.959964 if p == 0.95 else 2.575829
    def q(zz):
        return k * (1 - 2 / (9 * k) + zz * np.sqrt(2 / (9 * k)))**3
    return q(-z) / n_runs, q(z) / n_runs


# name -> (teleop script, output folder)
SCENARIOS = {
    'default': (DEFAULT_SCRIPT, 'outputs'),
    'straight': (STRAIGHT_SCRIPT, 'trajectory_1'),
}

if __name__ == '__main__':
    # usage: python run.py            (default script)
    #        python run.py straight   (straight line)
    import sys
    import plots
    script, out = SCENARIOS[sys.argv[1] if len(sys.argv) > 1 else 'default']
    plots.make_all(Config(), script, out)
