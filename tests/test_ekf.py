import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace
import numpy as np

from config import Config
from ekf import EKF, g, G, h, H, wrap
from sim import simulate_truth
from run import run_once, pose_error


def numeric_jacobian(f, mu, eps=1e-6):
    J = np.zeros((len(f(mu)), len(mu)))
    for i in range(len(mu)):
        d = np.zeros(len(mu)); d[i] = eps
        J[:, i] = (f(mu + d) - f(mu - d)) / (2 * eps)
    return J


def test_G_matches_finite_difference():
    mu = np.array([0.3, -0.2, 0.7, 0.15, -0.4])
    assert np.allclose(G(mu, 0.02), numeric_jacobian(lambda m: g(m, 0.02), mu), atol=1e-8)


def test_H_matches_finite_difference():
    cfg = Config()
    mu = np.array([0.3, -0.2, 0.7, 0.15, -0.4])
    assert np.allclose(H(cfg.r, cfg.L), numeric_jacobian(lambda m: h(m, cfg.r, cfg.L), mu), atol=1e-6)


def test_wrap():
    assert np.isclose(wrap(0.5), 0.5)
    assert np.isclose(wrap(3 * np.pi / 2), -np.pi / 2)
    assert np.isclose(wrap(-3 * np.pi / 2), np.pi / 2)
    assert np.isclose(wrap(10 * np.pi + 0.1), 0.1)
    a = np.linspace(-20, 20, 101)
    assert np.all((wrap(a) >= -np.pi) & (wrap(a) < np.pi))


def test_stacked_equals_sequential():
    cfg = Config()
    mu0 = np.array([0.1, 0.2, 0.3, 0.1, 0.2])
    Sigma0 = np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2])
    z = np.array([3.5, 4.1, 0.25])
    Hm = H(cfg.r, cfg.L)

    a = EKF(mu0, Sigma0, cfg)
    a.predict(0.02)
    a.update(z)

    b = EKF(mu0, Sigma0, cfg)
    b.predict(0.02)
    Q = b.Q
    b.correct(z[:2], h(b.mu, cfg.r, cfg.L)[:2], Hm[:2], Q[:2, :2])   # encoders
    b.correct(z[2:], h(b.mu, cfg.r, cfg.L)[2:], Hm[2:], Q[2:, 2:])   # then gyro

    assert np.allclose(a.mu, b.mu)
    assert np.allclose(a.Sigma, b.Sigma)


def test_zero_noise_matches_truth():
    # very fine encoders, no gyro noise, perfect track width
    cfg = replace(Config(), cpr=10**7, sigma_wheel=0.0, sigma_g=1e-6, L_true_scale=1.0, T=20.0)
    res = run_once(cfg, 0, simulate_truth(cfg))
    e = pose_error(res['true'], res['mu'])
    assert np.max(np.hypot(e[:, 0], e[:, 1])) < 0.01   # < 1 cm
    assert np.max(np.abs(e[:, 2])) < 0.01               # < 0.6 deg
