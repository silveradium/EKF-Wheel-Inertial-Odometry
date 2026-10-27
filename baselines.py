"""Two simple dead reckoning baselines to compare against the EKF."""
import numpy as np
from ekf import wrap


def integrate(x0, v, w, dt):
    """Integrate pose with speeds v[k], w[k] used for the step k-1 -> k.

    x0 = [x, y, theta]. Returns an N x 3 array of poses (row 0 = x0).
    """
    pose = np.zeros((len(v), 3))
    pose[0] = x0
    for k in range(1, len(v)):
        x, y, th = pose[k - 1]
        pose[k] = [x + v[k] * dt[k] * np.cos(th),
                   y + v[k] * dt[k] * np.sin(th),
                   wrap(th + w[k] * dt[k])]
    return pose


def encoder_only(x0, wL, wR, dt, cfg):
    """Baseline 1: v and omega both from the wheel speeds."""
    v = cfg.r * (wR + wL) / 2
    w = cfg.r * (wR - wL) / cfg.L
    return integrate(x0, v, w, dt)


def encoder_gyro(x0, wL, wR, gyro, dt, cfg):
    """Baseline 2: v from the wheels, heading from integrating the raw gyro."""
    v = cfg.r * (wR + wL) / 2
    return integrate(x0, v, gyro, dt)
