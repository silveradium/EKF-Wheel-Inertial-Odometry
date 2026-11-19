import numpy as np
from ekf import wrap


def integrate(x0, v, w, dt):
    pose = np.zeros((len(v), 3))
    pose[0] = x0
    for k in range(1, len(v)):
        x, y, th = pose[k - 1]
        pose[k] = [x + v[k] * dt[k] * np.cos(th),
                   y + v[k] * dt[k] * np.sin(th),
                   wrap(th + w[k] * dt[k])]
    return pose


def encoder_only(x0, wL, wR, dt, cfg):
    v = cfg.r * (wR + wL) / 2
    w = cfg.r * (wR - wL) / cfg.L
    return integrate(x0, v, w, dt)


def encoder_gyro(x0, wL, wR, gyro, dt, cfg):
    v = cfg.r * (wR + wL) / 2
    return integrate(x0, v, gyro, dt)
