import numpy as np

# state mu = [x, y, theta, v, omega]
# R = process noise, Q = measurement noise


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def g(mu, dt):
    x, y, th, v, w = mu
    return np.array([
        x + v * dt * np.cos(th),
        y + v * dt * np.sin(th),
        th + w * dt,
        v,
        w,
    ])


def G(mu, dt):
    _, _, th, v, _ = mu
    return np.array([
        [1, 0, -v * dt * np.sin(th), dt * np.cos(th), 0],
        [0, 1,  v * dt * np.cos(th), dt * np.sin(th), 0],
        [0, 0, 1,                    0,              dt],
        [0, 0, 0,                    1,               0],
        [0, 0, 0,                    0,               1],
    ])


# z = [omega_L, omega_R, omega_gyro]
def h(mu, r, L):
    _, _, _, v, w = mu
    return np.array([
        (v - w * L / 2) / r,
        (v + w * L / 2) / r,
        w,
    ])


def H(r, L):
    return np.array([
        [0, 0, 0, 1 / r, -L / (2 * r)],
        [0, 0, 0, 1 / r,  L / (2 * r)],
        [0, 0, 0, 0,      1],
    ])


class EKF:
    def __init__(self, mu0, Sigma0, cfg):
        self.mu = np.array(mu0, dtype=float)
        self.Sigma = np.array(Sigma0, dtype=float)
        self.cfg = cfg
        self.Q = np.diag([cfg.sigma_enc**2, cfg.sigma_enc**2, cfg.sigma_g**2])

    def R(self, dt):
        c = self.cfg
        return np.diag([c.s_xy**2, c.s_xy**2, c.s_th**2,
                        (c.sigma_a * dt)**2, (c.sigma_alpha * dt)**2])

    def predict(self, dt):
        Gt = G(self.mu, dt)
        self.mu = g(self.mu, dt)                              # mu_bar = g(mu)
        self.Sigma = Gt @ self.Sigma @ Gt.T + self.R(dt)      # Sigma_bar = G Sigma G^T + R
        self.mu[2] = wrap(self.mu[2])

    def update(self, z):
        return self.correct(z, h(self.mu, self.cfg.r, self.cfg.L), H(self.cfg.r, self.cfg.L), self.Q)

    def correct(self, z, z_pred, Ht, Qt):
        y = z - z_pred                                        # innovation
        S = Ht @ self.Sigma @ Ht.T + Qt                       # S = H Sigma_bar H^T + Q
        K = self.Sigma @ Ht.T @ np.linalg.inv(S)              # K = Sigma_bar H^T S^-1
        self.mu = self.mu + K @ y
        I_KH = np.eye(len(self.mu)) - K @ Ht
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + K @ Qt @ K.T  # Joseph form
        self.mu[2] = wrap(self.mu[2])
        return y, S
