import numpy as np

# state mu = [x, y, theta, v, omega] or [x, y, theta, v, omega, b]
# R = process noise, Q = measurement noise


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def g(mu, dt):
    x, y, th, v, w = mu[:5]
    out = mu.copy()
    out[:3] = [x + v * dt * np.cos(th), y + v * dt * np.sin(th), th + w * dt]
    return out


def G(mu, dt):
    th, v = mu[2], mu[3]
    Gt = np.eye(len(mu))
    Gt[0, 2], Gt[0, 3] = -v * dt * np.sin(th), dt * np.cos(th)
    Gt[1, 2], Gt[1, 3] = v * dt * np.cos(th), dt * np.sin(th)
    Gt[2, 4] = dt
    return Gt


# z = [omega_L, omega_R, omega_gyro], gyro reads omega + b
def h(mu, r, L):
    v, w = mu[3], mu[4]
    b = mu[5] if len(mu) > 5 else 0.0
    return np.array([
        (v - w * L / 2) / r,
        (v + w * L / 2) / r,
        w + b,
    ])


def H(r, L, n=5):
    Ht = np.zeros((3, n))
    Ht[0, 3:5] = [1 / r, -L / (2 * r)]
    Ht[1, 3:5] = [1 / r, L / (2 * r)]
    Ht[2, 4] = 1
    if n > 5:
        Ht[2, 5] = 1
    return Ht


# gps measures x, y directly
def H_gps(n=5):
    return np.eye(2, n)


class EKF:
    def __init__(self, mu0, Sigma0, cfg):
        self.mu = np.array(mu0, dtype=float)
        self.Sigma = np.array(Sigma0, dtype=float)
        self.cfg = cfg
        self.n = len(self.mu)
        self.Q = np.diag([cfg.sigma_enc**2, cfg.sigma_enc**2, cfg.sigma_g**2])

    def R(self, dt):
        c = self.cfg
        d = [c.s_xy**2, c.s_xy**2, c.s_th**2, (c.sigma_a * dt)**2, (c.sigma_alpha * dt)**2]
        if self.n > 5:
            d.append(c.sigma_b**2 * dt)          # random walk
        return np.diag(d)

    def predict(self, dt):
        Gt = G(self.mu, dt)
        self.mu = g(self.mu, dt)                              # mu_bar = g(mu)
        self.Sigma = Gt @ self.Sigma @ Gt.T + self.R(dt)      # Sigma_bar = G Sigma G^T + R
        self.mu[2] = wrap(self.mu[2])

    def update(self, z):
        c = self.cfg
        return self.correct(z, h(self.mu, c.r, c.L), H(c.r, c.L, self.n), self.Q)

    def update_gps(self, z_xy, sigma_gps, gate=None):
        Hg = H_gps(self.n)
        Qg = sigma_gps**2 * np.eye(2)
        y = z_xy - self.mu[:2]
        S = Hg @ self.Sigma @ Hg.T + Qg
        if gate is not None and y @ np.linalg.solve(S, y) > gate:
            return False                                      # outlier, skip it
        self.correct(z_xy, self.mu[:2], Hg, Qg)
        return True

    def correct(self, z, z_pred, Ht, Qt):
        y = z - z_pred                                        # innovation
        S = Ht @ self.Sigma @ Ht.T + Qt                       # S = H Sigma_bar H^T + Q
        K = self.Sigma @ Ht.T @ np.linalg.inv(S)              # K = Sigma_bar H^T S^-1
        self.mu = self.mu + K @ y
        I_KH = np.eye(self.n) - K @ Ht
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + K @ Qt @ K.T  # Joseph form
        self.mu[2] = wrap(self.mu[2])
        return y, S
