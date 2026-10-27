"""EKF for a differential-drive robot (wheel encoders + gyro).

State:  mu = [x, y, theta, v, omega]
Notation (Probabilistic Robotics, Thrun):
  g, G   motion model and its Jacobian
  h, H   measurement model and its Jacobian
  R      process noise covariance
  Q      measurement noise covariance
"""
import numpy as np


def wrap(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------- motion model ----------------

def g(mu, dt):
    """Constant velocity model, Euler step with the heading at the start of the step."""
    x, y, th, v, w = mu
    return np.array([
        x + v * dt * np.cos(th),
        y + v * dt * np.sin(th),
        th + w * dt,
        v,
        w,
    ])


def G(mu, dt):
    """Jacobian of g with respect to the state, evaluated at mu."""
    _, _, th, v, _ = mu
    return np.array([
        [1, 0, -v * dt * np.sin(th), dt * np.cos(th), 0],
        [0, 1,  v * dt * np.cos(th), dt * np.sin(th), 0],
        [0, 0, 1,                    0,              dt],
        [0, 0, 0,                    1,               0],
        [0, 0, 0,                    0,               1],
    ])


# ---------------- measurement model ----------------
# z = [omega_L, omega_R, omega_gyro]  (left wheel, right wheel, gyro)

def h(mu, r, L):
    _, _, _, v, w = mu
    return np.array([
        (v - w * L / 2) / r,   # left wheel speed
        (v + w * L / 2) / r,   # right wheel speed
        w,                     # gyro
    ])


def H(r, L):
    """Jacobian of h. It is constant because h is linear in v and omega."""
    return np.array([
        [0, 0, 0, 1 / r, -L / (2 * r)],
        [0, 0, 0, 1 / r,  L / (2 * r)],
        [0, 0, 0, 0,      1],
    ])


# ---------------- filter ----------------

class EKF:
    def __init__(self, mu0, Sigma0, cfg):
        self.mu = np.array(mu0, dtype=float)
        self.Sigma = np.array(Sigma0, dtype=float)
        self.cfg = cfg
        # Q = diag(sigma_L^2, sigma_R^2, sigma_g^2)
        self.Q = np.diag([cfg.sigma_enc**2, cfg.sigma_enc**2, cfg.sigma_g**2])

    def R(self, dt):
        c = self.cfg
        return np.diag([c.s_xy**2, c.s_xy**2, c.s_th**2,
                        (c.sigma_a * dt)**2, (c.sigma_alpha * dt)**2])

    def predict(self, dt):
        Gt = G(self.mu, dt)                                   # G evaluated at mu_{t-1}
        self.mu = g(self.mu, dt)                              # mu_bar = g(mu_{t-1})
        self.Sigma = Gt @ self.Sigma @ Gt.T + self.R(dt)      # Sigma_bar = G Sigma G^T + R
        self.mu[2] = wrap(self.mu[2])

    def update(self, z):
        """Stacked update with z = [omega_L, omega_R, omega_gyro]."""
        return self.correct(z, h(self.mu, self.cfg.r, self.cfg.L), H(self.cfg.r, self.cfg.L), self.Q)

    def correct(self, z, z_pred, Ht, Qt):
        """Generic EKF correction step. Returns innovation y and its covariance S."""
        y = z - z_pred                                        # innovation
        S = Ht @ self.Sigma @ Ht.T + Qt                       # S = H Sigma_bar H^T + Q
        K = self.Sigma @ Ht.T @ np.linalg.inv(S)              # K = Sigma_bar H^T S^-1
        self.mu = self.mu + K @ y                             # mu = mu_bar + K y
        I_KH = np.eye(len(self.mu)) - K @ Ht
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + K @ Qt @ K.T  # Joseph form
        self.mu[2] = wrap(self.mu[2])
        return y, S
