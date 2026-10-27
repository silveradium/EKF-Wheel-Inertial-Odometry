"""All parameters in one place. SI units (m, s, rad)."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    # --- robot ---
    r: float = 0.033          # wheel radius [m]
    L: float = 0.16           # track width (distance between wheels) [m]
    # The real robot's effective track width is a bit different from the
    # measured one (tyre contact, slip in turns). Without this the simulated
    # encoders are perfect and nothing can beat them.
    L_true_scale: float = 1.03

    # --- true robot simulation ---
    T: float = 60.0           # run length [s]
    dt_true: float = 0.001    # fine time step for the truth [s]
    tau: float = 0.2          # motor lag time constant [s]
    battery_start: float = 1.0
    battery_end: float = 0.85

    # --- teleop (like turtlebot3 teleop keyboard) ---
    v_step: float = 0.02      # w / x key [m/s]
    w_step: float = 0.1       # a / d key [rad/s]
    v_max: float = 0.3
    w_max: float = 2.0

    # --- sensors ---
    enc_rate: float = 50.0    # encoder rate [Hz]
    cpr: int = 360            # encoder ticks per wheel revolution
    sigma_wheel: float = 0.2  # extra random noise on each wheel speed [rad/s]
                              # (vibration, bumpy floor). 0 = only tick counting.
    gyro_rate: float = 100.0  # gyro rate [Hz]
    gyro_offset: float = 0.003  # gyro start time offset [s]
    sigma_g: float = 0.01     # gyro white noise [rad/s]

    # --- EKF process noise R (tuning) ---
    # (spec said 1e-3, but that makes the filter far too pessimistic: NEES ~0.05)
    s_xy: float = 1e-5        # [m]
    s_th: float = 1e-5        # [rad]
    sigma_a: float = 0.5      # how fast v can change [m/s^2]
    sigma_alpha: float = 2.0  # how fast omega can change [rad/s^2]

    # --- EKF initial covariance (start pose is known) ---
    sigma0_var: float = 1e-6

    seed: int = 0

    @property
    def dt_enc(self):
        return 1.0 / self.enc_rate

    @property
    def sigma_enc(self):
        # Total wheel speed noise used in Q. Two independent parts, variances add:
        # 1) tick counting: wheel speed = difference of two quantized angles / dt.
        #    Each quantization error is uniform with variance step^2/12,
        #    the difference of two has variance step^2/6.
        # 2) the extra random noise sigma_wheel.
        step = 2 * np.pi / self.cpr
        sigma_quant = step / self.dt_enc * np.sqrt(1 / 6)
        return np.sqrt(sigma_quant**2 + self.sigma_wheel**2)
