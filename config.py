from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    # robot
    r: float = 0.033
    L: float = 0.16
    L_true_scale: float = 1.03   # real track width is 3% off

    # truth sim
    T: float = 60.0
    dt_true: float = 0.001
    tau: float = 0.2             # motor lag
    battery_start: float = 1.0
    battery_end: float = 0.85

    # teleop
    v_step: float = 0.02
    w_step: float = 0.1
    v_max: float = 0.3
    w_max: float = 2.0

    # sensors
    enc_rate: float = 50.0
    cpr: int = 360
    sigma_wheel: float = 0.2     # extra wheel speed noise (rad/s)
    gyro_rate: float = 100.0
    gyro_offset: float = 0.003
    sigma_g: float = 0.01

    # process noise R
    s_xy: float = 1e-5           # 1e-3 was too pessimistic
    s_th: float = 1e-5
    sigma_a: float = 0.5
    sigma_alpha: float = 2.0

    sigma0_var: float = 1e-6
    seed: int = 0

    @property
    def dt_enc(self):
        return 1.0 / self.enc_rate

    @property
    def sigma_enc(self):
        # quantization: var = step^2/6 for the difference of two readings
        step = 2 * np.pi / self.cpr
        sigma_quant = step / self.dt_enc * np.sqrt(1 / 6)
        return np.sqrt(sigma_quant**2 + self.sigma_wheel**2)
