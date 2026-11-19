import numpy as np


def press(t, key, n=1, gap=0.1):
    return [(t + i * gap, key) for i in range(n)]


DEFAULT_SCRIPT = (
    press(1.0, 'w', 5)        # 0.1 m/s
    + press(9.0, 'a', 5)      # left
    + press(15.0, 'd', 5)
    + press(19.0, 'd', 5)     # right
    + press(25.0, 'a', 5)
    + press(29.0, 's')
    + press(31.0, 'a', 10)    # spin
    + press(37.0, 's')
    + press(39.0, 'x', 5)     # reverse
    + press(43.0, 'a', 3)
    + press(48.0, 's')
    + press(50.0, 'w', 8)
    + press(57.0, 's')
)

STRAIGHT_SCRIPT = (
    press(1.0, 'w', 10)       # 0.2 m/s
    + press(51.0, 's')
)


def apply_key(key, v_cmd, w_cmd, cfg):
    if key == 'w':
        v_cmd += cfg.v_step
    elif key == 'x':
        v_cmd -= cfg.v_step
    elif key == 'a':
        w_cmd += cfg.w_step
    elif key == 'd':
        w_cmd -= cfg.w_step
    elif key == 's':
        v_cmd, w_cmd = 0.0, 0.0
    v_cmd = np.clip(v_cmd, -cfg.v_max, cfg.v_max)
    w_cmd = np.clip(w_cmd, -cfg.w_max, cfg.w_max)
    return v_cmd, w_cmd


def simulate_truth(cfg, script=DEFAULT_SCRIPT):
    dt = cfg.dt_true
    n = int(round(cfg.T / dt)) + 1
    t = np.arange(n) * dt
    out = {k: np.zeros(n) for k in ['x', 'y', 'theta', 'v', 'omega', 'phi_L', 'phi_R']}
    out['t'] = t

    events = sorted(script)
    e = 0
    v_cmd = w_cmd = 0.0
    x = y = th = v = w = phiL = phiR = 0.0

    for i in range(1, n):
        while e < len(events) and events[e][0] <= t[i - 1]:
            v_cmd, w_cmd = apply_key(events[e][1], v_cmd, w_cmd, cfg)
            e += 1

        gain = cfg.battery_start + (cfg.battery_end - cfg.battery_start) * t[i - 1] / cfg.T
        v += (gain * v_cmd - v) * dt / cfg.tau
        w += (gain * w_cmd - w) * dt / cfg.tau

        x += v * np.cos(th) * dt
        y += v * np.sin(th) * dt
        th += w * dt

        L_true = cfg.L * cfg.L_true_scale
        phiL += (v - w * L_true / 2) / cfg.r * dt
        phiR += (v + w * L_true / 2) / cfg.r * dt

        out['x'][i], out['y'][i], out['theta'][i] = x, y, th
        out['v'][i], out['omega'][i] = v, w
        out['phi_L'][i], out['phi_R'][i] = phiL, phiR
    return out


def make_sensors(truth, cfg, rng):
    dt = cfg.dt_true
    step = 2 * np.pi / cfg.cpr

    # encoders
    t_enc = np.arange(0, cfg.T + 1e-9, cfg.dt_enc)
    idx = np.round(t_enc / dt).astype(int)
    off_L, off_R = rng.uniform(0, step, size=2)   # random start angle
    ticks_L = np.floor((truth['phi_L'][idx] + off_L) / step)
    ticks_R = np.floor((truth['phi_R'][idx] + off_R) / step)
    wL = np.diff(ticks_L) * step / cfg.dt_enc
    wR = np.diff(ticks_R) * step / cfg.dt_enc
    t_enc, idx = t_enc[1:], idx[1:]
    wL = wL + rng.normal(0, cfg.sigma_wheel, size=len(wL))
    wR = wR + rng.normal(0, cfg.sigma_wheel, size=len(wR))

    # gyro
    t_gyro = np.arange(cfg.gyro_offset, cfg.T + 1e-9, 1.0 / cfg.gyro_rate)
    gidx = np.round(t_gyro / dt).astype(int)
    gyro = truth['omega'][gidx] + rng.normal(0, cfg.sigma_g, size=len(t_gyro))

    true = np.column_stack([truth[k][idx] for k in ['x', 'y', 'theta', 'v', 'omega']])
    return dict(t_enc=t_enc, wL=wL, wR=wR, t_gyro=t_gyro, gyro=gyro, true=true)


def sync_gyro(t_enc, t_gyro, gyro):
    # interpolate gyro at encoder times, drop encoder samples outside gyro range
    keep = (t_enc >= t_gyro[0]) & (t_enc <= t_gyro[-1])
    return keep, np.interp(t_enc[keep], t_gyro, gyro)
