# NCLT 2012-01-08. gt and ms25 are z-down, so y, yaw and gyro are flipped. kvh is already ccw.
import os
from dataclasses import replace
import numpy as np

from config import Config
from ekf import wrap
from run import run_all_methods, final_errors, nees_pose
import plots
import matplotlib.pyplot as plt

SESSION = '2012-01-08'
DIR = 'NCLT'
OUT = 'nclt_outputs'
DT_ENC = 0.1
CALIB = (0, 2700)     # first half used for calibration
TESTS = [2800, 3200, 3600, 4000, 4400, 4800, 5200]
DURATION = 180


def load():
    def csv(name):
        return np.loadtxt(os.path.join(DIR, name), delimiter=',')
    w = csv(f'{SESSION}/wheels.csv')
    m = csv(f'{SESSION}/ms25.csv')
    k = csv(f'{SESSION}/kvh.csv')
    gt = csv(f'groundtruth_{SESSION}.csv')
    gt = gt[~np.isnan(gt).any(axis=1)]
    t0 = w[0, 0]

    def sec(a):
        return (a[:, 0] - t0) / 1e6
    return dict(
        t_wheel=sec(w), vL=w[:, 1], vR=w[:, 2],
        t_gyro=sec(m), gyro=-m[:, 9],
        t_kvh=sec(k), kvh=np.unwrap(k[:, 1]),
        t_gt=sec(gt), x=gt[:, 1], y=-gt[:, 2], yaw=-np.unwrap(gt[:, 6]),
    )


def resample(d, t, cal=None):
    wL = np.interp(t, d['t_wheel'], d['vL'])
    wR = np.interp(t, d['t_wheel'], d['vR'])
    gyro = np.interp(t, d['t_gyro'], d['gyro'])
    # kvh is heading, so differentiate it
    h = DT_ENC / 2
    kvh = (np.interp(t + h, d['t_kvh'], d['kvh']) - np.interp(t - h, d['t_kvh'], d['kvh'])) / DT_ENC

    x = np.interp(t, d['t_gt'], d['x'])
    y = np.interp(t, d['t_gt'], d['y'])
    yaw = np.interp(t, d['t_gt'], d['yaw'])
    v = np.gradient(x, t) * np.cos(yaw) + np.gradient(y, t) * np.sin(yaw)
    w = np.gradient(yaw, t)
    true = np.column_stack([x, y, wrap(yaw), v, w])

    if cal is not None:
        wL, wR = cal['scale_L'] * wL, cal['scale_R'] * wR
        gyro = gyro - cal['bias']
    return wL, wR, gyro, kvh, true


def calibrate(d):
    t = np.arange(CALIB[0], CALIB[1], DT_ENC)
    wL, wR, gyro, kvh, true = resample(d, t)
    v, w = true[:, 3], true[:, 4]
    ok = np.abs(w) < 2

    # v = scale * (wL + wR) / 2
    s = (wL + wR) / 2
    scale = np.sum(v[ok] * s[ok]) / np.sum(s[ok]**2)
    # w = p*wR - q*wL, p = sR/L, q = sL/L
    A = np.column_stack([wR, -wL])
    p, q = np.linalg.lstsq(A[ok], w[ok], rcond=None)[0]
    sL = 2 * scale / (1 + p / q)
    sR = sL * p / q
    L = sR / p
    bias = np.mean(gyro[ok] - w[ok])

    # noise from residuals
    resid_L = sL * wL - (v - w * L / 2)
    resid_R = sR * wR - (v + w * L / 2)
    sigma_wheel = np.std(np.r_[resid_L[ok], resid_R[ok]])
    sigma_g = np.std(gyro[ok] - bias - w[ok])
    sigma_kvh = np.std(kvh[ok] - w[ok])
    return dict(scale_L=sL, scale_R=sR, L=L, bias=bias, sigma_wheel=sigma_wheel,
                sigma_g=sigma_g, sigma_kvh=sigma_kvh)


def nclt_config(cal, gyro_noise):
    return replace(
        Config(),
        r=1.0,                # wheels already in m/s
        L=cal['L'],
        enc_rate=1 / DT_ENC,
        cpr=10**9,            # no quantization term
        sigma_wheel=cal['sigma_wheel'],
        sigma_g=gyro_noise,
    )


def run_segment(d, cal, t_start, gyro_name='ms25', duration=DURATION):
    t = np.arange(t_start, t_start + duration, DT_ENC)
    wL, wR, gyro, kvh, true = resample(d, t, cal)
    if gyro_name == 'ms25':
        cfg = nclt_config(cal, cal['sigma_g'])
        return run_all_methods(t - t[0], wL, wR, gyro, true, cfg)
    cfg = nclt_config(cal, cal['sigma_kvh'])
    return run_all_methods(t - t[0], wL, wR, kvh, true, cfg)


def plot_overview(d):
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#7a5af5']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(d['x'], d['y'], color='#bbbbbb', lw=1, label='Whole session (ground truth)')
    for i, t0 in enumerate(TESTS):
        k = (d['t_gt'] >= t0) & (d['t_gt'] < t0 + DURATION)
        ax.plot(d['x'][k], d['y'][k], color=colors[i], lw=2.5, label=f'Test segment {i + 1} ({t0} s)')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_aspect('equal')
    ax.set_title('NCLT %s: whole drive and the test segments' % SESSION)
    ax.legend(fontsize=8, loc='best')
    plots.save(fig, '0_overview.png')


def plot_full_run(d, cal):
    # note: first half is also the calibration data
    t_start, t_end = d['t_gt'][0] + 1, d['t_gt'][-1] - 1
    res = run_segment(d, cal, t_start, 'ms25', duration=t_end - t_start)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(res['true'][:, 0], res['true'][:, 1], color=plots.TRUTH, lw=2, label='Truth')
    ax.plot(res['enc'][:, 0], res['enc'][:, 1], color=plots.ENC_C, lw=1, label='Encoder only')
    ax.plot(res['enc_gyro'][:, 0], res['enc_gyro'][:, 1], color=plots.ENCG_C, lw=1, label='Encoder + gyro')
    ax.plot(res['mu'][:, 0], res['mu'][:, 1], color=plots.EKF_C, lw=1.5, ls='--', label='EKF')
    ax.plot(*res['true'][0, :2], 'o', color=plots.TRUTH, ms=8)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_aspect('equal')
    dist = np.sum(np.hypot(*np.diff(res['true'][:, :2], axis=0).T))
    ax.set_title('Whole session, %.0f min, %.1f km (start = dot)' % ((t_end - t_start) / 60, dist / 1000))
    ax.legend()
    plots.save(fig, '5_full_session.png')
    fe = final_errors(res)
    return ['\nWhole session (%.0f min, %.1f km), MS25 gyro, final errors:' % ((t_end - t_start) / 60, dist / 1000)] + [
        f'{name:<16}{fe[m][0]:>14.1f} m{np.degrees(fe[m][1]):>12.1f} deg'
        for m, name in [('mu', 'EKF'), ('enc', 'Encoder only'), ('enc_gyro', 'Encoder + gyro')]]


def main():
    print('loading NCLT data ...')
    d = load()
    cal = calibrate(d)
    print('calibration (from t = %d..%d s):' % CALIB)
    for k, v in cal.items():
        print(f'  {k:<12} {v:.5f}')

    plots.OUT = OUT
    plot_overview(d)
    names = {'mu': 'EKF', 'enc': 'Encoder only', 'enc_gyro': 'Encoder + gyro'}
    lines = []
    for gyro_name in ['ms25', 'kvh']:
        results = [run_segment(d, cal, t0, gyro_name) for t0 in TESTS]
        fe = [final_errors(r) for r in results]
        lines.append(f'\nGyro = {gyro_name}. Final errors after {DURATION} s, '
                     f'mean over {len(TESTS)} segments')
        lines.append(f'{"method":<16}{"position [m]":>14}{"heading [deg]":>15}')
        for m, name in names.items():
            p = np.mean([f[m][0] for f in fe])
            hd = np.degrees(np.mean([f[m][1] for f in fe]))
            lines.append(f'{name:<16}{p:>14.2f}{hd:>15.2f}')
        lines.append('mean NEES (expected 3): %.1f   mean NIS (expected 3): %.1f' % (
            np.mean([np.mean(nees_pose(r)[1:]) for r in results]),
            np.mean([np.nanmean(r['nis']) for r in results])))
        lines.append('distance driven per segment [m]: ' + ', '.join(
            '%.0f' % np.sum(np.hypot(*np.diff(r['true'][:, :2], axis=0).T)) for r in results))

        if gyro_name == 'ms25':
            plots.plot_trajectory(results[0], every=300)
            plots.plot_errors(results[0])
            plots.plot_nis(results[0])

    lines += plot_full_run(d, cal)
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(OUT, 'summary.txt'), 'w') as f:
        f.write('calibration:\n' + '\n'.join(f'  {k} = {v:.5f}' for k, v in cal.items()) + '\n' + text + '\n')


if __name__ == '__main__':
    main()
