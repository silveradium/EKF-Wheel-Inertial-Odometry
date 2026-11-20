# NCLT 2012-01-08. gt and ms25 are z-down, so y, yaw and gyro are flipped. kvh is already ccw.
import os
from dataclasses import replace
import numpy as np

from config import Config
from ekf import wrap
from run import run_all_methods, run_ekf, final_errors, nees_pose
import plots
import matplotlib.pyplot as plt

SESSION = '2012-01-08'
DIR = 'NCLT'
OUT = 'nclt_outputs'
DT_ENC = 0.1
CALIB = (0, 2700)     # first half used for calibration
TESTS = [2800, 3200, 3600, 4000, 4400, 4800, 5200]
DURATION = 180
GPS_FILE = 'gps_rtk'
GPS_PERIOD = 10.0     # use one gps fix every 10 s
GPS_GATE = 13.8       # chi2, 2 dof, 99.9%
BIAS_PRIOR = 0.002   # initial bias std, rad/s (see bias_prior_study)
PLOT_SEG = 4          # segment 5 has the most gps fixes
NAMES = {'enc': 'Encoder only', 'enc_gyro': 'Encoder + gyro', 'mu': 'EKF',
         'mu_cal': 'EKF, bias from gt', 'mu_b': 'EKF + bias', 'mu_gps_nb': 'EKF + GPS',
         'mu_gps': 'EKF + bias + GPS'}


def csv(name):
    return np.loadtxt(os.path.join(DIR, name), delimiter=',')


def load():
    w = csv(f'{SESSION}/wheels.csv')
    m = csv(f'{SESSION}/ms25.csv')
    k = csv(f'{SESSION}/kvh.csv')
    gt = csv(f'groundtruth_{SESSION}.csv')
    gt = gt[~np.isnan(gt).any(axis=1)]
    t0 = w[0, 0]

    def sec(a):
        return (a[:, 0] - t0) / 1e6
    d = dict(
        t0=t0,
        t_wheel=sec(w), vL=w[:, 1], vR=w[:, 2],
        t_gyro=sec(m), gyro=-m[:, 9],
        t_kvh=sec(k), kvh=np.unwrap(k[:, 1]),
        t_gt=sec(gt), x=gt[:, 1], y=-gt[:, 2], yaw=-np.unwrap(gt[:, 6]),
    )
    d.update(load_gps(GPS_FILE, t0))
    return d


def load_gps(name, t0):
    gps = csv(f'{SESSION}/{name}.csv')
    gps = gps[(gps[:, 1] == 3) & np.isfinite(gps[:, 3])]      # 3D fixes only
    # lat/lon (rad) -> metres, x north, y west (z-up like the flipped gt)
    R_earth = 6378137.0
    lat0, lon0 = gps[0, 3], gps[0, 4]
    return dict(t_gps=(gps[:, 0] - t0) / 1e6,
                gps_x=R_earth * (gps[:, 3] - lat0),
                gps_y=-R_earth * np.cos(lat0) * (gps[:, 4] - lon0))


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
    # only used for the 'bias from gt' comparison, the filter does not get it
    bias = np.mean(gyro[ok] - w[ok])
    bias_kvh = np.mean(kvh[ok] - w[ok])

    # noise from residuals
    resid_L = sL * wL - (v - w * L / 2)
    resid_R = sR * wR - (v + w * L / 2)
    sigma_wheel = np.std(np.r_[resid_L[ok], resid_R[ok]])
    sigma_g = np.std(gyro[ok] - bias - w[ok])
    sigma_kvh = np.std(kvh[ok] - bias_kvh - w[ok])

    # gps: the origin is different from gt, so fit an offset
    tg = d['t_gps']
    c = (tg > max(CALIB[0], d['t_gt'][0])) & (tg < CALIB[1])
    ex = np.interp(tg[c], d['t_gt'], d['x']) - d['gps_x'][c]
    ey = np.interp(tg[c], d['t_gt'], d['y']) - d['gps_y'][c]
    gps_dx, gps_dy = np.median(ex), np.median(ey)
    # robust std (MAD) so the big outliers don't blow it up
    res_gps = np.r_[ex - gps_dx, ey - gps_dy]
    sigma_gps = 1.4826 * np.median(np.abs(res_gps))
    return dict(scale_L=sL, scale_R=sR, L=L, bias=bias, bias_kvh=bias_kvh, sigma_wheel=sigma_wheel,
                sigma_g=sigma_g, sigma_kvh=sigma_kvh,
                gps_dx=gps_dx, gps_dy=gps_dy, sigma_gps=sigma_gps)


def gps_on_timeline(d, cal, t):
    # first fix after every GPS_PERIOD seconds, put on the nearest step
    gps = np.full((len(t), 2), np.nan)
    for ts in np.arange(t[0] + GPS_PERIOD, t[-1], GPS_PERIOD):
        j = np.searchsorted(d['t_gps'], ts)
        if j == len(d['t_gps']) or d['t_gps'][j] - ts > 1.0:
            continue                                  # no fix around then
        k = int(round((d['t_gps'][j] - t[0]) / DT_ENC))
        if k < len(t):
            gps[k] = [d['gps_x'][j] + cal['gps_dx'], d['gps_y'][j] + cal['gps_dy']]
    return gps


def nclt_config(cal, gyro_noise):
    return replace(
        Config(),
        r=1.0,                # wheels already in m/s
        L=cal['L'],
        enc_rate=1 / DT_ENC,
        cpr=10**9,            # no quantization term
        sigma_wheel=cal['sigma_wheel'],
        sigma_g=gyro_noise,
        sigma_b0=BIAS_PRIOR,
    )


def run_segment(d, cal, t_start, gyro_name='ms25', duration=DURATION):
    t = np.arange(t_start, t_start + duration, DT_ENC)
    wL, wR, gyro, kvh, true = resample(d, t, cal)
    if gyro_name == 'kvh':
        gyro = kvh
    cfg = nclt_config(cal, cal['sigma_g'] if gyro_name == 'ms25' else cal['sigma_kvh'])
    res = run_all_methods(t - t[0], wL, wR, gyro, true, cfg)

    z = res['z'].copy()
    z[:, 2] -= cal['bias'] if gyro_name == 'ms25' else cal['bias_kvh']
    res['mu_cal'], res['Sigma_cal'] = run_ekf(t - t[0], z, true[0], cfg)[:2]

    gps = gps_on_timeline(d, cal, t)
    res['mu_gps_nb'] = run_ekf(t - t[0], res['z'], true[0], cfg, gps, cal['sigma_gps'], GPS_GATE)[0]
    mu, Sigma, _, _, used = run_ekf(t - t[0], res['z'], true[0], cfg, gps, cal['sigma_gps'], GPS_GATE, bias=True)
    res.update(mu_gps=mu, Sigma_gps=Sigma, gps=gps, gps_used=used)
    return res


def plot_sigma(res):
    # position uncertainty over time, the drops are the gps updates
    def pos_sd(S):
        return np.sqrt(S[:, 0, 0] + S[:, 1, 1])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(res['t'], pos_sd(res['Sigma_b']), color=plots.BIAS_C, lw=1.5, label='EKF + bias')
    ax.plot(res['t'], pos_sd(res['Sigma_gps']), color=plots.GPS_C, lw=1.5, label='EKF + bias + GPS')
    for k in np.flatnonzero(res['gps_used']):
        ax.axvline(res['t'][k], color=plots.GPS_C, lw=0.5, alpha=0.4)
    ax.plot([], [], color=plots.GPS_C, lw=0.5, alpha=0.4, label='GPS update')
    ax.set_xlabel('time [s]'); ax.set_ylabel('position sigma [m]')
    ax.set_yscale('log')
    ax.set_title('Position uncertainty: grows between fixes, drops at each fix')
    ax.legend()
    plots.save(fig, '7_gps_sigma.png')


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
    ax.plot(res['mu_b'][:, 0], res['mu_b'][:, 1], color=plots.BIAS_C, lw=1, label='EKF + bias')
    ax.plot(res['mu_gps'][:, 0], res['mu_gps'][:, 1], color=plots.GPS_C, lw=1, label='EKF + bias + GPS')
    ax.plot(*res['true'][0, :2], 'o', color=plots.TRUTH, ms=8)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_aspect('equal')
    dist = np.sum(np.hypot(*np.diff(res['true'][:, :2], axis=0).T))
    ax.set_title('Whole session, %.0f min, %.1f km (start = dot)' % ((t_end - t_start) / 60, dist / 1000))
    ax.legend()
    plots.save(fig, '5_full_session.png')
    fe = final_errors(res)
    return ['\nWhole session (%.0f min, %.1f km), MS25 gyro, final errors:' % ((t_end - t_start) / 60, dist / 1000)] + [
        f'{name:<16}{fe[m][0]:>14.1f} m{np.degrees(fe[m][1]):>12.1f} deg'
        for m, name in NAMES.items()]


def gps_comparison(d):
    lines = ['\nGPS source comparison, ms25 gyro, EKF + bias + GPS, final position error per segment [m]']
    no_gps = None
    for name in ['gps_rtk', 'gps']:
        dd = dict(d, **load_gps(name, d['t0']))
        cal = calibrate(dd)
        results = [run_segment(dd, cal, t0) for t0 in TESTS]
        if no_gps is None:
            no_gps = [final_errors(r)['mu_b'][0] for r in results]
            lines.append(f'{"no gps":<10}' + ''.join(f'{e:>7.2f}' for e in no_gps) + f'   mean {np.mean(no_gps):.2f}')
        e = [final_errors(r)['mu_gps'][0] for r in results]
        lines.append(f'{name:<10}' + ''.join(f'{x:>7.2f}' for x in e) + f'   mean {np.mean(e):.2f}'
                     f'   sigma_gps {cal["sigma_gps"]:.2f} m, worse than no gps on {sum(a > b for a, b in zip(e, no_gps))}/7')
    return lines


def bias_prior_study(d, cal):
    lines = ['\nBias prior study, ms25 gyro: mean final position error [m] and NEES over the 7 segments']
    lines.append(f'{"sigma_b0":<10}{"EKF + bias":>12}{"NEES":>8}{"+ GPS":>10}{"worst + GPS":>13}')
    for b0 in [0.05, 0.01, 0.005, 0.002, 0.001]:
        e, ne, eg = [], [], []
        for t0 in TESTS:
            t = np.arange(t0, t0 + DURATION, DT_ENC)
            wL, wR, gyro, _, true = resample(d, t, cal)
            z = np.column_stack([wL, wR, gyro])
            cfg = replace(nclt_config(cal, cal['sigma_g']), sigma_b0=b0)
            mu, S = run_ekf(t - t[0], z, true[0], cfg, bias=True)[:2]
            r = dict(true=true, mu_b=mu, Sigma_b=S)
            e.append(final_errors(r)['mu_b'][0])
            ne.append(np.mean(nees_pose(r, 'mu_b')[1:]))
            mu = run_ekf(t - t[0], z, true[0], cfg, gps_on_timeline(d, cal, t), cal['sigma_gps'], GPS_GATE, bias=True)[0]
            eg.append(np.hypot(*(true[-1, :2] - mu[-1, :2])))
        lines.append(f'{b0:<10}{np.mean(e):>12.2f}{np.mean(ne):>8.0f}{np.mean(eg):>10.2f}{max(eg):>13.1f}')
    return lines


def main():
    print('loading NCLT data ...')
    d = load()
    cal = calibrate(d)
    print('calibration (from t = %d..%d s):' % CALIB)
    for k, v in cal.items():
        print(f'  {k:<12} {v:.5f}')

    plots.OUT = OUT
    plot_overview(d)
    lines = [f'GPS: {GPS_FILE}, one fix every {GPS_PERIOD:.0f} s']
    for gyro_name in ['ms25', 'kvh']:
        results = [run_segment(d, cal, t0, gyro_name) for t0 in TESTS]
        fe = [final_errors(r) for r in results]
        lines.append(f'\nGyro = {gyro_name}. Final errors after {DURATION} s, '
                     f'mean over {len(TESTS)} segments')
        lines.append(f'{"method":<16}{"position [m]":>14}{"heading [deg]":>15}')
        for m, name in NAMES.items():
            p = np.mean([f[m][0] for f in fe])
            hd = np.degrees(np.mean([f[m][1] for f in fe]))
            lines.append(f'{name:<16}{p:>14.2f}{hd:>15.2f}')
        lines.append('mean NEES (expected 3): EKF %.1f, EKF + bias %.1f, EKF + bias + GPS %.1f' % tuple(
            np.mean([np.mean(nees_pose(r, k)[1:]) for r in results]) for k in ['mu', 'mu_b', 'mu_gps']))
        lines.append('mean NIS (expected 3): EKF %.1f, EKF + bias %.1f' % (
            np.mean([np.nanmean(r['nis']) for r in results]),
            np.mean([np.nanmean(r['nis_b']) for r in results])))
        lines.append('final bias estimate per segment [rad/s]: ' + ', '.join(
            '%.4f' % r['mu_b'][-1, 5] for r in results) + '  (from gt: %.4f)' % (
            cal['bias'] if gyro_name == 'ms25' else cal['bias_kvh']))
        lines.append('gps fixes used / rejected: %d / %d' % (
            sum(r['gps_used'].sum() for r in results),
            sum((~np.isnan(r['gps'][:, 0]) & ~r['gps_used']).sum() for r in results)))
        lines.append('distance driven per segment [m]: ' + ', '.join(
            '%.0f' % np.sum(np.hypot(*np.diff(r['true'][:, :2], axis=0).T)) for r in results))

        if gyro_name == 'ms25':
            r = results[PLOT_SEG]
            plots.plot_trajectory(r, every=300)
            plots.plot_errors(r)
            plots.plot_nis(r)
            plots.plot_errors(r, key='mu_b', name='3_errors_bias.png')
            plots.plot_bias(r, cal['bias'], name='9_bias.png')
            plots.plot_errors(r, key='mu_gps', name='6_gps_errors.png')
            plot_sigma(r)
            plots.make_animation(r, every=5, key='mu_gps', scale=1, view=15,
                                 size=1.5, name='8_gps_animation.gif')

    lines += bias_prior_study(d, cal)
    lines += gps_comparison(d)
    lines += plot_full_run(d, cal)
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(OUT, 'summary.txt'), 'w') as f:
        f.write('calibration:\n' + '\n'.join(f'  {k} = {v:.5f}' for k, v in cal.items()) + '\n' + text + '\n')


if __name__ == '__main__':
    main()
