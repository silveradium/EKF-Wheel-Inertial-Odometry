import os
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from sim import simulate_truth, DEFAULT_SCRIPT
from run import run_once, monte_carlo, pose_error, nees_pose, final_errors, chi2_bounds

OUT = 'outputs'
TRUTH, EKF_C, ENC_C, ENCG_C, GPS_C, BIAS_C = '#333333', '#2a78d6', '#eb6834', '#1baf7a', '#7a5af5', '#d6336c'
LABEL = {'mu': 'EKF', 'mu_b': 'EKF + bias', 'mu_gps': 'EKF + bias + GPS'}
COLOR = {'mu': EKF_C, 'mu_b': BIAS_C, 'mu_gps': GPS_C}
plt.rcParams.update({'figure.dpi': 120, 'axes.grid': True, 'grid.alpha': 0.3,
                     'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print('saved', name)


def ellipse(mean, cov, n_sigma=2):
    vals, vecs = np.linalg.eigh(cov)
    a = np.linspace(0, 2 * np.pi, 50)
    circle = np.array([np.cos(a), np.sin(a)])
    return (vecs @ (n_sigma * np.sqrt(vals)[:, None] * circle)).T + mean


def plot_trajectory(res, every=150):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(res['true'][:, 0], res['true'][:, 1], color=TRUTH, lw=2, label='Truth')
    ax.plot(res['enc'][:, 0], res['enc'][:, 1], color=ENC_C, lw=1.5, label='Encoder only')
    ax.plot(res['mu'][:, 0], res['mu'][:, 1], color=EKF_C, lw=1.5, ls='--', label='EKF')
    for k in range(every, len(res['t']), every):
        e = ellipse(res['mu'][k, :2], res['Sigma'][k, :2, :2])
        ax.plot(e[:, 0], e[:, 1], color=EKF_C, lw=0.8)
    ax.plot([], [], color=EKF_C, lw=0.8, label='EKF 2-sigma ellipse')
    if 'mu_b' in res:
        ax.plot(res['mu_b'][:, 0], res['mu_b'][:, 1], color=BIAS_C, lw=1.5, label='EKF + bias')
    if 'mu_gps' in res:
        gps, used = res['gps'], res['gps_used']
        rejected = ~np.isnan(gps[:, 0]) & ~used
        ax.plot(gps[used, 0], gps[used, 1], 'o', mfc='none', color=GPS_C, ms=5, label='GPS fix')
        if rejected.any():
            ax.plot(gps[rejected, 0], gps[rejected, 1], 'x', color=GPS_C, ms=6, label='GPS fix rejected')
        ax.plot(res['mu_gps'][:, 0], res['mu_gps'][:, 1], color=GPS_C, lw=1.5, label=LABEL['mu_gps'])
        for k in range(every, len(res['t']), every):
            e = ellipse(res['mu_gps'][k, :2], res['Sigma_gps'][k, :2, :2])
            ax.plot(e[:, 0], e[:, 1], color=GPS_C, lw=0.8)
    ax.plot(*res['true'][0, :2], 'o', color=TRUTH, ms=8)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_aspect('equal')
    ax.set_title('Trajectory (start = dot)')
    ax.legend()
    save(fig, '1_trajectory.png')


def plot_errors(res, key='mu', name='2_errors.png'):
    e = pose_error(res['true'], res[key])
    Sig = res['Sigma' + key[2:]]
    sd = np.sqrt(np.array([Sig[:, i, i] for i in range(3)])).T
    color, label = COLOR[key], LABEL[key]
    fig, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for i, (ax, lab) in enumerate(zip(axs, ['x error [m]', 'y error [m]', 'theta error [rad]'])):
        ax.plot(res['t'], e[:, i], color=color, lw=1, label=label + ' error')
        ax.plot(res['t'], 3 * sd[:, i], color=TRUTH, lw=1, ls='--', label='+/- 3 sigma')
        ax.plot(res['t'], -3 * sd[:, i], color=TRUTH, lw=1, ls='--')
        ax.set_ylabel(lab)
    axs[0].legend(loc='upper left')
    axs[-1].set_xlabel('time [s]')
    axs[0].set_title(label + ' error and its 3-sigma bound')
    save(fig, name)


def plot_nees(runs):
    lo, hi = chi2_bounds(3, len(runs))
    t = runs[0]['t']
    fig, ax = plt.subplots(figsize=(8, 4))
    for key in ['mu', 'mu_b']:
        nees = np.mean([nees_pose(r, key) for r in runs], axis=0)
        ax.plot(t[1:], nees[1:], color=COLOR[key], lw=1, label=f'{LABEL[key]}, average NEES ({len(runs)} runs)')
    ax.axhline(3, color=TRUTH, lw=1, label='Expected value (3)')
    ax.axhline(lo, color=TRUTH, lw=1, ls='--', label='95% bounds')
    ax.axhline(hi, color=TRUTH, lw=1, ls='--')
    ax.set_yscale('log')
    ax.set_xlabel('time [s]'); ax.set_ylabel('NEES (x, y, theta)')
    ax.set_title('Is the EKF uncertainty honest? (NEES)')
    ax.legend()
    save(fig, '3_nees.png')


def plot_nis(res):
    lo, hi = chi2_bounds(3, 1)
    inside = np.mean((res['nis'][1:] > lo) & (res['nis'][1:] < hi))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(res['t'], res['nis'], '.', color=EKF_C, ms=2, label='NIS')
    ax.axhline(lo, color=TRUTH, lw=1, ls='--', label='95% bounds')
    ax.axhline(hi, color=TRUTH, lw=1, ls='--')
    ax.set_yscale('log')
    ax.set_xlabel('time [s]'); ax.set_ylabel('NIS')
    ax.set_title(f'NIS (single run): {100 * inside:.0f}% inside the bounds, '
                 f'mean = {np.nanmean(res["nis"]):.2f} (expected 3)')
    ax.legend()
    save(fig, '4_nis.png')


def plot_sensitivity(cfg, truth, seeds):
    def mean_final(c, tr):
        fe = [final_errors(run_once(c, s, tr)) for s in seeds]
        return [np.mean([f[m][0] for f in fe]) for m in ['mu', 'enc', 'enc_gyro', 'mu_b']]

    sig = [0.001, 0.003, 0.01, 0.03, 0.1]
    cprs = [90, 180, 360, 720, 1440]
    a = np.array([mean_final(replace(cfg, sigma_g=s), truth) for s in sig])
    b = np.array([mean_final(replace(cfg, cpr=c), truth) for c in cprs])

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, xs, ys, xl in [(axs[0], sig, a, 'gyro noise sigma_g [rad/s]'),
                           (axs[1], cprs, b, 'encoder CPR [ticks/rev]')]:
        ax.plot(xs, ys[:, 0], 'o-', color=EKF_C, label='EKF')
        ax.plot(xs, ys[:, 3], 'd-', color=BIAS_C, label='EKF + bias')
        ax.plot(xs, ys[:, 2], 's--', color=ENCG_C, label='Encoder + gyro')
        ax.plot(xs, ys[:, 1], '^-', color=ENC_C, label='Encoder only')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(xl); ax.set_ylabel('final position error [m]')
    axs[0].legend()
    axs[0].set_title('Effect of gyro noise'); axs[1].set_title('Effect of encoder resolution')
    save(fig, '5_sensitivity.png')


def plot_tuning(cfg, truth, seeds, key='mu'):
    scales = [0.01, 0.1, 1, 10, 100]
    err, nees = [], []
    for s in scales:
        c = replace(cfg, sigma_a=cfg.sigma_a * s, sigma_alpha=cfg.sigma_alpha * s)
        runs = [run_once(c, sd, truth) for sd in seeds]
        err.append(np.mean([final_errors(r)[key][0] for r in runs]))
        nees.append(np.mean([np.mean(nees_pose(r, key)[1:]) for r in runs]))
    lo, hi = chi2_bounds(3, len(seeds))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(scales, err, 'o-', color=COLOR[key])
    axs[0].set_ylabel('final position error [m]')
    axs[0].set_title('Accuracy')
    axs[1].plot(scales, nees, 'o-', color=COLOR[key], label=f'Mean NEES ({LABEL[key]})')
    axs[1].axhline(3, color=TRUTH, lw=1, label='Expected value (3)')
    axs[1].axhspan(lo, hi, color=TRUTH, alpha=0.1, label='95% bounds')
    axs[1].set_yscale('log'); axs[1].set_ylabel('mean NEES')
    axs[1].set_title('Honesty of the uncertainty')
    axs[1].legend()
    for ax in axs:
        ax.set_xscale('log')
        ax.set_xlabel('scale on sigma_a and sigma_alpha (1 = default)')
    save(fig, '6_tuning.png')


def make_animation(res, every=10, key='mu', scale=10, view=None, size=0.06, name='8_animation.gif'):
    # view: half width of a window that follows the robot (None = whole path)
    fig, ax = plt.subplots(figsize=(6, 6))
    tr, mu, Sig = res['true'], res[key], res['Sigma' + key[2:]]
    color, label = COLOR[key], LABEL[key]
    if view is None:
        pad = 0.2
        ax.set_xlim(tr[:, 0].min() - pad, tr[:, 0].max() + pad)
        ax.set_ylim(tr[:, 1].min() - pad, tr[:, 1].max() + pad)
    ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    true_line, = ax.plot([], [], color=TRUTH, lw=2, label='Truth')
    ekf_line, = ax.plot([], [], color=color, lw=1.5, ls='--', label=label)
    robot, = ax.fill([0], [0], color=color, alpha=0.8)
    ell_label = f'{label} 2-sigma ellipse' + (f' (x{scale})' if scale != 1 else '')
    ell, = ax.plot([], [], color=color, lw=1, label=ell_label)
    show_gps = 'gps' in res and key == 'mu_gps'
    if show_gps:
        gps, used = res['gps'], res['gps_used']
        gps_dots, = ax.plot([], [], 'o', mfc='none', color=GPS_C, ms=5, label='GPS fix')
    title = ax.set_title('')
    ax.legend(loc='upper right')
    frames = range(0, len(res['t']), every)

    def draw(k):
        true_line.set_data(tr[:k + 1, 0], tr[:k + 1, 1])
        ekf_line.set_data(mu[:k + 1, 0], mu[:k + 1, 1])
        x, y, th = mu[k, :3]
        s = size
        pts = np.array([[s, 0], [-s / 2, s / 2], [-s / 2, -s / 2]])
        Rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        robot.set_xy(pts @ Rot.T + [x, y])
        e = ellipse(mu[k, :2], scale**2 * Sig[k, :2, :2])
        ell.set_data(e[:, 0], e[:, 1])
        if show_gps:
            seen = used[:k + 1]
            gps_dots.set_data(gps[:k + 1][seen, 0], gps[:k + 1][seen, 1])
        if view is not None:
            ax.set_xlim(x - view, x + view)
            ax.set_ylim(y - view, y + view)
        title.set_text(f't = {res["t"][k]:.1f} s')
        return true_line, ekf_line, robot, ell, title

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    os.makedirs(OUT, exist_ok=True)
    anim.save(os.path.join(OUT, name), writer=PillowWriter(fps=15))
    plt.close(fig)
    print('saved', name)


def plot_bias(res, true_bias=None, name='7_bias.png'):
    b, sd = res['mu_b'][:, 5], np.sqrt(res['Sigma_b'][:, 5, 5])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(res['t'], b, color=BIAS_C, lw=1.5, label='Estimated bias')
    ax.fill_between(res['t'], b - 2 * sd, b + 2 * sd, color=BIAS_C, alpha=0.2, label='+/- 2 sigma')
    if true_bias is not None:
        ax.axhline(true_bias, color=TRUTH, lw=1, ls='--', label='True bias')
    lim = 3 * max(abs(b[len(b) // 10:]).max(), 1e-3)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('time [s]'); ax.set_ylabel('gyro bias [rad/s]')
    ax.set_title('Gyro bias estimate')
    ax.legend()
    save(fig, name)


def bias_study(cfg, script, seeds):
    names = {'mu': 'EKF', 'mu_b': 'EKF + bias', 'enc': 'Encoder only', 'enc_gyro': 'Encoder + gyro'}
    lines = [f'Bias study, mean over {len(seeds)} runs: final position [m] / heading [deg], NEES',
             f'{"case":<26}' + ''.join(f'{n:>18}' for n in names.values()) + f'{"NEES EKF":>10}{"NEES +b":>9}']
    for Ls in [1.0, cfg.L_true_scale]:
        for gb in [0.0, 0.01]:
            c = replace(cfg, L_true_scale=Ls, gyro_bias=gb)
            truth = simulate_truth(c, script)
            runs = [run_once(c, s, truth) for s in seeds]
            fe = [final_errors(r) for r in runs]
            row = f'L error {100 * (Ls - 1):.0f}%, bias {gb:<5}'.ljust(26)
            for m in names:
                row += f'{np.mean([f[m][0] for f in fe]):>10.3f} /{np.degrees(np.mean([f[m][1] for f in fe])):>5.1f}'
            row += f'{np.mean([np.mean(nees_pose(r)[1:]) for r in runs]):>10.1f}'
            row += f'{np.mean([np.mean(nees_pose(r, "mu_b")[1:]) for r in runs]):>9.1f}'
            lines.append(row)
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(OUT, '10_bias_study.txt'), 'w') as f:
        f.write(text + '\n')


def summary_table(runs):
    names = {'mu': 'EKF', 'mu_b': 'EKF + bias', 'enc': 'Encoder only', 'enc_gyro': 'Encoder + gyro'}
    fe = [final_errors(r) for r in runs]
    lines = [f'Final errors, mean over {len(runs)} Monte Carlo runs',
             f'{"method":<16}{"position [m]":>14}{"heading [deg]":>15}']
    for m, name in names.items():
        p = np.mean([f[m][0] for f in fe])
        h = np.degrees(np.mean([f[m][1] for f in fe]))
        lines.append(f'{name:<16}{p:>14.4f}{h:>15.2f}')
    text = '\n'.join(lines)
    print(text)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, '9_summary.txt'), 'w') as f:
        f.write(text + '\n')


def make_all(cfg, script=DEFAULT_SCRIPT, out='outputs'):
    global OUT
    OUT = out
    truth = simulate_truth(cfg, script)
    res = run_once(cfg, cfg.seed, truth)
    plot_trajectory(res)
    plot_errors(res)
    plot_errors(res, key='mu_b', name='2_errors_bias.png')
    plot_nis(res)
    if cfg.gyro_bias:
        plot_bias(res, cfg.gyro_bias)

    runs = monte_carlo(cfg, range(50), truth)
    plot_nees(runs)
    summary_table(runs)

    plot_sensitivity(cfg, truth, range(10))
    key = 'mu_b' if cfg.gyro_bias else 'mu'
    plot_tuning(cfg, truth, range(10), key)
    make_animation(res, key=key)
