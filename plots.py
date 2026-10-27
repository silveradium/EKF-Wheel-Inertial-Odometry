"""All figures. Saved to outputs/."""
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
TRUTH, EKF_C, ENC_C, ENCG_C = '#333333', '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'figure.dpi': 120, 'axes.grid': True, 'grid.alpha': 0.3,
                     'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print('saved', name)


def ellipse(mean, cov, n_sigma=2):
    """Points on the n-sigma ellipse of a 2D Gaussian."""
    vals, vecs = np.linalg.eigh(cov)
    a = np.linspace(0, 2 * np.pi, 50)
    circle = np.array([np.cos(a), np.sin(a)])
    return (vecs @ (n_sigma * np.sqrt(vals)[:, None] * circle)).T + mean


# 1. trajectory
def plot_trajectory(res, every=150):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(res['true'][:, 0], res['true'][:, 1], color=TRUTH, lw=2, label='Truth')
    ax.plot(res['enc'][:, 0], res['enc'][:, 1], color=ENC_C, lw=1.5, label='Encoder only')
    ax.plot(res['mu'][:, 0], res['mu'][:, 1], color=EKF_C, lw=1.5, ls='--', label='EKF')
    for k in range(every, len(res['t']), every):
        e = ellipse(res['mu'][k, :2], res['Sigma'][k, :2, :2])
        ax.plot(e[:, 0], e[:, 1], color=EKF_C, lw=0.8)
    ax.plot([], [], color=EKF_C, lw=0.8, label='EKF 2-sigma ellipse')
    ax.plot(*res['true'][0, :2], 'o', color=TRUTH, ms=8)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_aspect('equal')
    ax.set_title('Trajectory (start = dot)')
    ax.legend()
    save(fig, '1_trajectory.png')


# 2. errors with 3-sigma bounds
def plot_errors(res):
    e = pose_error(res['true'], res['mu'])
    sd = np.sqrt(np.array([res['Sigma'][:, i, i] for i in range(3)])).T
    fig, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for i, (ax, lab) in enumerate(zip(axs, ['x error [m]', 'y error [m]', 'theta error [rad]'])):
        ax.plot(res['t'], e[:, i], color=EKF_C, lw=1, label='EKF error')
        ax.plot(res['t'], 3 * sd[:, i], color=TRUTH, lw=1, ls='--', label='+/- 3 sigma')
        ax.plot(res['t'], -3 * sd[:, i], color=TRUTH, lw=1, ls='--')
        ax.set_ylabel(lab)
    axs[0].legend(loc='upper left')
    axs[-1].set_xlabel('time [s]')
    axs[0].set_title('EKF error and its 3-sigma bound')
    save(fig, '2_errors.png')


# 3. NEES over Monte Carlo runs
def plot_nees(runs):
    nees = np.mean([nees_pose(r) for r in runs], axis=0)
    lo, hi = chi2_bounds(3, len(runs))
    t = runs[0]['t']
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t[1:], nees[1:], color=EKF_C, lw=1, label=f'Average NEES ({len(runs)} runs)')
    ax.axhline(3, color=TRUTH, lw=1, label='Expected value (3)')
    ax.axhline(lo, color=TRUTH, lw=1, ls='--', label='95% bounds')
    ax.axhline(hi, color=TRUTH, lw=1, ls='--')
    ax.set_xlabel('time [s]'); ax.set_ylabel('NEES (x, y, theta)')
    ax.set_title('Is the EKF uncertainty honest? (NEES)')
    ax.legend()
    save(fig, '3_nees.png')


# 4. NIS over time (single run)
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


# 5. sensitivity to gyro noise and encoder CPR
def plot_sensitivity(cfg, truth, seeds):
    def mean_final(c, tr):
        fe = [final_errors(run_once(c, s, tr)) for s in seeds]
        return [np.mean([f[m][0] for f in fe]) for m in ['mu', 'enc', 'enc_gyro']]

    sig = [0.001, 0.003, 0.01, 0.03, 0.1]
    cprs = [90, 180, 360, 720, 1440]
    a = np.array([mean_final(replace(cfg, sigma_g=s), truth) for s in sig])
    b = np.array([mean_final(replace(cfg, cpr=c), truth) for c in cprs])

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, xs, ys, xl in [(axs[0], sig, a, 'gyro noise sigma_g [rad/s]'),
                           (axs[1], cprs, b, 'encoder CPR [ticks/rev]')]:
        ax.plot(xs, ys[:, 0], 'o-', color=EKF_C, label='EKF')
        ax.plot(xs, ys[:, 2], 's--', color=ENCG_C, label='Encoder + gyro')
        ax.plot(xs, ys[:, 1], '^-', color=ENC_C, label='Encoder only')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(xl); ax.set_ylabel('final position error [m]')
    axs[0].legend()
    axs[0].set_title('Effect of gyro noise'); axs[1].set_title('Effect of encoder resolution')
    save(fig, '5_sensitivity.png')


# 6. process noise tuning
def plot_tuning(cfg, truth, seeds):
    scales = [0.01, 0.1, 1, 10, 100]
    err, nees = [], []
    for s in scales:
        c = replace(cfg, sigma_a=cfg.sigma_a * s, sigma_alpha=cfg.sigma_alpha * s)
        runs = [run_once(c, sd, truth) for sd in seeds]
        err.append(np.mean([final_errors(r)['mu'][0] for r in runs]))
        nees.append(np.mean([np.mean(nees_pose(r)[1:]) for r in runs]))
    lo, hi = chi2_bounds(3, len(seeds))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(scales, err, 'o-', color=EKF_C)
    axs[0].set_ylabel('final position error [m]')
    axs[0].set_title('Accuracy')
    axs[1].plot(scales, nees, 'o-', color=EKF_C, label='Mean NEES')
    axs[1].axhline(3, color=TRUTH, lw=1, label='Expected value (3)')
    axs[1].axhspan(lo, hi, color=TRUTH, alpha=0.1, label='95% bounds')
    axs[1].set_yscale('log'); axs[1].set_ylabel('mean NEES')
    axs[1].set_title('Honesty of the uncertainty')
    axs[1].legend()
    for ax in axs:
        ax.set_xscale('log')
        ax.set_xlabel('scale on sigma_a and sigma_alpha (1 = default)')
    save(fig, '6_tuning.png')


# 8. animation
def make_animation(res, every=10):
    fig, ax = plt.subplots(figsize=(6, 6))
    tr, mu = res['true'], res['mu']
    pad = 0.2
    ax.set_xlim(tr[:, 0].min() - pad, tr[:, 0].max() + pad)
    ax.set_ylim(tr[:, 1].min() - pad, tr[:, 1].max() + pad)
    ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    true_line, = ax.plot([], [], color=TRUTH, lw=2, label='Truth')
    ekf_line, = ax.plot([], [], color=EKF_C, lw=1.5, ls='--', label='EKF')
    robot, = ax.fill([0], [0], color=EKF_C, alpha=0.8)
    ell, = ax.plot([], [], color=EKF_C, lw=1, label='EKF 2-sigma ellipse (x10)')
    title = ax.set_title('')
    ax.legend(loc='upper right')
    frames = range(0, len(res['t']), every)

    def draw(k):
        true_line.set_data(tr[:k + 1, 0], tr[:k + 1, 1])
        ekf_line.set_data(mu[:k + 1, 0], mu[:k + 1, 1])
        x, y, th = mu[k, :3]
        s = 0.06  # triangle size
        pts = np.array([[s, 0], [-s / 2, s / 2], [-s / 2, -s / 2]])
        Rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        robot.set_xy(pts @ Rot.T + [x, y])
        # ellipse is tiny, so it is drawn 10x bigger to be visible
        e = ellipse(mu[k, :2], 100 * res['Sigma'][k, :2, :2])
        ell.set_data(e[:, 0], e[:, 1])
        title.set_text(f't = {res["t"][k]:.1f} s')
        return true_line, ekf_line, robot, ell, title

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    os.makedirs(OUT, exist_ok=True)
    anim.save(os.path.join(OUT, '8_animation.gif'), writer=PillowWriter(fps=15))
    plt.close(fig)
    print('saved 8_animation.gif')


# 9. summary table
def summary_table(runs):
    names = {'mu': 'EKF', 'enc': 'Encoder only', 'enc_gyro': 'Encoder + gyro'}
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
    truth = simulate_truth(cfg, script)  # same true motion for every run
    res = run_once(cfg, cfg.seed, truth)
    plot_trajectory(res)
    plot_errors(res)
    plot_nis(res)

    runs = monte_carlo(cfg, range(50), truth)
    plot_nees(runs)
    summary_table(runs)

    plot_sensitivity(cfg, truth, range(10))
    plot_tuning(cfg, truth, range(10))
    make_animation(res)
