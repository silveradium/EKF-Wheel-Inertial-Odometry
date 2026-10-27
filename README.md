# EKF wheel + gyro odometry (simulation)

Setup (once):

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt

Run everything (about 1 minute, figures go to `outputs/`):

    .venv/bin/python run.py

Straight-line test (figures go to `trajectory_1/`):

    .venv/bin/python run.py straight

Real data, NCLT session 2012-01-08 (figures and summary go to `nclt_outputs/`).
Needs `NCLT/groundtruth_2012-01-08.csv` and `wheels.csv`, `ms25.csv`, `kvh.csv`
extracted from `2012-01-08_sen.tar.gz` into `NCLT/2012-01-08/`:

    .venv/bin/python nclt.py

Tests:

    .venv/bin/python -m pytest -q tests

| file | what it does |
|---|---|
| config.py | all parameters |
| sim.py | true robot motion from teleop keys, encoder + gyro sensors, gyro interpolation onto encoder times |
| ekf.py | the EKF (g, G, h, H, predict, update) |
| baselines.py | encoder-only and encoder+gyro dead reckoning |
| run.py | one run, Monte Carlo runs, NEES |
| plots.py | all figures |
| nclt.py | load, calibrate and test on the real NCLT data |
# EKF-Wheel-Inertial-Odometry
