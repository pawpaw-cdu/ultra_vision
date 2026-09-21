#!/usr/bin/env python3
"""Accuracy regression against simulator ground truth.

The simulator writes dataset.csv with the true camera-frame position of the four
armor plates. Comparing our recorded CSVs against it turns "the estimate looks
wrong" into numbers that say which stage is actually off:

  * observation error  -> detector + PnP quality (before the estimator)
  * chassis center     -> the whole-chassis estimator
  * plate radius       -> whether the geometric model matches the target
  * omega              -> the yaw-rate estimate under rotation

Usage (recorded run):
    truth_regression.py --truth truth.csv --observation observation.csv \
                        --estimate estimate.csv [--selector selector.csv]

Exit status is non-zero when a metric exceeds its threshold, so it can gate CI.
`--selftest` exercises the metric code on synthetic data and needs no recording.
"""

import argparse
import bisect
import csv
import math
import random
import statistics as st
import sys


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_truth(path):
    """time_us -> list of (x, y, z) plate centers."""
    frames = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            t = _num(row.get("time_us"))
            x, y, z = _num(row.get("x")), _num(row.get("y")), _num(row.get("z"))
            if None in (t, x, y, z):
                continue
            frames.setdefault(int(t), []).append((x, y, z))
    return frames


def nearest_frame(times, t, tolerance_us):
    index = bisect.bisect_left(times, t)
    best = None
    for candidate in (index - 1, index, index + 1):
        if 0 <= candidate < len(times):
            delta = abs(times[candidate] - t)
            if best is None or delta < best[0]:
                best = (delta, times[candidate])
    if best is None or best[0] > tolerance_us:
        return None
    return best[1]


def load_points(path):
    points = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            t = _num(row.get("time_us"))
            x, y, z = _num(row.get("x")), _num(row.get("y")), _num(row.get("z"))
            if None in (t, x, y, z):
                continue
            points.append((int(t), (x, y, z)))
    return points


def observation_error(truth, observation, tolerance_us):
    """Distance from each observed plate to the closest true plate."""
    times = sorted(truth)
    errors = []
    for t, point in observation:
        frame = nearest_frame(times, t, tolerance_us)
        if frame is None:
            continue
        errors.append(min(math.dist(point, plate) for plate in truth[frame]))
    return errors


def center_stats(truth, estimate, tolerance_us):
    """Chassis-center error and estimated radius, grouped per frame."""
    times = sorted(truth)
    grouped = {}
    for t, point in estimate:
        grouped.setdefault(t, []).append(point)

    errors = []
    radii = []
    for t, points in grouped.items():
        if len(points) < 4:
            continue
        frame = nearest_frame(times, t, tolerance_us)
        if frame is None:
            continue
        center = tuple(sum(p[i] for p in points) / 4 for i in range(3))
        true_center = tuple(sum(p[i] for p in truth[frame]) / 4 for i in range(3))
        errors.append(math.dist(center, true_center))
        mean_radius = sum(math.hypot(p[0] - center[0], p[2] - center[2])
                          for p in points) / 4
        radii.append(mean_radius)
    return errors, radii


def load_omega(path):
    values = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            omega = _num(row.get("omega"))
            if omega is not None:
                values.append(omega)
    return values


def summarize(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": st.mean(values),
        "median": st.median(values),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def report(name, values, unit="m"):
    stats = summarize(values)
    if stats is None:
        print(f"{name:<28} (no data)")
        return None
    print(f"{name:<28} n={stats['n']:<5} mean={stats['mean']:.3f} "
          f"median={stats['median']:.3f} p90={stats['p90']:.3f} "
          f"max={stats['max']:.3f} {unit}")
    return stats


def selftest():
    """Exercise the metric code on synthetic data with a known offset."""
    truth = {}
    observation = []
    estimate = []
    random.seed(7)
    radius = 0.21
    for frame in range(60):
        t = 1_000_000 + frame * 33_000
        plates = []
        for plate in range(4):
            angle = plate * math.pi / 2
            x = radius * math.cos(angle)
            z = 1.5 + radius * math.sin(angle)
            plates.append((x, 0.0, z))
        truth[t] = plates

        # observations: 2 cm of isotropic noise
        obs = tuple(plates[0][i] + random.gauss(0, 0.02) for i in range(3))
        observation.append((t, obs))

        # estimate: known 3 cm offset in x, plates on the correct circle
        for i, plate in enumerate(plates):
            estimate.append((t, (plate[0] + 0.03, plate[1], plate[2])))

    obs_stats = summarize(observation_error(truth, observation, 20_000))
    center_errors, radii = center_stats(truth, estimate, 20_000)
    center_stats_out = summarize(center_errors)
    radius_stats = summarize(radii)

    ok = True
    def check(condition, message):
        nonlocal ok
        if not condition:
            ok = False
            print("FAILED: " + message)

    check(obs_stats is not None and 0.010 < obs_stats["mean"] < 0.040,
          f"observation noise not recovered: {obs_stats}")
    check(center_stats_out is not None and abs(center_stats_out["mean"] - 0.03) < 0.005,
          f"injected 0.03 m center offset not recovered: {center_stats_out}")
    check(radius_stats is not None and abs(radius_stats["mean"] - radius) < 0.005,
          f"radius not recovered: {radius_stats}")
    if not ok:
        return 1
    print(f"truth_regression selftest passed "
          f"(obs={obs_stats['mean']:.3f} m, center={center_stats_out['mean']:.3f} m, "
          f"radius={radius_stats['mean']:.3f} m)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--truth")
    parser.add_argument("--observation")
    parser.add_argument("--estimate")
    parser.add_argument("--selector")
    parser.add_argument("--tolerance-us", type=int, default=25_000,
                        help="max timestamp gap when matching frames")
    parser.add_argument("--max-observation-error", type=float, default=0.05)
    parser.add_argument("--max-center-error", type=float, default=0.06)
    parser.add_argument("--expected-radius", type=float, default=0.21)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()
    if not (args.truth and args.observation):
        parser.error("--truth and --observation are required (or use --selftest)")

    truth = load_truth(args.truth)
    if not truth:
        print("no truth frames loaded", file=sys.stderr)
        return 2
    print(f"truth frames: {len(truth)}")

    failed = False

    obs_stats = report("observation error",
                       observation_error(truth, load_points(args.observation),
                                         args.tolerance_us))
    if obs_stats and obs_stats["median"] > args.max_observation_error:
        print(f"FAILED: observation error median {obs_stats['median']:.3f} m "
              f"> {args.max_observation_error} m (detector/PnP stage)")
        failed = True

    if args.estimate:
        center_errors, radii = center_stats(truth, load_points(args.estimate),
                                            args.tolerance_us)
        center_stats_out = report("chassis center error", center_errors)
        radius_stats = report("estimated plate radius", radii)
        if center_stats_out and center_stats_out["median"] > args.max_center_error:
            print(f"FAILED: center error median {center_stats_out['median']:.3f} m "
                  f"> {args.max_center_error} m (estimator stage)")
            failed = True
        if radius_stats and abs(radius_stats["median"] - args.expected_radius) > 0.02:
            print(f"FAILED: plate radius {radius_stats['median']:.3f} m is not the "
                  f"configured {args.expected_radius} m (geometry model)")
            failed = True

    if args.selector:
        omegas = load_omega(args.selector)
        omega_stats = report("omega", omegas, unit="rad/s")
        if omega_stats:
            print(f"{'omega magnitude':<28} mean={st.mean(abs(v) for v in omegas):.3f} "
                  f"median={st.median(abs(v) for v in omegas):.3f} rad/s")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
