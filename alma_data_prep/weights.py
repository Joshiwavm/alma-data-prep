"""Measure and correct the weights of channel-averaged visibilities.

statwt gives correct per-channel weights, but ALMA channels are correlated
(Hanning smoothing in TDM windows, a different response in the ACA and FDM
windows). Averaging an spw to one channel with `split(width=nchan)` sums
the channel weights as if the channels were independent, so the weights of
the averaged visibilities are too large by a factor that depends on the
window (~2.7 for 12-m TDM, ~2.9 for 7-m, ~1.2 for FDM).

The factor is measured here from the visibilities themselves, never taken
from metadata: consecutive integrations of one baseline see the same sky,
so their difference is pure noise. With z2 = |V_j - V_k|^2 w_pair and
w_pair = 1 / (1/w_j + 1/w_k), the weighted mean of z2 / 2 is the factor
by which the weights are too large, as a natural-weighted map sees it.

Only numpy and casatools are needed (no package-relative imports), so this
file also runs as a script:

    python weights.py NPZ_ROOT OUT_ROOT MS [MS ...]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil
from typing import Iterable, List, Tuple

import numpy as np

C_LIGHT = 299792458.0  # m/s
MAX_PAIR_GAP = 450.0  # s; covers one mosaic cycle (~200-400 s)
W_ROW = 4  # rows of the NPZ arr_0: u, v, re, im, w, freq
NPZ_NAME = re.compile(r"output_.*\.im\.field-(\d+)\.spw-(\d+)\.data\.npz$")


def baseline_id(ant1: np.ndarray, ant2: np.ndarray) -> np.ndarray:
    return np.asarray(ant1) * 1000 + np.asarray(ant2)


def time_pairs(time: np.ndarray, base: np.ndarray,
               max_gap: float = MAX_PAIR_GAP) -> Tuple[np.ndarray, np.ndarray]:
    """Pairs (j, k) of consecutive integrations of the same baseline,
    at most `max_gap` seconds apart; each row is used at most once."""
    order = np.lexsort((time, base))
    t, b = time[order], base[order]
    # A run is a stretch of one baseline without gaps > max_gap; within
    # each run take rows (0, 1), (2, 3), ...
    continues = (b[1:] == b[:-1]) & (np.diff(t) <= max_gap)
    run = np.r_[0, np.cumsum(~continues)]
    pos_in_run = np.arange(len(t)) - np.searchsorted(run, run)
    first = np.flatnonzero((pos_in_run[:-1] % 2 == 0) & continues)
    return order[first], order[first + 1]


def variance_scale(re_: np.ndarray, im_: np.ndarray, w: np.ndarray,
                   time: np.ndarray, base: np.ndarray,
                   max_gap: float = MAX_PAIR_GAP) -> Tuple[float, float, int]:
    """Factor by which the weights are too large: (scale, median-based
    scale as an outlier check, number of pairs). 1 for correct weights."""
    j, k = time_pairs(time, base, max_gap)
    good = (w[j] > 0) & (w[k] > 0)
    j, k = j[good], k[good]
    w_pair = 1 / (1 / w[j] + 1 / w[k])
    z2 = ((re_[j] - re_[k]) ** 2 + (im_[j] - im_[k]) ** 2) * w_pair
    scale = np.sum(w_pair * z2) / np.sum(w_pair) / 2
    robust = np.median(z2) / (2 * np.log(2))
    return float(scale), float(robust), int(len(j))


def npz_row_meta(data: np.ndarray, ms_path: str, field: int,
                 spw: int) -> Tuple[np.ndarray, np.ndarray]:
    """Time and baseline id of each row of an exported NPZ, from its source
    MS. The NPZ rows are the MS rows of (field, spw) with an unflagged
    channel in both polarisations (possibly reordered by `split`); they are
    matched on (u, v). Raises ValueError if the MS rows do not match."""
    from casatools import ms as ms_tool

    u, v, freq = data[0], data[1], data[5]
    tool = ms_tool()
    tool.open(ms_path)
    try:
        tool.selectinit(datadescid=spw)
        tool.select({"field_id": field})
        rec = tool.getdata(["u", "v", "time", "antenna1", "antenna2",
                            "flag"])
    finally:
        tool.close()
    if "flag" not in rec:
        raise ValueError(f"{ms_path}: no rows for field {field} spw {spw}")
    kept = (~rec["flag"][0]).any(0) & (~rec["flag"][1]).any(0)
    if kept.sum() != u.size:
        raise ValueError(f"{ms_path}: {kept.sum()} rows, NPZ has {u.size}")
    u_ms = rec["u"][kept] * freq[0] / C_LIGHT
    v_ms = rec["v"][kept] * freq[0] / C_LIGHT
    sort_ms, sort_npz = np.lexsort((u_ms, v_ms)), np.lexsort((u, v))
    if not (np.allclose(u_ms[sort_ms], u[sort_npz])
            and np.allclose(v_ms[sort_ms], v[sort_npz])):
        raise ValueError(f"{ms_path}: (u, v) do not match the NPZ")
    ms_row = np.empty_like(sort_ms)  # MS row of each NPZ row
    ms_row[sort_npz] = sort_ms
    base = baseline_id(rec["antenna1"], rec["antenna2"])
    return rec["time"][kept][ms_row], base[kept][ms_row]


def reweight_exports(npz_root: str, ms_paths: Iterable[str], out_root: str,
                     max_gap: float = MAX_PAIR_GAP) -> List[dict]:
    """Copy every `<array>/output_*.data.npz` under `npz_root` to
    `out_root` with its weights divided by its own measured scale, copy the
    matching FITS products, and write `out_root/weight_scales.csv`."""
    if os.path.realpath(out_root) == os.path.realpath(npz_root):
        raise ValueError("out_root must differ from npz_root")
    ms_paths = list(ms_paths)
    rows = []
    for path in sorted(glob.glob(os.path.join(npz_root, "*", "*.data.npz"))):
        match = NPZ_NAME.search(os.path.basename(path))
        if not match:
            continue
        field, spw = map(int, match.groups())
        data = np.load(path)["arr_0"]
        for ms_path in ms_paths:
            try:
                time, base = npz_row_meta(data, ms_path, field, spw)
                break
            except (ValueError, RuntimeError):
                continue
        else:
            raise ValueError(f"no source MS matches {path}")
        scale, robust, npairs = variance_scale(
            data[2], data[3], data[W_ROW], time, base, max_gap)

        out_dir = os.path.join(out_root, os.path.basename(
            os.path.dirname(path)))
        os.makedirs(out_dir, exist_ok=True)
        fixed = data.copy()
        fixed[W_ROW] /= scale
        np.savez_compressed(os.path.join(out_dir, os.path.basename(path)),
                            fixed)
        for fits in glob.glob(path.replace(".data.npz", ".*.fits")):
            shutil.copy2(fits, out_dir)
        rows.append(dict(file=os.path.relpath(path, npz_root), field=field,
                         spw=spw, ms=os.path.basename(ms_path),
                         npairs=npairs, scale=f"{scale:.6f}",
                         robust_scale=f"{robust:.6f}"))
        print(rows[-1], flush=True)
    write_scales(os.path.join(out_root, "weight_scales.csv"), rows)
    return rows


def write_scales(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("npz_root", help="dir with <array>/ NPZ exports")
    parser.add_argument("out_root", help="dir for the corrected copies")
    parser.add_argument("ms", nargs="+", help="candidate source MSs")
    args = parser.parse_args()
    reweight_exports(args.npz_root, args.ms, args.out_root)


if __name__ == "__main__":
    main()
