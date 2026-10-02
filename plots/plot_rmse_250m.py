"""Re-plot figRMSE_250M from the saved benchmark .mat results.

Python port of localPerRangeFigure_ in Copy/build_rmse_figures.m. Every value
is read from the MAT-files inside GPT_3.zip; nothing is taken from the PNG.

Axis limits are shared across ranges exactly as in the MATLAB script, so the
other ranges' result files are loaded too (only for the limits). The original
figures_rmse_seed41_20260928_171520 run used ranges 250 500 600 603 606 612 750.

    python3 plots/plot_rmse_250m.py
    python3 plots/plot_rmse_250m.py --range 600 --out plots/figRMSE_600M.png
"""

import argparse
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RESULT_DIR = "GPT_3/Copy/results"
SHARED_RANGES = [250, 500, 600, 603, 606, 612, 750]

# MATLAB default line colours and the script's styles {'--o','-s','-^','-d'}.
COLORS = [(0.0, 0.447, 0.741), (0.850, 0.325, 0.098),
          (0.929, 0.694, 0.125), (0.494, 0.184, 0.556)]
STYLES = [("--", "o"), ("-", "s"), ("-", "^"), ("-", "d")]


def load_result(zf, rangeM, seed):
    member = f"{RESULT_DIR}/spatial_ace_R{rangeM}_seed{seed}.mat"
    raw = zf.read(member)
    m = sio.loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)
    assert "output" in m, f"{member} has no 'output'."
    return m["output"], method_names(m, m["output"], member)


def method_names(m, o, member):
    """Method.Name is a MATLAB string object (MCOS), which scipy cannot decode.

    Its UTF-16 text sits in the MAT-file subsystem block. The first distinct
    strings there are taken in order, then checked against the block size the
    same file records for each method in Results.Complexity.Method.
    """
    fw = m["__function_workspace__"].tobytes()
    found = []
    for t in re.findall(rb"(?:[\x20-\x7e]\x00){4,}", fw):
        s = t.decode("utf-16le")
        if s not in found:
            found.append(s)
    nM = len(o.Results.Method)
    names = found[:nM]
    blocks = [int(c.BlockSizeRangeBins) for c in o.Results.Complexity.Method]
    for n, b in zip(names, blocks):
        ok = n.startswith("Full") if b == 1 else n.startswith(f"Block-{b} ")
        assert ok, f"{member}: name '{n}' does not match block size {b}."
    return names


def axis_limit(v):
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [0.0, 1.0]
    hi = v.max() * 1.08
    return [0.0, hi if hi > 0 else 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(REPO / "GPT_3.zip"))
    ap.add_argument("--range", type=int, default=250)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_png = Path(args.out or REPO / "plots" / f"figRMSE_{args.range}M.png")

    ranges = list(dict.fromkeys(SHARED_RANGES + [args.range]))
    with zipfile.ZipFile(args.zip) as zf:
        loaded = {r: load_result(zf, r, args.seed) for r in ranges}

    o, names = loaded[args.range]
    off = np.atleast_1d(o.Results.TargetPowerOffsetDB).astype(float)
    rangeBinM = float(np.median(np.diff(o.RangeGridM)))
    dopBinHz = float(o.DopplerGridSpacingHz)

    # Same consistency checks and shared limits as build_rmse_figures.m.
    allRng, allDop, allN = [], [], []
    for r, (ok, nk) in loaded.items():
        assert np.array_equal(np.atleast_1d(ok.Results.TargetPowerOffsetDB), off), \
            f"{r} m was swept on a different target-power grid."
        assert nk == names, f"{r} m has a different method set."
        assert abs(np.median(np.diff(ok.RangeGridM)) - rangeBinM) < 1e-9 and \
            abs(ok.DopplerGridSpacingHz - dopBinHz) < 1e-9, \
            f"{r} m has a different range/Doppler grid spacing."
        for mm in ok.Results.Method:
            allRng.append(np.atleast_1d(mm.RangeRMSEPowerSweepM).astype(float))
            allDop.append(np.atleast_1d(mm.DopplerRMSEPowerSweepHz).astype(float))
            allN.append(np.atleast_1d(mm.NumRMSESamplesPowerSweep).astype(float))
    allRng, allDop, allN = map(np.concatenate, (allRng, allDop, allN))

    dopLim = axis_limit(allDop / dopBinHz)
    nLim = [0.0, max(1.0, allN.max()) * 1.08]
    xLim = [off.min() - 1.5, off.max() + 1.5]
    capBins, extendedBins = 1.0, 1.4

    M = o.Results.Method
    v = np.concatenate([np.atleast_1d(mm.RangeRMSEPowerSweepM) for mm in M]) / rangeBinM
    v = v[np.isfinite(v)]
    rngMax = v.max() if v.size else 0.0
    assert rngMax <= extendedBins + 1e-9, \
        f"Range RMSE reaches {rngMax:.3f} bins, beyond the {extendedBins}-bin axis."
    extended = rngMax > capBins
    rngLim = [0.0, extendedBins if extended else capBins]

    fig, axs = plt.subplots(1, 3, figsize=(1180 / 96, 430 / 96), layout="constrained")
    panels = [
        ("RangeRMSEPowerSweepM", 1 / rangeBinM, rngLim, "Range RMSE (bins)",
         f"Range  (1 bin = {rangeBinM:.3f} m)"
         + (f"  -  axis extended to {extendedBins:.1f} bins" if extended else ""), True),
        ("DopplerRMSEPowerSweepHz", 1 / dopBinHz, dopLim, "Doppler RMSE (bins)",
         f"Doppler  (1 bin = {dopBinHz:.2f} Hz)", True),
        ("NumRMSESamplesPowerSweep", 1.0, nLim, "Samples detected",
         "Samples detected", False),
    ]
    for ax, (field, scale, ylim, ylab, title, unitLine) in zip(axs, panels):
        for k, mm in enumerate(M):
            y = np.atleast_1d(getattr(mm, field)).astype(float) * scale
            ls, mk = STYLES[min(k, len(STYLES) - 1)]
            ax.plot(off, y, linestyle=ls, marker=mk, color=COLORS[k % len(COLORS)],
                    linewidth=1.4, markersize=6, markerfacecolor="none",
                    markeredgewidth=1.4, label=names[k])
        if unitLine:
            ax.axhline(1, linestyle=":", color="k", linewidth=0.6)
        ax.set_xlim(xLim)
        ax.set_xticks(off)
        ax.set_ylim(ylim)
        ax.grid(True, color="0.88")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("Target power offset (dB)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontweight="bold")
    axs[2].legend(loc="best")
    fig.suptitle(f"{args.range} m, seed {int(o.ChannelSeed)} - axes shared with every "
                 "other range; RMSE conditional on association")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)

    print(f"{args.range} m - associated detections behind each RMSE point")
    print(f"{'offset dB:':24s}" + "".join(f"{int(x):7d}" for x in off))
    for k, mm in enumerate(M):
        print(f"{names[k]:24s}" + "".join(f"{int(x):7d}" for x in np.atleast_1d(mm.NumRMSESamplesPowerSweep))
              + f"   (nominal n={int(mm.NumRMSESamples)})")
    print(f"Shared axes: target power {xLim[0]:g}..{xLim[1]:g} dB; "
          f"Doppler RMSE {dopLim[0]:.3f}..{dopLim[1]:.3f} bins; range axis to {rngLim[1]:.1f} bins")
    print(f"Bin spacing: {rangeBinM:.6f} m, {dopBinHz:.4f} Hz")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
