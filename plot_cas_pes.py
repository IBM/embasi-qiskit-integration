import csv
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Crop with:  plot_cas_pes.py --xmax 2.2
XLIM = (0.75, 2.55)
XMAX_DATA = None  # drop points beyond this R before plotting
OUTNAME = "cas_pes.png"

ROOT = Path(__file__).resolve().parent
XLSX = ROOT / "jphychemlett_2023_data[81].xlsx"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

# One entry per panel: (title, reference sheets, EmbASI csvs).
PANELS = [
    # (
    #     "AS(4,4)",
    #     {},
    #     {"cas_4_4_EXACT_fci.csv": "EmbASI + FCI",
    #      "cas_4_4_aer.csv": "EmbASI + Qiskit Aer (SqDRIFT)",
    #      "cas_4_4_EXACT_aer.csv": "EmbASI + NEW Aer (SqDRIFT)",
    #      "cas_4_4_EXACT_fake_kingston.csv" : "EmbASI + Qiskit fake_kingston (SqDRIFT)",
    #      "cas_4_4_ibm_kingston.csv" : "EmbASI + Qiskit ibm_kingston (SqDRIFT)"
    #      }
    # ),
    (
        "AS(6,6)",
        {},
        {"cas_6_6_fci.csv": "EmbASI + FCI",
         "cas_6_6_EXACT_fci.csv": "EmbASI + EXACT FCI",
         "cas_6_6_aer.csv": "EmbASI + Qiskit Aer (SqDRIFT)",
         "cas_6_6_EXACT_aer.csv": "EmbASI + EXACT Qiskit Aer (SqDRIFT)",
         "cas_6_6_fake_kingston.csv" : "EmbASI + Qiskit fake_kingston (SqDRIFT)",
         "cas_6_6_ibm_kingston.csv" : "EmbASI + Qiskit ibm_kingston (SqDRIFT)"
         }
    ),
]

# label -> (color, marker, linestyle); shared so both panels look identical.
STYLES = [
    ("EmbASI + FCI", "tab:blue", "s", "-"),
    ("EmbASI + EXACT FCI", "lightskyblue", "s", "-"),
    ("EmbASI + Qiskit Aer (SqDRIFT)", "magenta", "x", "--"),
    ("EmbASI + EXACT Qiskit Aer (SqDRIFT)", "purple", "*", "--"),
    ("EmbASI + Qiskit fake_kingston (SqDRIFT)", "grey", ">", "-."),
    ("EmbASI + Qiskit ibm_kingston (SqDRIFT)", "orange", ".", ":"),
]


def read_xlsx_columns(path, wanted_sheets, columns=("A", "E")):
    """Read the given letter columns of the named sheets from an .xlsx file."""
    with zipfile.ZipFile(path) as z:
        strings = [
            "".join(t.text or "" for t in si.iter(f"{NS}t"))
            for si in ET.fromstring(z.read("xl/sharedStrings.xml"))
        ]
        rels = {
            r.get("Id"): r.get("Target")
            for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        }
        book = ET.fromstring(z.read("xl/workbook.xml"))
        rid = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

        out = {}
        for sheet in book.iter(f"{NS}sheet"):
            name = sheet.get("name")
            if name not in wanted_sheets:
                continue
            target = rels[sheet.get(rid)].lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            xs, ys = [], []
            for row in ET.fromstring(z.read(target)).iter(f"{NS}row"):
                cells = {}
                for c in row.iter(f"{NS}c"):
                    col = re.match(r"[A-Z]+", c.get("r")).group()
                    if col not in columns:
                        continue
                    v = c.find(f"{NS}v")
                    val = v.text if v is not None else None
                    if c.get("t") == "s" and val is not None:
                        val = strings[int(val)]
                    cells[col] = val
                try:  # non-numeric rows (the header) fall out here
                    x = float(cells[columns[0]])
                    y = float(cells[columns[1]])
                except (KeyError, TypeError, ValueError):
                    continue
                xs.append(x)
                ys.append(y)
            out[name] = (xs, ys)
    return out


def read_energy_csv(path):
    """Parse 'data/NN.inp; energy' rows into (distance/Ang, energy/Ha) arrays.

    Later rows win, so a partially rewritten file reports its newest values.
    Rows may run together when the writer omits the trailing newline, so match
    every occurrence rather than splitting on lines.
    """
    seen = {}
    with open(path, newline="") as fh:
        text = fh.read()
    for stem, energy in re.findall(r"(\d+)\.inp;\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)", text):
        seen[int(stem) / 10.0] = float(energy)
    xs = sorted(seen)
    return xs, [seen[x] for x in xs]


def _interp_at(xs_ref, xs_src, ys_src):
    """Return ys_src linearly interpolated onto xs_ref; drops points outside src range."""
    result = {}
    for x in xs_ref:
        # find the two nearest src points that bracket x
        lo = hi = None
        for i, xv in enumerate(xs_src):
            if abs(xv - x) < 1e-9:
                result[x] = ys_src[i]
                break
            if xv < x:
                lo = i
            elif xv > x and hi is None:
                hi = i
        else:
            if x not in result and lo is not None and hi is not None:
                t = (x - xs_src[lo]) / (xs_src[hi] - xs_src[lo])
                result[x] = ys_src[lo] + t * (ys_src[hi] - ys_src[lo])
    xs_out = sorted(result)
    return xs_out, [result[xv] for xv in xs_out]


def main():
    n = len(PANELS)
    fig, axes = plt.subplots(
        2, n, figsize=(10.0, 7.0), dpi=200, sharex=True, squeeze=False,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    top_axes = axes[0, :]
    bot_axes = axes[1, :]

    FCI_LABEL = "EmbASI + FCI"

    all_data = []
    for ax, (title, ref_sheets, csv_series) in zip(top_axes, PANELS):
        refs = read_xlsx_columns(XLSX, set(ref_sheets))
        data = {label: refs[sheet] for sheet, label in ref_sheets.items() if sheet in refs}
        for fname, label in csv_series.items():
            path = ROOT / "output" / fname
            if path.exists():
                data[label] = read_energy_csv(path)
        all_data.append(data)

        for label, color, marker, ls in STYLES:
            if label not in data or not data[label][0]:
                continue
            xs, ys = data[label]
            if XMAX_DATA is not None:
                keep = [i for i, x in enumerate(xs) if x <= XMAX_DATA + 1e-9]
                xs = [xs[i] for i in keep]
                ys = [ys[i] for i in keep]
            ax.plot(
                xs, ys,
                color=color, marker=marker, linestyle=ls,
                markersize=4.5, linewidth=1.4, markeredgewidth=1.4,
                markerfacecolor="none" if marker in ("x", "s") else color,
                label=label,
            )

        ax.set_title(title)
        ax.set_xlim(*XLIM)
        ax.tick_params(direction="in", top=True, right=True)
        # Headroom so the legend clears the curves.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.28 * (hi - lo))
        ax.legend(frameon=False, fontsize=8, loc="upper left",
                  bbox_to_anchor=(0.28, 1.0), handlelength=2.4, borderaxespad=0.4)
        print(f"{title}:")
        for label in (l for l, *_ in STYLES if l in data):
            print(f"  {label}: {len(data[label][0])} points")

    top_axes[0].set_ylabel("Energy (Hartree)")

    # --- error panels ---
    for ax, data in zip(bot_axes, all_data):
        if FCI_LABEL not in data or not data[FCI_LABEL][0]:
            ax.set_visible(False)
            continue
        xs_fci, ys_fci = data[FCI_LABEL]
        for label, color, marker, ls in STYLES:
            if label == FCI_LABEL or label not in data or not data[label][0]:
                continue
            xs_m, ys_m = data[label]
            if XMAX_DATA is not None:
                keep = [i for i, x in enumerate(xs_m) if x <= XMAX_DATA + 1e-9]
                xs_m = [xs_m[i] for i in keep]
                ys_m = [ys_m[i] for i in keep]
            xs_err, ys_err = _interp_at(xs_m, xs_fci, ys_fci)
            ys_delta = [ym - yf for ym, yf in zip(
                [ys_m[xs_m.index(x)] for x in xs_err], ys_err
            )]
            ax.plot(
                xs_err, ys_delta,
                color=color, marker=marker, linestyle=ls,
                markersize=4.5, linewidth=1.4, markeredgewidth=1.4,
                markerfacecolor="none" if marker in ("x", "s") else color,
                label=label,
            )
        ax.axhline(0, color="tab:blue", linewidth=0.8, linestyle="-")
        ax.set_xlim(*XLIM)
        ax.tick_params(direction="in", top=True, right=True)

    bot_axes[0].set_ylabel(r"$\Delta E_{\mathrm{x-FCI}}$ (Hartree)")
    for ax in bot_axes:
        ax.set_xlabel(r"$R$ ($\mathrm{\AA}$)")

    fig.tight_layout()
    out = ROOT / "output" / OUTNAME
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xmax", type=float, help="upper R limit in Angstrom")
    ap.add_argument("--out", help="output filename under output/")
    a = ap.parse_args()
    if a.xmax is not None:
        # Trim the data at xmax, then pad the view so the last point has room.
        globals()["XMAX_DATA"] = a.xmax
        globals()["XLIM"] = (XLIM[0], a.xmax + 0.12)
        globals()["OUTNAME"] = a.out or f"cas_pes_to{a.xmax:g}".replace(".", "p") + ".png"
    elif a.out:
        globals()["OUTNAME"] = a.out
    main()
