"""Copy the shareable tables and figures of a full run from OUT_DIR into results/.

Caches (*.npz, *.pt, *.joblib) are never copied. Oxford MRC BNDU-derived files are excluded unless
--include-oxford-aggregate is given, which adds only the cohort-level Oxford joint table (no per-patient values);
check the data-sharing terms before committing it.

    python tools/collect_results.py --out-dir pd_pac_v4_outputs --dest results
"""
import argparse, re, shutil
from pathlib import Path

CACHE = re.compile(r"\.(npz|pt|joblib)$")
OXFORD = re.compile(r"(?i)^oxford")
OXFORD_AGGREGATE = {"Oxford_STN-only_joint_group.csv"}
RENAME = {re.compile(r"^metrics_ds004998_[0-9a-f]+\.csv$"): "ds004998_statistics.csv",
          re.compile(r"^selection_[0-9a-f]+\.json$"): "ds004998_channel_selection.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="pd_pac_v4_outputs")
    ap.add_argument("--dest", default="results")
    ap.add_argument("--include-oxford-aggregate", action="store_true")
    a = ap.parse_args()
    src, dst = Path(a.out_dir), Path(a.dest); dst.mkdir(parents=True, exist_ok=True)
    copied, skipped = [], []
    for f in sorted(src.iterdir()):
        if not f.is_file() or CACHE.search(f.name):
            continue
        if OXFORD.search(f.name) and not (a.include_oxford_aggregate and f.name in OXFORD_AGGREGATE):
            skipped.append(f.name); continue
        if f.suffix.lower() not in (".csv", ".png", ".json", ".pdf", ".svg"):
            continue
        name = next((new for pat, new in RENAME.items() if pat.match(f.name)), f.name)
        shutil.copy2(f, dst / name); copied.append(name)
    print(f"copied {len(copied)} file(s) to {dst}/:"); [print("  ", n) for n in copied]
    if skipped:
        print(f"excluded {len(skipped)} Oxford-derived file(s):"); [print("  ", n) for n in skipped]


if __name__ == "__main__":
    main()
