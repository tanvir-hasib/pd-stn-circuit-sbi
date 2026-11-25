"""Copy the shareable tables and figures of a full run from OUT_DIR into results/.

Caches (*.npz, *.pt, *.joblib) are never copied. Oxford MRC BNDU-derived files are excluded unless
--include-oxford-aggregate is given, which adds only the cohort-level Oxford joint table (no per-patient values);
check the data-sharing terms before committing it. With --revision, the tables and figures of the revision
analyses (OUT_DIR/revision/, including the R13/ and R14/ subfolders and figures/) are copied to results/revision/,
with the same exclusions.

    python tools/collect_results.py --out-dir pd_pac_v4_outputs --dest results
    python tools/collect_results.py --out-dir pd_pac_v4_outputs --dest results --revision
"""
import argparse, re, shutil
from pathlib import Path

CACHE = re.compile(r"\.(npz|pt|joblib)$")
OXFORD = re.compile(r"(?i)^oxford|oxford|^R7_")
OXFORD_AGGREGATE = {"Oxford_STN-only_joint_group.csv"}
KEEP = (".csv", ".png", ".json", ".pdf", ".svg")
RENAME = {re.compile(r"^metrics_ds004998_[0-9a-f]+\.csv$"): "ds004998_statistics.csv",
          re.compile(r"^selection_[0-9a-f]+\.json$"): "ds004998_channel_selection.json"}


def copy_tree(src, dst, recursive, include_oxford_aggregate=False):
    copied, skipped = [], []
    files = sorted(src.rglob("*") if recursive else src.iterdir())
    for f in files:
        if not f.is_file() or CACHE.search(f.name) or f.suffix.lower() not in KEEP:
            continue
        if OXFORD.search(f.name) and not (include_oxford_aggregate and f.name in OXFORD_AGGREGATE):
            skipped.append(str(f.relative_to(src))); continue
        rel = f.relative_to(src)
        name = next((new for pat, new in RENAME.items() if pat.match(f.name)), f.name)
        out = dst / rel.parent / name; out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out); copied.append(str(rel.parent / name))
    return copied, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="pd_pac_v4_outputs")
    ap.add_argument("--dest", default="results")
    ap.add_argument("--include-oxford-aggregate", action="store_true")
    ap.add_argument("--revision", action="store_true", help="also copy OUT_DIR/revision/ to DEST/revision/")
    a = ap.parse_args()
    src, dst = Path(a.out_dir), Path(a.dest); dst.mkdir(parents=True, exist_ok=True)
    jobs = [(src, dst, False)]
    if a.revision:
        if (src / "revision").is_dir():
            jobs.append((src / "revision", dst / "revision", True))
        else:
            print(f"no {src / 'revision'} folder: run the revision notebooks first")
    for s, d, rec in jobs:
        copied, skipped = copy_tree(s, d, rec, a.include_oxford_aggregate)
        print(f"copied {len(copied)} file(s) to {d}/:"); [print("  ", n) for n in copied]
        if skipped:
            print(f"excluded {len(skipped)} Oxford-derived file(s):"); [print("  ", n) for n in skipped]


if __name__ == "__main__":
    main()
