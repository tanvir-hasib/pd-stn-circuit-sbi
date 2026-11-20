"""Remove the outputs of Oxford MRC BNDU cells (§11) from an executed notebook before committing it.

The Oxford printouts list file names, channels and per-hemisphere values derived from restricted data.
ds004998 outputs (public, CC0) are kept. Use --all to clear every output instead.

    python tools/strip_restricted_outputs.py executed.ipynb notebooks/pd_stn_circuit_sbi_executed.ipynb
"""
import argparse, json, re

OXFORD_CELL = re.compile(r"OX_ROOT|OXM|OXD|OXS|read_smr|MATRIX_DBS|_joint_cohort|JPOST_OX|Oxford")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    nb = json.load(open(a.src, encoding="utf-8")); n = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if a.all or (OXFORD_CELL.search(src) and not src.lstrip().startswith("from pathlib import Path")):
            if c.get("outputs"):
                n += 1
            c["outputs"] = []; c["execution_count"] = None
    json.dump(nb, open(a.dst, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"cleared outputs of {n} cell(s) → {a.dst}")


if __name__ == "__main__":
    main()
