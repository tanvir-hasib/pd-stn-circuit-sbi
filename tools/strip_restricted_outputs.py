"""Remove restricted (Oxford MRC BNDU-derived) outputs from an executed notebook before committing it.

Cleared: outputs of Oxford cells (§11 of the main pipeline; R7 of the revision analyses), of the cells that execute
the frozen pipeline from a revision notebook (their printout includes §11 when RUN_OXFORD is on), and of the
revision summary cells (they repeat the R7 values). ds004998 outputs (public, CC0) are otherwise kept.
Use --all to clear every output instead. --scrub-paths replaces local Windows/home paths (C:\\Users\\<name>\\…,
/home/<name>/…) in code and outputs with a placeholder. After stripping, the remaining outputs are scanned for strings that should
not be published (Oxford references, local user paths); any hit is listed and the exit code is 1.

    python tools/strip_restricted_outputs.py executed.ipynb notebooks/pd_stn_circuit_sbi_executed.ipynb
    python tools/strip_restricted_outputs.py my_R14_run.ipynb revision/executed/revision_analyses_R14_executed.ipynb
    python tools/strip_restricted_outputs.py --check-only revision/executed/*.ipynb
"""
import argparse, json, re, sys

OXFORD_CELL = re.compile(r"OX_ROOT|OXM|OXD|OXS|read_smr|MATRIX_DBS|_joint_cohort|JPOST_OX|Oxford|RUN_OXFORD|SUMMARY\.items")
LOCAL_PATH = re.compile(r"[A-Za-z]:(\\\\|\\)Users(\\\\|\\)[^\"'\n]*|/home/[^/\s]+/[^\"'\s]*|/Users/[^/\s]+/[^\"'\s]*")
SUSPECT = re.compile(r"(?i)oxford|bndu|OX_ROOT|[A-Z]:\\\\?Users\\\\?|/home/[a-z]|/Users/[A-Za-z]|@[a-z0-9-]+\.[a-z]{2,}")


def text_of(outputs):
    parts = []
    for o in outputs:
        if "text" in o:
            parts.append("".join(o["text"]))
        for k, v in o.get("data", {}).items():
            if k.startswith("text/"):
                parts.append("".join(v))
        if "traceback" in o:
            parts.append("\n".join(o["traceback"]))
    return "\n".join(parts)


def scrub(nb):
    fix = lambda t: LOCAL_PATH.sub("<local path>", t)
    for c in nb["cells"]:
        c["source"] = [fix(x) for x in c["source"]] if isinstance(c["source"], list) else fix(c["source"])
        for o in c.get("outputs", []):
            if "text" in o:
                o["text"] = [fix(x) for x in o["text"]] if isinstance(o["text"], list) else fix(o["text"])
            for k, v in o.get("data", {}).items():
                if k.startswith("text/"):
                    o["data"][k] = [fix(x) for x in v] if isinstance(v, list) else fix(v)
            if "traceback" in o:
                o["traceback"] = [fix(x) for x in o["traceback"]]


def check(nb, name):
    hits = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code" and c.get("outputs"):
            for line in text_of(c["outputs"]).splitlines():
                if SUSPECT.search(line):
                    hits += 1; print(f"  {name} cell {i}: {line[:160]}")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="src dst, or the notebooks to scan with --check-only")
    ap.add_argument("--all", action="store_true", help="clear every output")
    ap.add_argument("--scrub-paths", action="store_true", help="replace local user paths with a placeholder")
    ap.add_argument("--check-only", action="store_true", help="only scan the given notebooks")
    a = ap.parse_args()
    if a.check_only:
        hits = sum(check(json.load(open(p, encoding="utf-8")), p) for p in a.paths)
    else:
        if len(a.paths) != 2:
            ap.error("give a source and a destination notebook")
        src, dst = a.paths
        nb = json.load(open(src, encoding="utf-8")); n = 0
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            s = "".join(c["source"])
            if a.all or (OXFORD_CELL.search(s) and not s.lstrip().startswith(("from pathlib import Path", "import os"))):
                if c.get("outputs"):
                    n += 1
                c["outputs"] = []; c["execution_count"] = None
        if a.scrub_paths:
            scrub(nb)
        json.dump(nb, open(dst, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"cleared outputs of {n} cell(s) → {dst}")
        hits = check(nb, dst)
    print("no suspect strings in the remaining outputs" if hits == 0 else f"{hits} suspect line(s): edit or clear these cells before committing")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
