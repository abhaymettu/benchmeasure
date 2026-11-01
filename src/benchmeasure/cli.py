import argparse
import json
import sys

import numpy as np
import pandas as pd

from .report import example_path, load_matrix, render, report

EPILOG = """examples:
  benchmeasure demo
  benchmeasure report results.csv
  benchmeasure report results.csv --groups meta.csv --group-col open_weights
  benchmeasure report verified.csv --criterion multilingual.csv
"""


def _groups(path, col, models):
    meta = pd.read_csv(path)
    meta = meta.set_index(meta.columns[0])
    if col not in meta.columns:
        raise SystemExit(f"--group-col {col!r} not in {path}; columns: {', '.join(meta.columns)}")
    g = meta[col].reindex(models)
    if g.isna().any():
        raise SystemExit(f"{int(g.isna().sum())} models in the results have no {col!r} in {path}")
    vals = sorted(g.astype(str).unique())
    if len(vals) != 2:
        raise SystemExit(f"--group-col needs exactly two values, found {vals}")
    focal = "True" if "True" in vals else vals[1]
    return (g.astype(str) == focal).to_numpy(), focal


def main(argv=None):
    ap = argparse.ArgumentParser(prog="benchmeasure", epilog=EPILOG,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description="Reliability, invariance and validity ceilings for benchmark results.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="report on a results file")
    r.add_argument("results", help="wide CSV (model column, one column per task) or long CSV/JSONL (model, task, score)")
    r.add_argument("--groups", help="CSV whose first column is the model and which has --group-col")
    r.add_argument("--group-col", default="group")
    r.add_argument("--criterion", help="second results file on the same models")
    r.add_argument("--top", type=int, nargs="+", default=[10, 20])
    r.add_argument("--json", action="store_true", help="print JSON instead of text")
    sub.add_parser("demo", help="run the report on the bundled SWE-bench Verified matrix")
    args = ap.parse_args(argv)

    if args.cmd == "demo":
        X = load_matrix(example_path())
        meta = pd.read_csv(example_path("swebench_verified_meta.csv")).set_index("submission").loc[X.index]
        crit = load_matrix(example_path("swebench_multilingual.csv.gz"))
        crit.index = [_mini_key(i) for i in crit.index]
        Xm = X.loc[[i for i in X.index if "_mini-v2" in i]]
        Xm.index = [_mini_key(i) for i in Xm.index]
        base = meta.model.fillna(pd.Series(meta.index, index=meta.index)).to_numpy()
        res = report(X, group=meta.os_model.astype(bool).to_numpy(), clusters=base)
        res_v = report(Xm, criterion=crit, top=())
        res["validity"] = res_v["validity"]
        print(render(res, "SWE-bench Verified, 173 public submissions (group 1 = open-weight models;"
                          " criterion = SWE-bench Multilingual, same scaffold)"))
        return 0

    X = load_matrix(args.results)
    group = None
    if args.groups:
        group, focal = _groups(args.groups, args.group_col, X.index)
    crit = load_matrix(args.criterion) if args.criterion else None
    res = report(X, group=group, criterion=crit, top=args.top)
    if args.json:
        res.get("pairs", {}).pop("table", None)
        print(json.dumps(res, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    else:
        print(render(res, args.results))
        if X.attrs.get("dropped_tasks"):
            print(f"\nnote: dropped {X.attrs['dropped_tasks']} tasks with missing scores", file=sys.stderr)
    return 0


def _mini_key(s):
    return s.split("_", 2)[2].replace("-high", "").lower()


if __name__ == "__main__":
    sys.exit(main())
