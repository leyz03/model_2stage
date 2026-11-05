#!/usr/bin/env python3
"""
prune.py — should a scaling run be skipped because a smaller run of the same kind already failed?

    python3 model_2stage/scaling/prune.py MODE N T NSCEN INNER JOB [JOB ...]
    exit 0 + message on stdout: skip;  exit 1: run it

A run (mode, n, T, K_s) is dominated by a recorded run (same mode, same inner for bnb /
oracle, same structure and levels) with n' <= n, T' <= T, K_s' <= K_s that failed:
  bnb     status time_limit, or no incumbent (value null)
  ef      status not OPTIMAL
  oracle  a formulation in INNER (or "pre" when INNER is empty) stopped by its MILP time limit
Records are read from results/<JOB>/scaling.jsonl for every JOB given (the current job and
any earlier jobs listed in PRUNE_JOBS).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def failed(r, mode, inner):
    if r["mode"] != mode:
        return False
    if mode == "bnb":
        return r.get("inner") == inner and (r["status"] != "optimal" or r.get("value") is None)
    if mode == "ef":
        return r["status"] != "OPTIMAL"
    forms = inner.split("/") if inner else ["pre"]
    return r.get("form") in forms and (r.get("not_optimal") or 0) > 0


def main():
    mode, n, T, ns, inner = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    if mode == "bnb" and not inner:
        inner = "pre"
    for job in sys.argv[6:]:
        f = os.path.join(HERE, "..", "results", job, "scaling.jsonl")
        if not os.path.exists(f):
            continue
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("structure", "type") != "type" or r.get("levels", 3) != 3:
                continue
            if failed(r, mode, inner) and r["n"] <= n and r["T"] <= T and r["nscen"] <= ns:
                print("skip: dominated by failed %s n=%d T=%d K_s=%d (%s, job %s)" % (
                    mode, r["n"], r["T"], r["nscen"], r.get("status", r.get("form")), job))
                return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
