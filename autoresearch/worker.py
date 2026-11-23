"""Subprocess entry point: python -m autoresearch.worker <experiment> <params.json> <outdir>"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

from .core import ROOT, dump_json, load_json


def main(argv):
    name, params_path, outdir = argv[1], argv[2], Path(argv[3])
    sys.path.insert(0, str(ROOT))
    from .experiments import REGISTRY
    try:
        res = REGISTRY[name]["fn"](load_json(params_path), outdir)
        dump_json(outdir / "result.json", res)
    except BaseException:                                   # noqa: BLE001 - report everything to the parent
        (outdir / "error.txt").write_text(traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
