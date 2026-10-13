"""Run every experiment, save results/results.json, figures in results/figures and models in models/.

Usage:
    python scripts/run_all.py                    # everything (about 15-25 minutes on a laptop)
    python scripts/run_all.py --only supervised improvements
"""

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hitpredict import config, data, plotting  # noqa: E402
from hitpredict.experiments import (ablation, evaluation_protocol, improvements,  # noqa: E402
                                    model_comparison, overfitting, supervised, temporal,
                                    unsupervised)

FAMILIES = {
    "supervised": supervised.run,
    "unsupervised": unsupervised.run,
    "overfitting": overfitting.run,
    "ablation": ablation.run,
    "temporal": temporal.run,
    "evaluation_protocol": evaluation_protocol.run,
    "model_comparison": model_comparison.run,
    "improvements": improvements.run,
}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(FAMILIES), help="run a subset of experiments")
    args = ap.parse_args()

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    if not data.data_present():
        data.download()
    plotting.setup()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    if config.RESULTS_JSON.exists() and args.only:
        results = json.loads(config.RESULTS_JSON.read_text())
    results["meta"] = {"seed": config.SEED, "python": platform.python_version(),
                       "run_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    results["dataset"] = _jsonable(data.summary())

    for name in args.only or FAMILIES:
        t0 = time.time()
        print(f"[{name}] running ...", flush=True)
        results[name] = _jsonable(FAMILIES[name]())
        results[name]["seconds"] = round(time.time() - t0, 1)
        print(f"[{name}] done in {results[name]['seconds']} s", flush=True)
        config.RESULTS_JSON.write_text(json.dumps(results, indent=1))
    print(f"Results written to {config.RESULTS_JSON}")


if __name__ == "__main__":
    main()
