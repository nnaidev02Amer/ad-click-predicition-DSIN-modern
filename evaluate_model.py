"""Evaluate a trained DSIN on its held-out test split.   python evaluate_model.py --mode demo"""
import argparse
import json
import sys

from dsin import config as C
from dsin.config import Paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=C.MODES, default='real')
    a = ap.parse_args()
    from dsin.evaluation import evaluate
    paths = Paths(a.mode)
    if not (paths.models / 'dsin.weights.h5').exists():
        print(f'ERROR: no trained model in {paths.models}. Run: python train_model.py --mode {a.mode}')
        return 2
    m = evaluate(paths)
    print(json.dumps(m, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
