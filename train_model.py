"""Train DSIN.   python train_model.py --mode demo --epochs 5     |    python train_model.py --mode real"""
import argparse
import logging
import sys

from dsin import config as C
from dsin.config import ModelParams, Paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=C.MODES, default='real')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--val-fraction', type=float, default=0.1)
    ap.add_argument('--patience', type=int, default=2)
    ap.add_argument('--bias-encoding', action='store_true', help='use bias encoding (paper) instead of positional encoding')
    ap.add_argument('--max-train-rows', type=int, default=None)
    ap.add_argument('--seed', type=int, default=1024)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    from dsin.training import train
    from dsin.evaluation import evaluate
    paths = Paths(a.mode)
    if a.mode == 'demo':
        print('DEMO MODE - SYNTHETIC DATA: results below say nothing about real Taobao performance.')
    train(paths, epochs=a.epochs, batch_size=a.batch_size, learning_rate=a.lr, val_fraction=a.val_fraction,
          patience=a.patience, params=ModelParams(bias_encoding=a.bias_encoding), seed=a.seed,
          max_train_rows=a.max_train_rows)
    m = evaluate(paths)
    print(f"\ntest AUC={m['auc']}  test LogLoss={m['logloss']:.4f}  (n={m['n']}, mode={a.mode})")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except FileNotFoundError as e:
        print('ERROR:', e)
        sys.exit(2)
