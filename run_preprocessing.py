"""Run the DSIN preprocessing pipeline (steps 0-2).

    python run_preprocessing.py --mode real            # needs the Taobao CSVs in raw_data/
    python run_preprocessing.py --mode demo --make-demo # generate SYNTHETIC data first, then process it
"""
import argparse
import json
import logging
import sys

from dsin import config as C
from dsin.config import Paths
from dsin.data_validation import check_files, save_report, validate_dataset
from dsin.preprocessing import available_ram_gb, run_all
from dsin.synthetic import generate_demo_raw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=C.MODES, default='real')
    ap.add_argument('--frac', type=float, default=None,
                    help=f'fraction of users to sample (default {C.FRAC} real / 1.0 demo). On an 8 GB laptop use <=0.05 for real data.')
    ap.add_argument('--make-demo', action='store_true', help='(demo mode) generate synthetic raw files first')
    ap.add_argument('--force', action='store_true', help='recompute even if outputs exist')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--chunksize', type=int, default=5_000_000)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

    paths = Paths(a.mode)
    paths.ensure()
    if a.mode == 'demo' and (a.make_demo or not (paths.raw / 'raw_sample.csv').exists()):
        print('DEMO MODE - SYNTHETIC DATA: generating', paths.raw)
        print(generate_demo_raw(paths))
    print(f'free RAM: {available_ram_gb():.1f} GB')

    rep = check_files(paths)
    if not rep['ok']:
        print('\nDATASET NOT READY:', json.dumps(rep['files'], indent=2))
        print('\nMissing:', rep['missing'])
        print('Download from https://tianchi.aliyun.com/dataset/dataDetail?dataId=56 and extract into', paths.raw)
        print('(no synthetic data was substituted; use --mode demo --make-demo for the clearly-labelled demo)')
        return 2
    rep = validate_dataset(paths)
    save_report(rep, paths.results / 'dataset_validation.json')
    print('dataset validation:', 'OK' if rep['ok'] else f"PROBLEMS {rep['problems']}")
    if a.validate_only:
        return 0 if rep['ok'] else 1
    if not rep['ok']:
        return 1
    frac = a.frac if a.frac is not None else (C.FRAC if a.mode == 'real' else 1.0)
    run_all(paths, frac=frac, force=a.force, chunksize=a.chunksize)
    print('preprocessing complete ->', paths.processed)
    return 0


if __name__ == '__main__':
    sys.exit(main())
