"""Verify the Python / dependency environment and dataset/model status.  python verify_environment.py"""
import importlib
import platform
import shutil
import sys

from dsin import config as C
from dsin.config import Paths

OK, BAD = 'OK  ', 'FAIL'


def main():
    failures = 0
    print(f'Python {platform.python_version()} ({sys.executable})')
    if sys.version_info[:2] != (3, 9):
        print(BAD, 'expected Python 3.9'); failures += 1
    for name in ['tensorflow', 'keras', 'deepctr', 'numpy', 'pandas', 'sklearn', 'joblib', 'requests', 'tqdm',
                 'streamlit', 'plotly', 'h5py', 'pytest']:
        try:
            m = importlib.import_module(name)
            print(OK, f'{name:12s}', getattr(m, '__version__', '?'))
        except Exception as e:  # noqa
            print(BAD, name, e); failures += 1
    try:
        import tensorflow as tf
        print(OK, 'TF devices:', [d.device_type for d in tf.config.list_physical_devices()])
    except Exception:
        pass
    try:
        from dsin.preprocessing import available_ram_gb
        print('free RAM %.1f GB, free disk %.1f GB' % (available_ram_gb(), shutil.disk_usage(C.ROOT).free / 2 ** 30))
    except Exception as e:  # noqa
        print('could not read RAM:', e)
    for mode in C.MODES:
        p = Paths(mode)
        from dsin.data_validation import check_files
        r = check_files(p)
        print(f'[{mode}] raw data: {"present" if r["ok"] else "MISSING " + str(r["missing"])}'
              f' | processed: {(p.processed / "dsin_input.pkl").exists()} | trained model: {(p.models / "dsin.weights.h5").exists()}')
    print('\nENVIRONMENT', 'OK' if not failures else f'HAS {failures} PROBLEM(S)')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
