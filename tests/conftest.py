import os
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from dsin.config import Paths  # noqa: E402
from dsin.preprocessing import run_all  # noqa: E402
from dsin.synthetic import generate_demo_raw  # noqa: E402


@pytest.fixture(scope='session')
def demo_paths(tmp_path_factory):
    """A tiny SYNTHETIC dataset, fully preprocessed in a temp dir (never touches project artifacts)."""
    root = tmp_path_factory.mktemp('dsin_root')
    paths = Paths('demo', root=root)
    paths.ensure()
    generate_demo_raw(paths, n_users=300, n_ads=60, n_cates=10, n_brands=15, n_samples=3000)
    run_all(paths, frac=1.0, chunksize=2000)
    return paths
