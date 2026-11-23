"""Recall benchmark as a test. Needs the real embedder, so it only runs when
asked:  HIPPOCAMP_QUALITY=1 pytest -m quality
Synthetic data only (evals/dataset.py), in a throwaway store."""
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.quality


@pytest.mark.skipif(os.environ.get("HIPPOCAMP_QUALITY") != "1", reason="set HIPPOCAMP_QUALITY=1 to run")
def test_recall_meets_thresholds():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
    import recall_bench

    _, fails = recall_bench.run(verbose=False)
    assert not fails, "recall fell below evals/thresholds.json:\n" + "\n".join(fails)
