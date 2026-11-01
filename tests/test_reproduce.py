import filecmp
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_swebench_results_regenerate_exactly(tmp_path):
    shutil.copytree(ROOT / "experiments", tmp_path / "experiments")
    shutil.copytree(ROOT / "src", tmp_path / "src")
    subprocess.run([sys.executable, str(tmp_path / "experiments" / "swebench.py")],
                   check=True, capture_output=True, cwd=tmp_path)
    assert filecmp.cmp(tmp_path / "results" / "swebench.json", ROOT / "results" / "swebench.json",
                       shallow=False)
