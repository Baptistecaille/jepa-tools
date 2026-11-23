import subprocess
import sys

import pytest

OPTIONAL = ("matplotlib", "wandb", "tensorboard", "transformers", "timm", "torchvision")

# Records every attempt to import an optional dependency, then refuses it.
# Works whether or not the package is installed.
PROBE = f"""
import sys
attempts = []
class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {OPTIONAL!r}:
            attempts.append(name)
            raise ImportError(name)
sys.meta_path.insert(0, Blocker())
import jepa_tools, jepa
print(attempts)
"""


def test_import_is_light():
    out = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout


def test_all_names_exist():
    import jepa

    missing = [n for n in jepa.__all__ if not hasattr(jepa, n)]
    assert missing == []
