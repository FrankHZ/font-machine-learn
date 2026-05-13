from __future__ import annotations

import importlib
import platform
import sys


PACKAGES = [
    "PIL",
    "numpy",
    "scipy",
    "pandas",
    "skimage",
    "sklearn",
    "matplotlib",
    "tqdm",
]


def main() -> int:
    print(f"python: {sys.version.split()[0]} ({platform.platform()})")
    missing = []
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            missing.append(name)
            print(f"{name}: missing")
            continue
        version = getattr(module, "__version__", "installed")
        print(f"{name}: {version}")

    if missing:
        print("\nMissing packages:", ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
