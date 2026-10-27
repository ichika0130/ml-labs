"""Print the Python version and the version of every package in requirements.txt."""
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parents[2] / "requirements.txt"


def main() -> None:
    print(f"Python {sys.version.split()[0]} ({platform.platform()})")
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("==")[0]
        try:
            print(f"{name}=={version(name)}")
        except PackageNotFoundError:
            print(f"{name}: NOT INSTALLED")


if __name__ == "__main__":
    main()
