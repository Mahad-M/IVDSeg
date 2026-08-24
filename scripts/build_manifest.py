#!/usr/bin/env python3
"""Create the checked IVDM3Seg subject manifest."""

from pathlib import Path
import sys

# The project is intentionally a lightweight application rather than an
# installed distribution, so make its root importable when this file is run.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivdseg.manifest import main


if __name__ == "__main__":
    main()
