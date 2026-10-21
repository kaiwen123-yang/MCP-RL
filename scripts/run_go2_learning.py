"""Entry point; an explicit subcommand is required and defaults never train."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from go2_learning.runner import main

if __name__ == "__main__":
    main()
