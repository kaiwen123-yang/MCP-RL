"""Explicit entry point. Import and plan do not train or collect data."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from task_response.runner import main


if __name__ == "__main__":
    main()
