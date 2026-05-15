"""
Root-level entrypoint wrapper.

Purpose:
- Keep `evaluate_labeled_flows_csv.py` callable from project root (`demo/`)
- Reuse the maintained implementation in `scripts/tools/evaluate_labeled_flows_csv.py`
"""

from scripts.tools.evaluate_labeled_flows_csv import *  # noqa: F401,F403
from scripts.tools.evaluate_labeled_flows_csv import main as _main


if __name__ == "__main__":
    _main()

