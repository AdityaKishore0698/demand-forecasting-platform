"""Write the synthetic demo dataset to data/synthetic/ (NOT real data; see src/data/synthetic.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic import write_synthetic_csvs  # noqa: E402

if __name__ == "__main__":
    out = write_synthetic_csvs(Path(__file__).resolve().parents[1] / "data" / "synthetic")
    print(f"Wrote SYNTHETIC demo data to {out}")
