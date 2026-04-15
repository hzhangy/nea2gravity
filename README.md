# From Holographic Dimension to Tension-Induced Gravity

This repository contains the full source code, data, and LaTeX source for the paper:

**Zhang Yu & AI Collaboration Staff.** *From Holographic Dimension to Tension-Induced Gravity: Observational Evidence for Dark Matter as a Projection Artifact* (2026).

## Repository Structure

- `paper/` – LaTeX source and figures for the main paper and the three theoretical papers.
- `code/` – Python scripts for numerical experiments and galaxy fitting.
- `data/` – SPARC rotation curve data (table2.txt) and galaxy parameters (Table1.mrt).
- `results/` – Fitting results and audit CSV files.
- `book/` – Introduction to the book *Being is Time* (Yixin Jing) and cover/copyright pages.

## Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib, pandas

Install with `pip install -r requirements.txt` (if provided).

## Running the Code

See individual scripts for usage. A typical workflow:

1. Run `code/nea_galaxy_fit.py` to fit galaxies.
2. Run `code/final_holo_audit.py` to compute the holographic invariant.
3. Run `code/nea_extreme_tests.py` for consistency checks.

Output figures and CSV files will appear in the working directory; move them to `paper/figures/` and `results/` as needed.

## Citation

If you use this work, please cite:
