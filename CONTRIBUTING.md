# Contributing to pyCMF (QuantumLab HCMUS)

Welcome! `pyCMF` is a **private** repository for QuantumLab HCMUS. Please follow these quick guidelines to keep our codebase clean, professional, and academically rigorous.

## 1. Naming Conventions

**DO NOT use subjective suffixes:** `_new`, `_faster`, `_mod`, `_v2`, `_test`.
**DO use standard academic suffixes:**

| Suffix | Meaning |
| :--- | :--- |
| *(no suffix)* | Core, optimized production code (e.g., `obmp2.py`, `dfuobmp2.py`). |
| `_slow` | Pedagogical `for`-loop baselines. Highly readable. |
| `_einsum` | Vectorized implementations utilizing `numpy.einsum`. |
| `_downfold` | Quantum downfolding algorithms (effective Hamiltonians). |
| `_cas` | Complete Active Space orbital optimization. |
| `_mom` | Maximum Overlap Method for excited states. |
| `_diis` | DIIS extrapolation for SCF convergence. |

---

## 2. Coding Standards (Ruff)

We use **Ruff** (configured in `pyproject.toml`). Always run `ruff check .` and `ruff format .` before committing your code.

* **Line Length**: 120 characters (extended to fit long tensor equations).
* **Indentation & Quotes**: 4 spaces, single quotes (`'...'`).
* **Math-Friendly Spacing**: Standard whitespace warnings are ignored so you can format matrix operations (e.g., `matrix[i,j]`) for maximum readability.

---

## 3. Testing & Coverage

All tests must be placed in the `tests/` directory. We use `pytest-cov` to measure code coverage.

* **Target**: We only measure coverage for the core `src/pycmf/` folder.
* **Ignored Files**: Pedagogical and legacy scripts (`_slow.py`, `_old.py`) are excluded from coverage scores.
* **Excluded Lines**: Boilerplate code like `pragma: no cover`, `raise RuntimeError`, and `if __name__ == '__main__':` will not penalize your coverage score.

---

## 4. General Guidelines

* **Environment**: The primary testing and working environment is **Python 3.11**.
* **90/10 Functional over OOP**: Favor pure functions. Use classes primarily for data management and wrappers, not deep inheritance trees.
* **90/10 Python over C**: Write logic in Python (`numpy`/`scipy`) first. Port to C/Fortran only if it becomes a strict performance bottleneck.
* **Docstrings**: Always document inputs, outputs, and the mathematical formulas used in your functions.