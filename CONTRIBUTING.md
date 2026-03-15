# Contributing to pyCMF (QuantumLab HCMUS)

Welcome to the `pyCMF` development guide. This repository is maintained by **QuantumLab HCMUS** and is currently for **internal/private use only**. 

This document outlines the coding standards, naming conventions, and testing protocols required for all contributions to `pyCMF`.

---

## 1. File and Module Naming Conventions

To maintain a highly academic and professional codebase, we enforce strict naming conventions for algorithmic variations.

**WHAT NOT TO DO:**
Never use subjective, ambiguous, or temporary suffixes for file names or modules. 
* Do **NOT** use: `_new`, `_faster`, `_mod`, `_v2`, `_test`.

**WHAT TO DO:**
Use standardized suffixes that explicitly describe the underlying mathematical or physical technique. 

| Suffix | Meaning & Usage |
| :--- | :--- |
| (no suffix) | The core, most optimized, and production-ready implementation (e.g., `obmp2.py`, `dfuobmp2.py`). |
| `_slow` | Pedagogical implementations using standard `for`-loops. Highly readable, used as baselines/references. |
| `_einsum` | Vectorized implementations utilizing `numpy.einsum` instead of C/BLAS level optimizations. |
| `_downfold` | Quantum downfolding algorithms (e.g., generating effective Hamiltonians for VQE/Quantum Solvers). |
| `_cas` | Complete Active Space orbital optimization algorithms. |
| `_mom` | Maximum Overlap Method for excited state calculations. |
| `_diis` | Algorithms incorporating DIIS (Direct Inversion in the Iterative Subspace) extrapolation for SCF convergence. |

---

## 2. Code Standards (Linting & Formatting)

We use **Ruff** (configured in `pyproject.toml`) as our unified linter and formatter. All code must pass `ruff check .` and be formatted with `ruff format .` before committing.

### Formatting Rules
* **Line Length**: Set to **120 characters**. This is larger than standard PEP8 to accommodate long tensor equations and complex mathematical formulas.
* **Indentation**: Use spaces (4 spaces per indent).
* **Quotes**: Use single quotes (`'...'`) for standard strings.

### Math-Friendly Spacing (Linting Exceptions)
To keep matrix/tensor operations visually readable (e.g., `matrix[i,j]` or `a = b+c`), we intentionally ignore several whitespace and comment-related warnings. 
Ignored codes include: `E201`, `E202`, `E203`, `E211`, `E221`, `E222`, `E225`, `E226`, `E228`, `E231`, `E241`, `E251`, `E261`, `E262`, `E265`, `E266`, `E301`, `E302`, `E303`, `E305`, `E306`, `E401`, `E402`, `E701`, `E713`, `E721`, `E731`, `E741`, `E275`, `F401`, `F403`, `C901`, `W391`.

---

## 3. Unit Tests & Code Coverage

Robustness is critical for quantum chemistry software. All tests must be placed in the `tests/` directory.

### Coverage Policy
We use `pytest-cov` to measure test coverage, which is configured centrally in `pyproject.toml`.
* **Scope**: Coverage is only measured for the core `src/pycmf` directory.
* **Branch Coverage**: Enabled (`branch = true`) to ensure all `if/else` logic paths are evaluated.

### Ignored Files & Lines
To ensure coverage metrics accurately reflect production code, the following are omitted from coverage reports:
1. **Omitted Files:**
   * `*/tests/*`: The test scripts themselves.
   * `*_slow.py`: Pedagogical unoptimized scripts.
   * `*_einsum.py`: Intermediate vectorized scripts.
   * `*_old.py`: Legacy scripts kept for archival purposes.
2. **Excluded Lines (Do not penalize coverage):**
   * `pragma: no cover`
   * `def __repr__`
   * `raise RuntimeError`
   * `raise NotImplementedError`
   * Temporary execution blocks: `if __name__ == '__main__':` (or similar).

---

## 4. General Programming Guidelines

* **Primary Python Version**: The primary working and testing environment for this project is **Python 3.11**. Ensure all syntax and type hinting are compatible with this version.
* **90/10 Functional/OOP**: Favor pure functions. Object-Oriented Programming (Classes) should be used primarily for high-level wrappers and data management, unless performance critical.
* **90/10 Python/C**: Keep the package lightweight. Write logic in Python first using `numpy`/`scipy` for vectorization. Only port computational hotspots to C or Fortran if Python performance becomes a strict bottleneck.
* **Documentation**: Always provide docstrings for functions and classes outlining inputs, outputs, and the mathematical context of the implemented formulas.