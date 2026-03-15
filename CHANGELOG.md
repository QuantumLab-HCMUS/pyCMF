# Changelog

All notable changes to the `pyCMF` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), 
and this project adheres to Semantic Versioning.

## [0.1.0] - 2026-03-15 (Major Architectural Refactoring)

This release marks a massive architectural overhaul of the `pyCMF` library. The codebase has been transitioned from a basic two-folder structure into a highly modular, professional package conforming to `pySCF` standards.

### Changed
* **Directory Restructuring:** Completely refactored the original `OBMP2` and `UOBMP2` directories into 5 topic-specific research modules:
  * `OBMP`: Standard Orbital-Optimized MP2 algorithms (incore, restricted, unrestricted, MOM, CAS, Downfolding).
  * `OBDF`: Density Fitting approximations for RAM-reduced OBMP2 calculations.
  * `OBDH`: Orbital-Optimized Double Hybrid DFT methods (e.g., B2PLYP).
  * `KOBMP`: Periodic boundary condition (k-point) OBMP2 calculations.
  * `OBCC`: Orbital-Optimized Coupled Cluster module.
* **File Naming Conventions:** Standardized file names to strictly follow academic and `pySCF` naming conventions. 
  * Removed subjective, ambiguous suffixes (e.g., `_faster`, `_slower`, `_new`, `_mod`, `_ram_reduced`).
  * Adopted clear, descriptive suffixes: `_slow` (pedagogical loops), `_einsum` (vectorized), `_cas` (Complete Active Space), `_mom` (Maximum Overlap Method), `_diis` (SCF convergence), and `_downfold` (Quantum Downfolding).
  * Corrected the misnomer `dfold` to `_downfold` to accurately reflect its quantum downfolding nature rather than density fitting.

### Added
* **Centralized Facade API:** Created a unified `src/pycmf/__init__.py` to handle all user-facing imports. Users can now call state-of-the-art algorithms directly (e.g., `from pycmf import OBMP2, DFUOBMP2`) without needing to know internal file paths.
* **Public APIs (`__all__`):** Implemented strict `__init__.py` files inside each of the 5 modules to define clear public APIs and prevent namespace collisions.
* **Modern Tooling Configurations:** Integrated `ruff` (linter/formatter) and `coverage` configurations directly into `pyproject.toml`. The configurations are specifically tailored to ignore math-heavy whitespace rules and pedagogical `_slow.py` scripts.
* **Project Documentation:** Added standard contribution guidelines (`CONTRIBUTING.md`) outlining the 90/10 Python/C and Functional/OOP philosophies.